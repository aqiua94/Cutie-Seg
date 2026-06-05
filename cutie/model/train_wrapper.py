import logging
from omegaconf import DictConfig
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict

from einops.layers.torch import Rearrange
from cutie.model.cutie import CUTIE
from cutie.model.mv_warp import warp_feature

log = logging.getLogger()


class CutieTrainWrapper(CUTIE):
    def __init__(self, cfg: DictConfig, stage_cfg: DictConfig):
        super().__init__(cfg, single_object=(stage_cfg.num_objects == 1))

        self.sensory_dim = cfg.model.sensory_dim
        self.seq_length = stage_cfg.seq_length
        self.num_ref_frames = stage_cfg.num_ref_frames
        self.deep_update_prob = stage_cfg.deep_update_prob
        self.use_amp = stage_cfg.amp
        self.move_t_out_of_batch = Rearrange('(b t) c h w -> b t c h w', t=self.seq_length)
        self.move_t_from_batch_to_volume = Rearrange('(b t) c h w -> b c t h w', t=self.seq_length)
        self.lr_scale = stage_cfg.get('lr_scale', cfg.model.get('lr_scale', 0.5))
        self.freeze_decoder_for_fst = stage_cfg.get('freeze_decoder_for_fst', False)
        self.trainable_mode = stage_cfg.get('trainable_mode', 'creff_only')
        self.feat_distill_type = stage_cfg.get('feat_distill_type', 'mse_creff_to_hr')
        if self.use_creff and self.freeze_decoder_for_fst:
            for param in self.parameters():
                param.requires_grad = False
            if self.trainable_mode == 'creff_only':
                for param in self.creff.parameters():
                    param.requires_grad = True
            elif self.trainable_mode == 'encoder_layer23_task_fst':
                for name, param in self.named_parameters():
                    if name.startswith(('pixel_encoder.layer2.', 'pixel_encoder.layer3.')):
                        param.requires_grad = True
            else:
                raise ValueError(f'Unknown trainable_mode: {self.trainable_mode}')

    @staticmethod
    def _cosine_feature_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        if student.shape[-2:] != teacher.shape[-2:]:
            student = F.interpolate(student,
                                    size=teacher.shape[-2:],
                                    mode='bilinear',
                                    align_corners=False)
        student = F.normalize(student.float(), dim=1)
        teacher = F.normalize(teacher.float(), dim=1)
        return 1.0 - (student * teacher).sum(dim=1).mean()


    def _encode_compressed_sequence(self, frames: torch.Tensor, mv_seq: torch.Tensor,
                                    is_i_frame: torch.Tensor):
        b, seq_length = frames.shape[:2]
        ms_feat_all = []
        pix_feat_all = []
        feat_losses = []
        ref_pix_feat_hr = None

        for ti in range(seq_length):
            is_i = bool(is_i_frame[:, ti].all().item())
            if is_i or ref_pix_feat_hr is None:
                ms_t, pix_t = self.encode_image(frames[:, ti])
                ref_pix_feat_hr = pix_t.detach()
            else:
                with torch.no_grad():
                    teacher_ms_t, teacher_pix_t = self.encode_image(frames[:, ti])

                image_lr = F.interpolate(frames[:, ti],
                                         scale_factor=self.lr_scale,
                                         mode='bilinear',
                                         align_corners=False)
                warped_ref = warp_feature(ref_pix_feat_hr, mv_seq[:, ti])
                ms_t, pix_lr = self.encode_image(image_lr)
                pix_t = self.creff(warped_ref, pix_lr)
                ms_t = [
                    F.interpolate(feat,
                                  size=teacher_feat.shape[-2:],
                                  mode='bilinear',
                                  align_corners=False)
                    for feat, teacher_feat in zip(ms_t, teacher_ms_t)
                ]
                if self.feat_distill_type == 'mse_creff_to_hr':
                    feat_losses.append(F.mse_loss(pix_t.float(), teacher_pix_t.float()))
                elif self.feat_distill_type == 'cosine_lr_to_hr':
                    feat_losses.append(self._cosine_feature_loss(pix_lr, teacher_pix_t))
                elif self.feat_distill_type == 'cosine_creff_to_hr':
                    feat_losses.append(self._cosine_feature_loss(pix_t, teacher_pix_t))
                else:
                    raise ValueError(f'Unknown feat_distill_type: {self.feat_distill_type}')

            ms_feat_all.append(ms_t)
            pix_feat_all.append(pix_t)

        ms_feat = [torch.stack([ms_t[level] for ms_t in ms_feat_all], dim=1) for level in range(len(ms_feat_all[0]))]
        pix_feat = torch.stack(pix_feat_all, dim=1)
        if feat_losses:
            feat_distill_loss = torch.stack(feat_losses).mean()
        else:
            feat_distill_loss = frames.new_tensor(0.0)
        return ms_feat, pix_feat, feat_distill_loss

    def forward(self, data: Dict):
        out = {}
        frames = data['rgb']
        first_frame_gt = data['first_frame_gt'].float()
        b, seq_length = frames.shape[:2]
        num_filled_objects = [o.item() for o in data['info']['num_objects']]
        max_num_objects = max(num_filled_objects)
        first_frame_gt = first_frame_gt[:, :, :max_num_objects]
        selector = data['selector'][:, :max_num_objects].unsqueeze(2).unsqueeze(2)

        num_objects = first_frame_gt.shape[2]
        out['num_filled_objects'] = num_filled_objects

        def get_ms_feat_ti(ti):
            return [f[:, ti] for f in ms_feat]

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            use_compressed = self.use_creff and 'mv' in data and 'is_i_frame' in data
            if use_compressed:
                ms_feat, pix_feat, feat_distill_loss = self._encode_compressed_sequence(
                    frames, data['mv'], data['is_i_frame'])
                out['feat_distill_loss'] = feat_distill_loss
                with torch.cuda.amp.autocast(enabled=False):
                    keys, shrinkages, selections = self.transform_key(ms_feat[0].flatten(0, 1).float())
            else:
                frames_flat = frames.view(b * seq_length, *frames.shape[2:])
                ms_feat, pix_feat = self.encode_image(frames_flat)
                with torch.cuda.amp.autocast(enabled=False):
                    keys, shrinkages, selections = self.transform_key(ms_feat[0].float())

                # ms_feat: tuples of (B*T)*C*H*W -> B*T*C*H*W
                ms_feat = [self.move_t_out_of_batch(f) for f in ms_feat]
                pix_feat = self.move_t_out_of_batch(pix_feat)

            # keys/shrinkages/selections: (B*T)*C*H*W -> B*C*T*H*W
            h, w = keys.shape[-2:]
            keys = self.move_t_from_batch_to_volume(keys)
            shrinkages = self.move_t_from_batch_to_volume(shrinkages)
            selections = self.move_t_from_batch_to_volume(selections)

            # zero-init sensory
            sensory = torch.zeros((b, num_objects, self.sensory_dim, h, w), device=frames.device)
            msk_val, sensory, obj_val, _ = self.encode_mask(frames[:, 0], pix_feat[:, 0], sensory,
                                                            first_frame_gt[:, 0])
            masks = first_frame_gt[:, 0]

            # add the time dimension
            msk_values = msk_val.unsqueeze(3)  # B*num_objects*C*T*H*W
            obj_values = obj_val.unsqueeze(
                2) if obj_val is not None else None  # B*num_objects*T*Q*C

            for ti in range(1, seq_length):
                if ti <= self.num_ref_frames:
                    ref_msk_values = msk_values
                    ref_keys = keys[:, :, :ti]
                    ref_shrinkages = shrinkages[:, :, :ti] if shrinkages is not None else None
                else:
                    # pick num_ref_frames random frames
                    # this is not very efficient but I think we would
                    # need broadcasting in gather which we don't have
                    ridx = [torch.randperm(ti)[:self.num_ref_frames] for _ in range(b)]
                    ref_msk_values = torch.stack(
                        [msk_values[bi, :, :, ridx[bi]] for bi in range(b)], 0)
                    ref_keys = torch.stack([keys[bi, :, ridx[bi]] for bi in range(b)], 0)
                    ref_shrinkages = torch.stack([shrinkages[bi, :, ridx[bi]] for bi in range(b)],
                                                 0)

                # Segment frame ti
                readout, aux_input = self.read_memory(keys[:, :, ti], selections[:, :,
                                                                                 ti], ref_keys,
                                                      ref_shrinkages, ref_msk_values, obj_values,
                                                      pix_feat[:, ti], sensory, masks, selector)
                aux_output = self.compute_aux(pix_feat[:, ti], aux_input, selector)
                sensory, logits, masks = self.segment(get_ms_feat_ti(ti),
                                                      readout,
                                                      sensory,
                                                      selector=selector)
                # remove background
                masks = masks[:, 1:]

                # No need to encode the last frame
                if ti < (self.seq_length - 1):
                    deep_update = np.random.rand() < self.deep_update_prob
                    msk_val, sensory, obj_val, _ = self.encode_mask(frames[:, ti],
                                                                    pix_feat[:, ti],
                                                                    sensory,
                                                                    masks,
                                                                    deep_update=deep_update)
                    msk_values = torch.cat([msk_values, msk_val.unsqueeze(3)], 3)
                    obj_values = torch.cat([obj_values, obj_val.unsqueeze(2)],
                                           2) if obj_val is not None else None

                out[f'masks_{ti}'] = masks
                out[f'logits_{ti}'] = logits
                out[f'aux_{ti}'] = aux_output

        return out
