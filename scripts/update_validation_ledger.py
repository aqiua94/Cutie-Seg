from pathlib import Path


CSV_ROW_OLD = (
    '2026-05-30 17:41:00,CReFF-only,output/creff_davis_creff_only_b32/'
    'compressed_davis_dataloader_step_500.pth,500,"batch_size=32; trainable=CReFF only; '
    'compressed DAVIS; GOP12; freeze decoder; lr_scale=0.5",DAVIS-2017 val,full val,'
    '"compressed_root=data/DAVIS/2017/trainval/compressed/3M-GOP12; gop_length=12; '
    'size=480; amp; skip_first; eval_size=480",full-val J-only,'
    '0.6067702067173109,,,1969,output/creff_davis_eval_creff_only_b32_step500/'
    'Annotations,output/creff_davis_eval_creff_only_b32_step500/j_metrics_480.csv,'
    '"full prediction masks generated; full J&F script was too slow, so only fast J was recorded"'
)

CSV_ROWS_NEW = (
    '2026-05-30 23:30:00,CReFF-only,output/creff_davis_creff_only_b32/'
    'compressed_davis_dataloader_step_500.pth,500,"batch_size=32; trainable=CReFF only; '
    'compressed DAVIS; GOP12; freeze decoder; lr_scale=0.5",DAVIS-2017 val,full val,'
    '"compressed_root=data/DAVIS/2017/trainval/compressed/3M-GOP12; gop_length=12; '
    'size=480; amp; skip_first; eval_size=480; jf_workers=12",full-val J&F,'
    '0.6067702067173109,0.68885548881533,0.6478128477663204,1969,'
    'output/creff_davis_eval_creff_only_b32_step500/Annotations,'
    'output/creff_davis_eval_creff_only_b32_step500/jf_metrics_480_parallel.csv,'
    '"full prediction masks generated; J&F computed with parallel evaluator"\n'
    '2026-05-30 23:36:00,Official Cutie no-CReFF baseline,weights/cutie-base-mega.pth,'
    '0,"official cutie-base-mega; no CReFF; no training",DAVIS-2017 val,full val,'
    '"compressed_root=data/DAVIS/2017/trainval/compressed/3M-GOP12; gop_length=1; '
    'disable_creff; size=480; amp; skip_first; eval_size=480; jf_workers=12",'
    'full-val J&F,0.8243244680191995,0.9016387796834019,0.8629816238513006,1969,'
    'output/baseline_official_no_creff_gop1_fullval/Annotations,'
    'output/baseline_official_no_creff_gop1_fullval/jf_metrics_480_parallel.csv,'
    '"baseline on HEVC decoded frames; gop1/no-CReFF forces ordinary HR encode path"'
)

MD_ROW_OLD = (
    '| 2026-05-30 17:41 | CReFF-only | `output/creff_davis_creff_only_b32/'
    'compressed_davis_dataloader_step_500.pth` | 500 | DAVIS-2017 val | GOP12, '
    'size 480, AMP, skip first, eval-size 480 | 0.6068 |  |  | 1969 | '
    '`output/creff_davis_eval_creff_only_b32_step500/j_metrics_480.csv` |'
)

MD_ROWS_NEW = (
    '| 2026-05-30 23:30 | CReFF-only | `output/creff_davis_creff_only_b32/'
    'compressed_davis_dataloader_step_500.pth` | 500 | DAVIS-2017 val | GOP12, '
    'size 480, AMP, skip first, eval-size 480, J&F workers 12 | 0.6068 | 0.6889 | '
    '0.6478 | 1969 | `output/creff_davis_eval_creff_only_b32_step500/'
    'jf_metrics_480_parallel.csv` |\n'
    '| 2026-05-30 23:36 | Official Cutie no-CReFF baseline | `weights/'
    'cutie-base-mega.pth` | 0 | DAVIS-2017 val | GOP1, no-CReFF, size 480, AMP, '
    'skip first, eval-size 480, J&F workers 12 | 0.8243 | 0.9016 | 0.8630 | 1969 | '
    '`output/baseline_official_no_creff_gop1_fullval/jf_metrics_480_parallel.csv` |'
)


def replace_once(text: str, old: str, new: str) -> str:
    if new in text:
        return text
    if old not in text:
        raise RuntimeError(f'Could not find ledger text to replace: {old[:80]}')
    return text.replace(old, new, 1)


def main():
    csv_path = Path('output/validation_ledger.csv')
    md_path = Path('output/validation_ledger.md')

    csv_text = replace_once(csv_path.read_text(), CSV_ROW_OLD, CSV_ROWS_NEW)
    csv_path.write_text(csv_text)

    md_text = replace_once(md_path.read_text(), MD_ROW_OLD, MD_ROWS_NEW)
    md_text = md_text.replace(
        'Empty `F` / `J&F` means only the fast J-only evaluator was completed. '
        'The full boundary-F evaluator was too slow in the current environment.',
        'Empty `F` / `J&F` means only the fast J-only evaluator was completed. '
        'Full J&F now uses `scripts/evaluate_davis_jf_parallel.py`.',
    )
    md_text = md_text.replace(
        'Default compressed validation uses `data/DAVIS/2017/trainval/compressed/3M-GOP12`, '
        'so `gop_length=12` unless explicitly stated otherwise.',
        'Default compressed validation uses `data/DAVIS/2017/trainval/compressed/3M-GOP12`, '
        'so `gop_length=12` unless explicitly stated otherwise.\n'
        '- The official no-CReFF baseline uses HEVC decoded frames from the same compressed root, '
        'with `gop_length=1` and `--disable-creff`; this gives the current Stage 2 ceiling under '
        'the local evaluator: J&F 0.8630.',
    )
    md_path.write_text(md_text)


if __name__ == '__main__':
    main()
