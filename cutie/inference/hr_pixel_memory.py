class HRPixelMemory:
    """Small cache for HR keyframe pixel features used by CReFF."""

    def __init__(self, max_keep: int = 2):
        self.max_keep = max_keep
        self.store = {}
        self.last_gop_idx = None

    def clear(self) -> None:
        self.store.clear()
        self.last_gop_idx = None

    def add(self, gop_idx: int, ti: int, pix_feat):
        self.store[gop_idx] = (ti, pix_feat.detach())
        self.last_gop_idx = gop_idx
        while len(self.store) > self.max_keep:
            oldest = min(self.store.keys())
            del self.store[oldest]

    def get_latest(self):
        if self.last_gop_idx is None:
            return None
        return self.store[self.last_gop_idx][1]
