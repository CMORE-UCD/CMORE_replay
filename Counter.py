import cv2 as cv
import numpy as np

from Block import Block


class Counter:

    def __init__(self, target_zone, frame, threshold: float = 0.25):
        self.target_zone = target_zone
        self.threshold = threshold
        self.counter = 0
        self.over_count_block = False
        self.state: str | None = None
        self.counted_ids: set[int] = set()
        self.curr_target_tids: set[int] = set()
        self.coords_last_block = None
        self.target_block_registry: dict[int, Block] = {}
        self.height, self.width, _ = frame.shape

    def has_movement(self, block: Block) -> bool:
        """True if block center moved at least `threshold` fraction of its bbox size."""
        if len(block.last5box) < 2:
            return False

        centers = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in block.last5box]
        sizes   = [(b[2] - b[0], b[3] - b[1]) for b in block.last5box]
        ref = max((sum(s[i] for s in sizes) / len(sizes) for i in (0, 1)))

        if ref == 0:
            return False

        dx = centers[-1][0] - centers[0][0]
        dy = centers[-1][1] - centers[0][1]
        displacement = (dx ** 2 + dy ** 2) ** 0.5
        return self.threshold <= (displacement / ref) <= 3

    def update_curr_blocks_in_target(self, tracked: np.ndarray | None = None):
        if tracked is None or len(tracked) == 0:
            return
        for row in tracked:
            x1, y1, x2, y2, tid = row[0], row[1], row[2], row[3], int(row[4])
            px1, py1 = int(x1 * self.width), int(y1 * self.height)
            px2, py2 = int(x2 * self.width), int(y2 * self.height)

            if self._block_in_target_zone([px1, py1, px2, py2]):
                self.curr_target_tids.add(tid)
                if tid not in self.target_block_registry:
                    self.target_block_registry[tid] = Block(id=tid)
                self.target_block_registry[tid].update([x1, y1, x2, y2])

    def clear_stale_target_blocks(self):
        for tid in self.target_block_registry:
            if tid not in self.curr_target_tids:
                self.target_block_registry[tid].update([0, 0, 0, 0])
        self.curr_target_tids.clear()

    def update_counter(self):
        counter_changed = False
        for tid, blk in self.target_block_registry.items():
            if self.over_count_block:
                continue
            if not self.has_movement(blk) or tid in self.counted_ids:
                continue
            counter_changed = True
            self.counter += 1
            self.counted_ids.add(tid)
            self.coords_last_block = blk.last5box[-1]
            self.over_count_block = True

        if not counter_changed:
            self.coords_last_block = None

    def reset_states(self, next_state: str):
        if self.state != 'crossed' and next_state == 'crossed':
            self.over_count_block = False
        self.state = next_state

    def update_all(self, frame_state: str, tracked: np.ndarray | None = None):
        self.update_curr_blocks_in_target(tracked)
        self.clear_stale_target_blocks()
        self.update_counter()
        self.reset_states(frame_state)

    def _block_in_target_zone(self, bbox_px: list) -> bool:
        """True if any corner of the pixel bbox lies inside the target zone trapezoid."""
        x1, y1, x2, y2 = bbox_px
        poly = np.array([
            self.target_zone["top_left"],
            self.target_zone["top_right"],
            self.target_zone["bottom_right"],
            self.target_zone["bottom_left"],
        ], dtype=np.float32)
        corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
        return any(cv.pointPolygonTest(poly, pt, measureDist=False) >= 0 for pt in corners)
