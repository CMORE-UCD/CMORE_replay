from Block import Block
import numpy as np
import cv2 as cv

class Counter:
    counter = 0
    height = 0
    width = 0
    active_counting_state = False
    crossed_back = False
    counted_ids = set()
    curr_target_tids = set()
    coords_last_block = []
    target_block_registry: dict[int, Block] = {}   # track_id -> Block
    target_zone = None
    threshold = 0

    def __init__(self, target_zone, frame, threshold: float = 0.25): 
        self.target_zone = target_zone
        self.set_dimensions(frame)
        self.threshold = threshold

    def has_movement(self, block: Block) -> bool:
        """Return True if the block's center has moved by at least `threshold` fraction
        of the bbox dimensions across its last-5 history.

        Movement is measured as the Chebyshev-style max displacement of the center
        relative to the mean bbox size (width or height), so the threshold is
        scale-invariant.

        Args:
            block:     Block instance with last5box entries [x1, y1, x2, y2] normalized.
            threshold: Minimum fractional displacement to count as movement (default 0.25).

        Returns:
            bool
        """
        if len(block.last5box) < 2:
            return False

        centers = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in block.last5box]
        sizes   = [(b[2] - b[0], b[3] - b[1]) for b in block.last5box]

        mean_w = sum(s[0] for s in sizes) / len(sizes)
        mean_h = sum(s[1] for s in sizes) / len(sizes)
        ref    = max(mean_w, mean_h)  # single scale reference

        if ref == 0:
            return False

        first_cx, first_cy = centers[0]
        last_cx,  last_cy  = centers[-1]

        displacement = ((last_cx - first_cx) ** 2 + (last_cy - first_cy) ** 2) ** 0.5
        return 3 >= (displacement / ref) >= self.threshold
    
    def update_curr_blocks_in_target(self, tracked: 'np.ndarray | None' = None):
        if tracked is not None and len(tracked) > 0:
            for row in tracked:
                x1, y1, x2, y2, tid = row[0], row[1], row[2], row[3], int(row[4])
                px1, py1 = int(x1 * self.width), int(y1 * self.height)
                px2, py2 = int(x2 * self.width), int(y2 * self.height)

                if self.tracker_block_in_target_zone([px1, py1, px2, py2]):
                    self.curr_target_tids.add(tid)

                    if tid not in self.target_block_registry:
                        self.target_block_registry[tid] = Block(id=tid)
                    
                    self.target_block_registry[tid].update([x1, y1, x2, y2])

    def update_prev_blocks_in_target(self):
        for tid in self.target_block_registry:
            if tid not in self.curr_target_tids:
                self.target_block_registry[tid].update([0, 0, 0, 0])
        
        self.curr_target_tids.clear()

    def update_counter(self):
        counter_changed = False
        for tid, blk in self.target_block_registry.items():
            if self.active_counting_state:
                continue
            if not self.has_movement(blk):
                continue
            if tid in self.counted_ids:
                continue

            counter_changed = True
            self.counter += 1
            self.counted_ids.add(tid)
            self.coords_last_block = blk.last5box[-1]
            self.active_counting_state = True
            self.crossed_back = False
        
        if not counter_changed:
            self.coords_last_block = None
    
    def set_dimensions(self, frame):
        annotated = frame.copy()
        self.height, self.width, _ = annotated.shape

    def reset_states(self, frame_state_result):
        if self.active_counting_state and self.crossed_back and frame_state_result == 'crossed':
            self.active_counting_state = False

        if frame_state_result == 'crossedBack':
            self.crossed_back = True

    def update_all(self, frame_result, tracked: 'np.ndarray | None' = None):
        self.update_curr_blocks_in_target(tracked)
        self.update_prev_blocks_in_target()
        self.update_counter()
        self.reset_states(frame_result)

    def tracker_block_in_target_zone(self, scaled_norm):
        """Return True if a cgRect bounding box overlaps the x and y-range of the delimiter line.

        Args:
            scaled_norm: [px1, py1, px2, py2] — normalized and scaled Vision coord.

        Returns:
            bool
        """
        block_x1, block_y1, block_x2, block_y2 = scaled_norm

        poly = self.trapezoid_polygon()

        # Check all four corners of the block bbox
        corners = [
            (block_x1, block_y1),  # top-left
            (block_x2, block_y1),  # top-right
            (block_x1, block_y2),  # bottom-left
            (block_x2, block_y2),  # bottom-right
        ]
        for pt in corners:
            # >= 0 means inside or on the edge
            if cv.pointPolygonTest(poly, pt, measureDist=False) >= 0:
                return True

        return False
    
    def trapezoid_polygon(self):
        """Return the trapezoid as an ordered numpy contour for cv.pointPolygonTest.
        Order: top-left → top-right → bottom-right → bottom-left (clockwise).
        """
        return np.array([
            self.target_zone["top_left"],
            self.target_zone["top_right"],
            self.target_zone["bottom_right"],
            self.target_zone["bottom_left"],
        ], dtype=np.float32)