import sys
import cv2 as cv
import numpy as np
import pandas as pd
import boxmot
from abc import abstractmethod
from pathlib import Path

from Config import *
from Counter import Counter


class Detection:

    def __init__(self, args):
        self.args = args
        self.video_path = args.video_path
        self.block_tracked: dict[int, np.ndarray | None] = {}
        self.tracker_head = -1
        self.frame: np.ndarray | None = None

        self._setup_cap()

        timeTag = Path(self.video_path).stem.split('_')[2]
        self.df = pd.read_json(f"CMORE_Results_{timeTag}.json")

        _, self.frame = self.cap.read()

        self.target_zone = self._compute_target_zone()
        self.counter = Counter(self.target_zone, self.frame, args.mvmt_threshold)
        self.tracker = self._make_tracker(args.tracker, args.track_buffer)

    def _setup_cap(self):
        self.cap: cv.VideoCapture = cv.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            print("Error: Could not open video file")
            sys.exit(1)

        fps = self.cap.get(cv.CAP_PROP_FPS)
        self.tolerance = 10
        self.frame_count = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))
        self.img_width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.img_height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.current_frame = 0
        print(f"Video loaded. FPS: {fps}, Total frames: {self.frame_count}")

    def _make_tracker(self, name: str, track_buffer: int = 30):
        options = {
            'bytetrack':  lambda: boxmot.ByteTrack(track_buffer=track_buffer, min_hits=1),
            'ocsort':     lambda: boxmot.OcSort(min_hits=1),
            'sfsort':     lambda: boxmot.SFSORT(min_hits=1),
            'boosttrack': lambda: boxmot.BoostTrack(min_hits=1),
        }
        if name not in options:
            raise ValueError(f"Unknown tracker '{name}'. Choose from: {MOTION_TRACKERS}")
        return options[name]()

    def _advance_tracker_to(self, target_idx: int):
        """Feed df rows tracker_head+1..target_idx into the tracker, caching results."""
        if target_idx <= self.tracker_head:
            return
        w, h = self.img_width, self.img_height
        frame = self.frame if self.frame is not None else np.zeros((h, w, 3), dtype=np.uint8)
        for idx in range(self.tracker_head + 1, target_idx + 1):
            row = self.df.iloc[idx]
            block_dets = row.get('blockDetections') or []
            if not isinstance(block_dets, list):
                block_dets = []
            boxes = [self._cgrect_to_norm_xyxy(bd['boundingBox']) for bd in block_dets]
            valid = [(b, bd) for b, bd in zip(boxes, block_dets) if b is not None]
            if valid:
                xyxy = np.array([b for b, _ in valid], dtype=np.float32)
                xyxy[:, [0, 2]] *= w
                xyxy[:, [1, 3]] *= h
                conf = np.array([float(bd.get('confidence', 1.0)) for _, bd in valid], dtype=np.float32)
                cls = np.zeros(len(valid), dtype=np.float32)
                dets = np.column_stack([xyxy, conf, cls])
                result = self.tracker.update(dets, frame)
                if result is not None and len(result) > 0:
                    result = result.astype(np.float32).copy()
                    result[:, [0, 2]] /= w
                    result[:, [1, 3]] /= h
                self.block_tracked[idx] = result
            else:
                self.block_tracked[idx] = self.tracker.update(np.empty((0, 6), dtype=np.float32), frame)
        self.tracker_head = target_idx

    def setup_tracker(self):
        self._advance_tracker_to(len(self.df) - 1)

    def _cgrect_to_norm_xyxy(self, detection) -> list | None:
        """Convert a Vision cgRect to normalized [x1, y1, x2, y2] with top-left origin."""
        rect = detection.get('cgRect')
        if not rect or len(rect) != 2:
            return None
        (x, y), (w, h) = rect
        if w <= 0 or h <= 0:
            return None
        return [x, 1.0 - (y + h), x + w, 1.0 - y]

    def _compute_target_zone(self) -> dict | None:
        """Build the trapezoidal target zone from the first valid box detection."""
        box_detection = None
        for _, row in self.df.iterrows():
            bd = row.get('boxDetection')
            if bd:
                box_detection = bd
                break
        if box_detection is None:
            return None

        keypoints = box_detection.get('keypoints', [])
        if len(keypoints) < 10:
            return None

        pts = [(kp.get('position', [0, 0])[0], self.img_height - kp.get('position', [0, 0])[1])
               for kp in keypoints]

        top_middle, top_left, top_right = pts[6], pts[8], pts[9]
        bottom_left, bottom_right = pts[0], pts[4]

        split_x = top_middle[0]
        if top_right[0] != top_left[0]:
            t = (split_x - top_left[0]) / (top_right[0] - top_left[0])
            split_y = top_left[1] + t * (top_right[1] - top_left[1])
        else:
            split_y = (top_left[1] + top_right[1]) / 2.0
        split_pt = (split_x, split_y)

        if self._target_side(split_x) == 'right':
            top_left, bottom_left = split_pt, pts[2]
        else:
            top_right, bottom_right = split_pt, pts[2]

        return {
            "top_left": top_left,
            "bottom_left": bottom_left,
            "top_right": top_right,
            "bottom_right": bottom_right,
        }

    def _target_side(self, middle_x: float) -> str:
        """Return 'left' or 'right' based on where the first-frame blocks cluster."""
        left_count = right_count = 0
        for _, row in self.df.iterrows():
            block_dets = row.get('blockDetections') or []
            if not isinstance(block_dets, list):
                continue
            for bd in block_dets:
                box = self._cgrect_to_norm_xyxy(bd['boundingBox'])
                if box is None:
                    continue
                center_x = ((box[0] + box[2]) / 2) * self.img_width
                if center_x > middle_x:
                    right_count += 1
                else:
                    left_count += 1
            break
        return 'right' if left_count > right_count else 'left'

    @abstractmethod
    def process_video(self):
        pass
