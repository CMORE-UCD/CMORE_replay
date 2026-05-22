from abc import ABC, abstractmethod
import sys
import cv2 as cv
import pandas as pd
import numpy as np
from pathlib import Path
import boxmot

from Block import Block
from Config import *
from Counter import Counter

class Detection:
    args = None
    video_path = None
    df = None
    counter = None
    cap = None
    target_zone = None
    tracker = None
    block_tracked = {}  # df row index -> np.ndarray (n, 8) or None
    current_frame = 0
    tolerance = 0
    frame_count = 0

    def __init__(self, args):
        self.args = args
        self.video_path = args.video_path
        self.setup_cap()

        # Open up the results
        timeTag = Path(self.video_path).stem.split('_')[2]
        self.df = pd.read_json(f"CMORE_Results_{timeTag}.json")
        
        ret, frame = self.cap.read()

        self.setup_target_zone()
        self.counter = Counter(self.target_zone, frame, args.mvmt_threshold)
        self.setup_tracker()
    
    def setup_cap(self):
        self.cap = cv.VideoCapture(self.video_path)

        if not self.cap.isOpened():
            print("Error: Could not open video file")
            sys.exit(1)

        fps = self.cap.get(cv.CAP_PROP_FPS)
        self.tolerance = 10 # 1000 / fps
        print(f"Using time tolerance of {self.tolerance:.3f}ms")
        self.frame_count = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))
        print(f"Video loaded. FPS: {fps}, Total frames: {self.frame_count}")

    def setup_target_zone(self):
        img_height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        img_width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.target_zone = self.compute_target_zone(self.df, img_height, img_width)

    def setup_tracker(self):
        self.tracker = self.make_tracker(self.args.tracker, self.args.track_buffer)
        dummy_frame = np.zeros((1, 1, 3), dtype=np.uint8)

        for idx, row in self.df.iterrows():
            block_dets = row.get('blockDetections') or []
            if not isinstance(block_dets, list):
                block_dets = []
            boxes = [self.cgrect_to_norm_xyxy(bd['boundingBox']) for bd in block_dets]
            valid = [(b, bd) for b, bd in zip(boxes, block_dets) if b is not None]
            if valid:
                xyxy = np.array([b for b, _ in valid], dtype=np.float32)
                conf = np.array([float(bd.get('confidence', 1.0)) for _, bd in valid], dtype=np.float32)
                cls = np.zeros(len(valid), dtype=np.float32)
                dets = np.column_stack([xyxy, conf, cls])
                self.block_tracked[idx] = self.tracker.update(dets, dummy_frame)
            else:
                self.block_tracked[idx] = self.tracker.update(np.empty((0, 6), dtype=np.float32), dummy_frame)

    def make_tracker(self, name: str, track_buffer: int = 30):
        """Instantiate a boxmot motion-only tracker by name."""
        if name == 'bytetrack':
            return boxmot.ByteTrack(track_buffer=track_buffer)
        elif name == 'ocsort':
            return boxmot.OcSort()
        elif name == 'sfsort':
            return boxmot.SFSORT()
        elif name == 'boosttrack':
            return boxmot.BoostTrack()
        else:
            raise ValueError(f"Unknown tracker '{name}'. Choose from: {MOTION_TRACKERS}")

    def cgrect_to_norm_xyxy(self, detection):
        """Convert a Vision cgRect dict to normalized [x1, y1, x2, y2] (top-left origin)."""
        rect = detection.get('cgRect')
        if rect and len(rect) == 2:
            (x_norm, y_norm), (w_norm, h_norm) = rect
            x1 = x_norm
            x2 = x_norm + w_norm
            y1 = 1.0 - (y_norm + h_norm)  # flip y: Vision bottom-left → top-left
            y2 = 1.0 - y_norm
            if w_norm > 0 and h_norm > 0:
                return [x1, y1, x2, y2]
        return None

    def compute_target_zone(self, df, img_height, img_width):
        """Computes the target zone, a trapezoidal shape, from keypoints. 

        The returned lines are always stored globally; call this once when the box is
        first detected.

        Args:
            box_detection: dict with 'keypoints' list; each keypoint has
                        'position' [x, y] in pixels (Vision y-axis, not flipped yet)
                        and optional 'confidence'.
            img_height: frame dimensions in pixels.

        Returns:
            (top_left, top_right): two (x, y) pixel tuples representing the delimiter
                                segment, or None if fewer than 3 keypoints exist.
        """
        box_detection = None

        # find first valid box detection
        for idx in range(0, len(df)):
            row = df.iloc[idx]
            box_detection = row.get('boxDetection')  # or row['boxDetection'] if you're sure it exists

            # Adjust validity check to match your data type:
            if box_detection is None or (hasattr(box_detection, '__len__') and len(box_detection) == 0):
                continue  # skip empty/null detections

            break

        keypoints = box_detection.get('keypoints', [])
        if len(keypoints) < 3:
            return None

        # Flip y to screen coordinates (same transform used during drawing)
        pts = []
        for kp in keypoints:
            x, y = kp.get('position', [0, 0])
            y_screen = img_height - y
            pts.append((x, y_screen))

        
        top_middle, top_left, top_right = pts[6], pts[8], pts[9]
        bottom_left, bottom_right = pts[0], pts[4]

        # Split the bottom segment at the x-coordinate of the top point
        split_x = top_middle[0]
        # Linearly interpolate y on the segment top_left→top_right at x=split_x
        if top_right[0] != top_left[0]:
            t = (split_x - top_left[0]) / (top_right[0] - top_left[0])
            split_y = top_left[1] + t * (top_right[1] - top_left[1])
        else:
            split_y = (top_left[1] + top_right[1]) / 2.0
        split_pt = (split_x, split_y)

        # Choose the half according to target_side
        # choose side
        if self.target_side(split_x, df, img_width) == 'right':
            top_left = split_pt
            bottom_left = pts[2]
        else:
            top_right = split_pt
            bottom_right = pts[2]

        return {
            "top_left" : top_left,
            "bottom_left" : bottom_left,
            "top_right" : top_right,
            "bottom_right" : bottom_right
        }

    def target_side(self, middle_x, df, img_width):
        left_count = 0
        right_count = 0

        # up till first valid block detections

        for idx, row in self.df.iterrows():
            block_dets = row.get('blockDetections') or []
            if not isinstance(block_dets, list):
                continue
            boxes = [self.cgrect_to_norm_xyxy(bd['boundingBox']) for bd in block_dets]

            for x1, y1, x2, y2 in boxes:
                box_center_x = ((x1 + x2) / 2) * img_width
                if box_center_x > middle_x:
                    right_count += 1
                else:
                    left_count += 1

            break
        
        if left_count > right_count:
            return 'right'
        else:
            return 'left'


    @abstractmethod
    def process_video(self):
        pass