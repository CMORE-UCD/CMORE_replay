import sys
import argparse
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2 as cv
import pandas as pd
import os
from pathlib import Path
import boxmot
from dataclasses import dataclass, field

# --- Block tracking struct ---
@dataclass
class Block:
    id: int
    last5box: list = field(default_factory=list)  # up to 5 [x1,y1,x2,y2] normalized bboxes

    def update(self, bbox_norm):
        """Append a new normalized [x1,y1,x2,y2] bbox, keeping only the last 5."""
        self.last5box.append(bbox_norm)
        if len(self.last5box) > 5:
            self.last5box.pop(0)

MOTION_TRACKERS = ['bytetrack', 'ocsort', 'sfsort', 'boosttrack']
MODES = ['manual', 'test']

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

    def __init__(self, target_zone): 
        self.target_zone = target_zone

    def has_movement(self, block: Block, threshold: float = 0.25) -> bool:
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
        return 1 >= (displacement / ref) >= threshold
    
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
    
    def update_dimensions(self, frame):
        annotated = frame.copy()
        self.height, self.width, _ = annotated.shape

    def reset_states(self, frame_state_result):
        if self.active_counting_state and self.crossed_back and frame_state_result == 'crossed':
            self.active_counting_state = False

        if frame_state_result == 'crossedBack':
            self.crossed_back = True

    def update_all(self, frame, frame_result, tracked: 'np.ndarray | None' = None):
        self.update_dimensions(frame)
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

def make_tracker(name: str, track_buffer: int = 30):
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

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green
TRACKER_COLOR = (255, 255, 0)  # cyan for tracker boxes
COUNTER_COLOR = (138,43,226) # violet for counting boxes

def cgrect_to_norm_xyxy(detection):
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

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

# Mapping from Vision framework joint names to MediaPipe landmark indices
VISION_TO_MEDIAPIPE = {
    'wrist': 0,
    'thumbCMC': 1, 'thumbMCP': 2, 'thumbIP': 3, 'thumbTip': 4,
    'indexMCP': 5, 'indexPIP': 6, 'indexDIP': 7, 'indexTip': 8,
    'middleMCP': 9, 'middlePIP': 10, 'middleDIP': 11, 'middleTip': 12,
    'ringMCP': 13, 'ringPIP': 14, 'ringDIP': 15, 'ringTip': 16,
    'littleMCP': 17, 'littlePIP': 18, 'littleDIP': 19, 'littleTip': 20,
    # Vision uses thumbMP instead of thumbMCP
    'thumbMP': 2
}

def compute_target_zone(df, img_height, target_side):
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
    if target_side == 'right':
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

def draw_landmarks_on_image(rgb_image, detection_result):
  """
  Draw hand landmarks on image using Vision framework detection results.
  
  Args:
    rgb_image: Input RGB image
    detection_result: List of hand detections from Vision framework
  """
  
  annotated_image = np.copy(rgb_image)
  height, width, _ = annotated_image.shape
  
  # Loop through detected hands
  for hand_detection in detection_result:
    # Parse allJoints list (alternates between name strings and data dicts)
    joints = {}
    all_joints = hand_detection.get('allJoints', [])
    
    for item in all_joints:
      if isinstance(item, dict) and 'jointName' in item:
        joint_name = item['jointName']
        location = item.get('location', {}).get('cgPoint', [0, 0])
        confidence = item.get('confidence', 0)
        joints[joint_name] = {'location': location, 'confidence': confidence}
    
    # Create MediaPipe-style landmarks
    landmarks = [None] * 21  # MediaPipe has 21 hand landmarks
    for joint_name, data in joints.items():
      if joint_name in VISION_TO_MEDIAPIPE:
        idx = VISION_TO_MEDIAPIPE[joint_name]
        x, y = data['location']
        landmarks[idx] = landmark_pb2.NormalizedLandmark(x=x, y=1-y, z=0)
    
    # Fill any missing landmarks with (0,0,0)
    for i in range(21):
      if landmarks[i] is None:
        landmarks[i] = landmark_pb2.NormalizedLandmark(x=0, y=0, z=0)
    
    # Draw the hand landmarks
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend(landmarks)
    
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())
    
    # Get handedness
    chirality = hand_detection.get('chirality', {})
    handedness_text = 'Left' if 'left' in chirality else 'Right' if 'right' in chirality else 'Unknown'
    
    # Get bounding box for text position
    valid_joints = [j for j in joints.values() if j['location'][0] > 0]
    if valid_joints:
      x_coordinates = [j['location'][0] for j in valid_joints]
      y_coordinates = [1 - j['location'][1] for j in valid_joints]
      text_x = int(min(x_coordinates) * width)
      text_y = int(min(y_coordinates) * height) - MARGIN
      
      # Draw handedness label
      cv.putText(annotated_image, handedness_text,
                  (text_x, text_y), cv.FONT_HERSHEY_DUPLEX,
                  FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv.LINE_AA)
  
  return annotated_image

def draw_keypoints_on_image(bgr_image, detection, point_color=(0, 255, 0), box_color=(255, 0, 0), radius=4):
    """Draw keypoints and bounding box for a Vision detection.

    Args:
        bgr_image: OpenCV BGR image.
        detection: Dict with 'keypoints' (list of dicts with 'position' [x, y] in pixels
                   and optional 'confidence'), plus centerX/centerY/width/height/objectConf.
        point_color: BGR tuple for keypoint circles.
        box_color: BGR tuple for bounding box rectangle.
        radius: Circle radius in pixels.

    Returns:
        Annotated copy of the image.
    """
    annotated = bgr_image.copy()
    height, width, _ = annotated.shape

    keypoints = detection.get('keypoints', [])
    for kp in keypoints:
        conf = kp.get('confidence', 0)
        x, y = kp.get('position', [0, 0])
        y = height - y  # Flip y-coordinate
        cv.circle(annotated, (int(x), int(y)), radius, point_color, thickness=-1)
        cv.putText(annotated, f"{conf:.3f}", (int(x) + 5, int(y) - 5),
                   cv.FONT_HERSHEY_SIMPLEX, 0.4, point_color, 1, cv.LINE_AA)

    return annotated

def draw_cgrect_bboxes(bgr_image, detection, color=(0, 0, 255), thickness=2):
    """Draw bounding boxes described by Vision-style cgRect on a BGR image.

    Args:
        bgr_image: OpenCV BGR image.
        detection: Dict containing 'cgRect': [[x, y], [w, h]] with normalized values (0-1).
                   Vision uses origin at bottom-left, so we flip y accordingly.
        color: BGR color for rectangle.
        thickness: Line thickness for rectangle.

    Returns:
        Annotated copy of the image.
    """
    annotated = bgr_image.copy()
    h_img, w_img, _ = annotated.shape

    rect = detection.get('cgRect')
    if rect and len(rect) == 2:
        (x_norm, y_norm), (w_norm, h_norm) = rect
        # Convert normalized rect (origin bottom-left) to pixel box (origin top-left)
        x1 = int(x_norm * w_img)
        x2 = int((x_norm + w_norm) * w_img)
        y_top_norm = y_norm + h_norm
        y1 = int((1 - y_top_norm) * h_img)
        y2 = int((1 - y_norm) * h_img)
        cv.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

    return annotated

def draw_target_zone_lines(bgr_image, target_zone): 
    """Overlay the computed delimiter line segment on the frame."""
    annotated = bgr_image.copy()
    
    top_left = (int(target_zone["top_left"][0]), int(target_zone["top_left"][1]))
    bottom_left = (int(target_zone["bottom_left"][0]), int(target_zone["bottom_left"][1]))
    top_right = (int(target_zone["top_right"][0]), int(target_zone["top_right"][1]))
    bottom_right = (int(target_zone["bottom_right"][0]), int(target_zone["bottom_right"][1]))

    cv.line(annotated, top_left, top_right, (0, 255, 255), 2)  # yellow line
    cv.line(annotated, top_left, bottom_left, (0, 255, 255), 2)  # yellow line
    cv.line(annotated, top_right, bottom_right, (0, 255, 255), 2)  # yellow line
    cv.line(annotated, bottom_left, bottom_right, (0, 255, 255), 2)  # yellow line
    return annotated

def visualize_frame(frame, frameResult: pd.Series, tracked: 'np.ndarray | None' = None, counter = None, target_zone = None):
    """Visualize all detections from a frame result on the input frame.
    
    Also updates the crossing counter based on blockDetections vs the delimiter line.
    
    Args:
        frame: BGR image from OpenCV
        frameResult: Detection result dictionary containing hands, faces, boxDetection, blockDetections
        
    Returns:
        Annotated BGR image with all detections drawn
    """

    annotated = frame.copy()
    height, width, _ = annotated.shape

    # --- Build / refresh the delimiter line from boxDetection keypoints ---
    if 'boxDetection' in frameResult and frameResult['boxDetection']:
        box = frameResult['boxDetection']
        annotated = draw_keypoints_on_image(annotated, box)
        annotated = draw_target_zone_lines(annotated, target_zone)

    # --- Draw hand landmarks ---
    if 'hands' in frameResult and isinstance(frameResult['hands'], list):
        annotated = draw_landmarks_on_image(annotated, frameResult['hands'])

    # --- Draw face bounding boxes ---
    if 'faces' in frameResult and frameResult['faces']:
        for face in frameResult['faces']:
            if 'boundingBox' in face and face['boundingBox']:
                annotated = draw_cgrect_bboxes(annotated, face['boundingBox'])

    # --- Draw block detections ---
    cur_detected_in_target = 0
    if 'blockDetections' in frameResult and frameResult['blockDetections']:
        for blockDetection in frameResult['blockDetections']:
            annotated = draw_cgrect_bboxes(annotated, blockDetection['boundingBox'],
                                           color=(255, 0, 255), thickness=2)

    # --- Draw tracker block detections ---
        # columns [x1,y1,x2,y2,track_id,...]
    if tracked is not None and len(tracked) > 0:
        for row in tracked:
            x1, y1, x2, y2, tid = row[0], row[1], row[2], row[3], int(row[4])
            px1, py1 = int(x1 * width), int(y1 * height)
            px2, py2 = int(x2 * width), int(y2 * height)
            cv.rectangle(annotated, (px1, py1), (px2, py2), TRACKER_COLOR, 2)
            cv.putText(annotated, f"T{tid}", (px1, max(py1 - 6, 10)),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, TRACKER_COLOR, 2, cv.LINE_AA)

    # --- Highlight possible new block added ---
    if counter is not None and counter.coords_last_block is not None:
        x1, y1, x2, y2 = counter.coords_last_block
        cv.rectangle(annotated, (int(x1 * width), int(y1 * height)), 
            (int(x2 * width), int(y2 * height)), COUNTER_COLOR, 2)
        cv.putText(annotated, f"T{tid}", (int(x1 * width), max(int(y1 * height) - 6, 10)),
            cv.FONT_HERSHEY_SIMPLEX, 0.6, COUNTER_COLOR, 2, cv.LINE_AA)

    return annotated

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', 
                        help='Path to the video file')
    parser.add_argument('--tracker', choices=MOTION_TRACKERS, default='bytetrack',
                        help='Tracker to use (default: bytetrack)')
    parser.add_argument('--track_buffer', type=int, default=30,
                        help='Frames to keep a lost track alive (bytetrack only, default: 30)')
    # parser.add_argument('--mode', choices=MODES, default='manual', 
    #                     help='Mode to enter (default: manual)')
    
    # add new CLI arg 
        # changing threshold for movement
        # automating process, writing out to data file

    args = parser.parse_args()

    video_path = args.video_path

    timeTag = Path(video_path).stem.split('_')[2]

    # Open up the results
    df = pd.read_json(f"CMORE_Results_{timeTag}.json")
    timestamps = df['presentationTime'].to_numpy() * 1000.0

    cap = cv.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video file")
        sys.exit(1)

    fps = cap.get(cv.CAP_PROP_FPS)
    tolerance = 10 # 1000 / fps
    print(f"Using time tolerance of {tolerance:.3f}ms")
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    print(f"Video loaded. FPS: {fps}, Total frames: {frame_count}")

    target_side = 'right'  # default: hand moving left→right; TO DO: is there data in the df that tells us side?
    img_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    target_zone = compute_target_zone(df, img_height, target_side)

    print(f"Pre-computing tracks with {args.tracker}...")

    tracker = make_tracker(args.tracker, args.track_buffer)
    block_tracked = {}  # df row index -> np.ndarray (n, 8) or None
    dummy_frame = np.zeros((1, 1, 3), dtype=np.uint8)
    counter = Counter(target_zone)

    for idx, row in df.iterrows():
        block_dets = row.get('blockDetections') or []
        if not isinstance(block_dets, list):
            block_dets = []
        boxes = [cgrect_to_norm_xyxy(bd['boundingBox']) for bd in block_dets]
        valid = [(b, bd) for b, bd in zip(boxes, block_dets) if b is not None]
        if valid:
            xyxy = np.array([b for b, _ in valid], dtype=np.float32)
            conf = np.array([float(bd.get('confidence', 1.0)) for _, bd in valid], dtype=np.float32)
            cls = np.zeros(len(valid), dtype=np.float32)
            dets = np.column_stack([xyxy, conf, cls])
            block_tracked[idx] = tracker.update(dets, dummy_frame)
        else:
            block_tracked[idx] = tracker.update(np.empty((0, 6), dtype=np.float32), dummy_frame)


    # mode = args.mode
    print("Controls: A/D (±1 frame), W/S (±10 frames), Q (quit)")

    while True:
        cap.set(cv.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()

        if not ret:
            break

        time_ms = cap.get(cv.CAP_PROP_POS_MSEC)
        print(f"Frame: {current_frame}, Time: {time_ms:.6f}ms")

        # Apply detections / update counter
        state_text = ""
        match_idx = np.where(np.abs(timestamps - time_ms) < tolerance)[0]
        if len(match_idx) > 0:
            print("Data frame indices: ", match_idx)
            frameResult = df.iloc[match_idx[0]]
            tracked = block_tracked.get(match_idx[0])
            counter.update_all(frame, frameResult['state'], tracked=tracked)
            frame = visualize_frame(frame, frameResult, tracked=tracked, counter=counter, target_zone=target_zone)
            state_text = f"State: {frameResult['state']}"

        # --- All HUD text drawn here, same position as original Time/Frame line ---
        cv.putText(frame, f"Time: {time_ms:.6f}ms | Frame: {current_frame} | Counter = {counter.counter}",
                   (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if state_text:
            cv.putText(frame, state_text,
                       (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv.imshow("Video Player", frame)

        key = cv.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a'):
            current_frame = max(0, current_frame - 1)
        elif key == ord('d'):
            current_frame = min(frame_count - 1, current_frame + 1)
        elif key == ord('w'):
            current_frame = max(0, current_frame - 10)
        elif key == ord('s'):
            current_frame = min(frame_count - 1, current_frame + 10)

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()