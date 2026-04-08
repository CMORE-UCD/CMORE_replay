import sys
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2 as cv
import pandas as pd
import os
from pathlib import Path
from scipy.optimize import linear_sum_assignment

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green
TRACKER_COLOR = (255, 255, 0)  # cyan for tracker boxes


def cgrect_to_pixel_xywh(detection, w_img, h_img):
    """Convert a Vision cgRect dict to OpenCV (x, y, w, h) in pixel coords."""
    rect = detection.get('cgRect')
    if rect and len(rect) == 2:
        (x_norm, y_norm), (w_norm, h_norm) = rect
        x = int(x_norm * w_img)
        y = int((1 - (y_norm + h_norm)) * h_img)
        w = int(w_norm * w_img)
        h = int(h_norm * h_img)
        if w > 0 and h > 0:
            return (x, y, w, h)
    return None


def compute_iou_matrix(boxes_a, boxes_b):
    """Compute IoU matrix between two lists of [x1, y1, x2, y2] boxes."""
    iou = np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    for i, a in enumerate(boxes_a):
        for j, b in enumerate(boxes_b):
            xi1 = max(a[0], b[0]); yi1 = max(a[1], b[1])
            xi2 = min(a[2], b[2]); yi2 = min(a[3], b[3])
            inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            area_a = (a[2] - a[0]) * (a[3] - a[1])
            area_b = (b[2] - b[0]) * (b[3] - b[1])
            union = area_a + area_b - inter
            iou[i, j] = inter / union if union > 0 else 0
    return iou


class OpenCVMultiTracker:
    """Manages one cv.TrackerCSRT per live track, matched to detections via IoU."""

    def __init__(self, iou_threshold=0.3):
        self.tracks = {}  # track_id -> {'tracker', 'bbox_xywh'}
        self.next_id = 1
        self.iou_threshold = iou_threshold

    def _xywh_to_xyxy(self, xywh):
        x, y, w, h = xywh
        return [x, y, x + w, y + h]

    def update(self, frame, det_xywh_list):
        """
        Args:
            frame: BGR image
            det_xywh_list: list of (x, y, w, h) detections in pixel coords
        Returns:
            (track_ids, tracker_bboxes, coasting) where track_ids and tracker_bboxes
            are aligned to det_xywh_list, and coasting is {track_id: (x,y,w,h)} for
            tracks still alive but unmatched this frame.
        """
        # Step 1: update existing trackers; drop only if CSRT itself fails
        predicted = {}  # track_id -> [x1, y1, x2, y2]
        for tid, track in list(self.tracks.items()):
            success, bbox = track['tracker'].update(frame)
            if success:
                predicted[tid] = self._xywh_to_xyxy(bbox)
            else:
                del self.tracks[tid]

        # Step 2: match predictions to current detections via IoU
        track_ids_ordered = list(predicted.keys())
        det_xyxy_list = [self._xywh_to_xyxy(d) for d in det_xywh_list]

        matched_det_to_track = {}  # det_idx -> track_id
        if track_ids_ordered and det_xywh_list:
            pred_boxes = [predicted[tid] for tid in track_ids_ordered]
            iou_mat = compute_iou_matrix(pred_boxes, det_xyxy_list)
            row_ind, col_ind = linear_sum_assignment(-iou_mat)
            for r, c in zip(row_ind, col_ind):
                if iou_mat[r, c] >= self.iou_threshold:
                    matched_det_to_track[c] = track_ids_ordered[r]

        # Step 3: build output; re-init matched trackers on detection bbox
        result_ids = []
        result_bboxes = []
        matched_track_ids = set()
        for det_idx, det_xywh in enumerate(det_xywh_list):
            if det_idx in matched_det_to_track:
                tid = matched_det_to_track[det_idx]
                matched_track_ids.add(tid)
                self.tracks[tid]['tracker'].init(frame, det_xywh)
                self.tracks[tid]['bbox_xywh'] = det_xywh
                result_ids.append(tid)
                result_bboxes.append(det_xywh)
            else:
                # New track
                tracker = cv.TrackerCSRT_create()
                tracker.init(frame, det_xywh)
                self.tracks[self.next_id] = {'tracker': tracker, 'bbox_xywh': det_xywh}
                result_ids.append(self.next_id)
                result_bboxes.append(det_xywh)
                self.next_id += 1

        # Step 4: collect coasting tracks (unmatched but still tracked by CSRT)
        coasting = {}  # track_id -> (x, y, w, h) CSRT-predicted bbox
        for tid in self.tracks.keys():
            if tid not in matched_track_ids:
                if tid in predicted:
                    x1, y1, x2, y2 = predicted[tid]
                    coasting[tid] = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))

        return result_ids, result_bboxes, coasting

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

def visualize_frame(frame, frameResult: pd.Series, track_ids=None, tracker_bboxes=None, coasting=None):
    """Visualize all detections from a frame result on the input frame.

    Args:
        frame: BGR image from OpenCV
        frameResult: Detection result dictionary containing hands, faces, boxDetection, blockDetections
        track_ids: list of int track IDs aligned to blockDetections
        tracker_bboxes: list of (x, y, w, h) tracker bboxes in pixel coords, aligned to blockDetections
        coasting: dict of {track_id: (x, y, w, h)} for tracks still alive but unmatched this frame

    Returns:
        Annotated BGR image with all detections drawn
    """
    annotated = frame.copy()

    # Draw box detection keypoints if present
    if 'boxDetection' in frameResult and frameResult['boxDetection']:
        annotated = draw_keypoints_on_image(annotated, frameResult['boxDetection'])

    # Draw hand landmarks if present
    if 'hands' in frameResult and isinstance(frameResult['hands'], list):
        annotated = draw_landmarks_on_image(annotated, frameResult['hands'])

    # Draw face bounding boxes if present
    if 'faces' in frameResult and frameResult['faces']:
        for face in frameResult['faces']:
            if 'boundingBox' in face and face['boundingBox']:
                annotated = draw_cgrect_bboxes(annotated, face['boundingBox'])

    # Draw block detections if present
    if 'blockDetections' in frameResult and frameResult['blockDetections']:
        for i, blockDetection in enumerate(frameResult['blockDetections']):
            # Draw the detection bounding box in magenta
            annotated = draw_cgrect_bboxes(annotated, blockDetection['boundingBox'], color=(255, 0, 255), thickness=2)
            # Draw tracker bbox and ID in cyan
            if track_ids and tracker_bboxes and i < len(track_ids):
                tid = track_ids[i]
                x, y, w, h = tracker_bboxes[i]
                cv.rectangle(annotated, (x, y), (x + w, y + h), TRACKER_COLOR, 2)
                cv.putText(annotated, f"T{tid}", (x, max(y - 6, 10)),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, TRACKER_COLOR, 2, cv.LINE_AA)

    # Draw coasting tracks (alive in tracker but no detection match this frame) in orange
    if coasting:
        for tid, (x, y, w, h) in coasting.items():
            cv.rectangle(annotated, (x, y), (x + w, y + h), (0, 165, 255), 2)
            cv.putText(annotated, f"T{tid}", (x, max(y - 6, 10)),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv.LINE_AA)

    # show the state
    cv.putText(annotated, f"State: {frameResult['state']}",
                    (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return annotated

def main():
    if len(sys.argv) < 2:
        print("Error: Please provide video file path as command line argument")
        print("Usage: python main.py <video_path>")
        sys.exit(1)

    # Open video file
    video_path = sys.argv[1] 
    
    timeTag = Path(video_path).stem.split('_')[2]

    # Open up the results
    df = pd.read_json(f"CMORE_Results_{timeTag}.json")
    timestamps = df['presentationTime'].to_numpy() * 1000.0
    
    if not video_path:
        print("Error: Please provide video file path as command line argument")
        print("Usage: python main.py <video_path>")
        sys.exit(1)
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
    print("Controls: A/D (±1 frame), Q (quit)")

    tracker = OpenCVMultiTracker()
    block_track_ids = {}      # df row index -> list[int]
    block_tracker_bboxes = {} # df row index -> list[(x,y,w,h)]
    block_coasting = {}       # df row index -> {track_id: (x,y,w,h)}
    frame_coasting = {}       # video frame index -> {track_id: (x,y,w,h)} for frames with no detection
    tracker_last_frame = -1   # tracker has been updated up to this video frame index

    while True:
        # Advance tracker forward to current_frame if not yet computed (cache miss going forward)
        if current_frame > tracker_last_frame:
            for fi in range(tracker_last_frame + 1, current_frame + 1):
                cap.set(cv.CAP_PROP_POS_FRAMES, fi)
                ret_t, frame_t = cap.read()
                if not ret_t:
                    break
                h_t, w_t = frame_t.shape[:2]
                t_ms = cap.get(cv.CAP_PROP_POS_MSEC)
                mi = np.where(np.abs(timestamps - t_ms) < tolerance)[0]
                if len(mi) > 0:
                    row = df.iloc[mi[0]]
                    block_dets = row.get('blockDetections') or []
                    if not isinstance(block_dets, list):
                        block_dets = []
                    xywh_list = [cgrect_to_pixel_xywh(bd['boundingBox'], w_t, h_t) for bd in block_dets]
                    xywh_list = [b for b in xywh_list if b is not None]
                    ids, tr_bboxes, coasting = tracker.update(frame_t, xywh_list)
                    block_track_ids[mi[0]] = ids
                    block_tracker_bboxes[mi[0]] = tr_bboxes
                    block_coasting[mi[0]] = coasting
                else:
                    _, _, coasting = tracker.update(frame_t, [])
                    frame_coasting[fi] = coasting
                tracker_last_frame = fi

        cap.set(cv.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()

        if not ret:
            break

        time_ms = cap.get(cv.CAP_PROP_POS_MSEC)
        cv.putText(frame, f"Time: {time_ms:.6f}ms | Frame: {current_frame}",
                    (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        print(f"Frame: {current_frame}, Time: {time_ms:.6f}ms")

        # find the index of the match in timestamps
        match_idx = np.where(np.abs(timestamps - time_ms) < tolerance)[0]
        if len(match_idx) > 0:
            print("Data frame indices: ", match_idx)
            frameResult = df.iloc[match_idx[0]]
            ids = block_track_ids.get(match_idx[0], [])
            tr_bboxes = block_tracker_bboxes.get(match_idx[0], [])
            coasting = block_coasting.get(match_idx[0], {})
            frame = visualize_frame(frame, frameResult, track_ids=ids, tracker_bboxes=tr_bboxes, coasting=coasting)
        else:
            # No detection data for this frame — draw any coasting tracks directly
            coasting = frame_coasting.get(current_frame, {})
            for tid, (x, y, w, h) in coasting.items():
                cv.rectangle(frame, (x, y), (x + w, y + h), TRACKER_COLOR, 2)
                cv.putText(frame, f"T{tid}", (x, max(y - 6, 10)),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, TRACKER_COLOR, 2, cv.LINE_AA)
        
        cv.imshow("Video Player", frame)
        
        key = cv.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a'): 
            current_frame = max(0, current_frame - 1)
        elif key == ord('d'):  
            current_frame = min(frame_count - 1, current_frame + 1)
        elif key == ord('w'):  # W key
            current_frame = max(0, current_frame - 10)
        elif key == ord('s'):  # S key
            current_frame = min(frame_count - 1, current_frame + 10)
    
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
