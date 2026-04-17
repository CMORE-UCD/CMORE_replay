import sys
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2 as cv
import pandas as pd
import os
from pathlib import Path

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

# --- Counter state ---
counter = 0
hand_over_line = False

# Delimiter line state: two (pixel) endpoints, and which x-side triggers a crossing.
# delimiter_line = (pt_left, pt_right) where each pt is (x, y) in pixel coords.
# delimiter_side = 'right' | 'left'  — the side a block must cross INTO to count.
delimiter_line = None
delimiter_side = None

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

def compute_delimiter_from_box(box_detection, img_width, img_height):
    """Compute the delimiter line from the boxDetection keypoints.

    Takes the three highest keypoints (lowest pixel y-value = highest on screen
    after the Vision y-flip), computes the segment between the bottom two of that
    trio, then splits it at the x-coordinate of the topmost point.  Returns
    (pt_left, pt_right) in pixel coords — the half-segment on the side selected
    by `delimiter_side`, i.e. the two endpoints of the delimiter.

    The returned line is always stored globally; call this once when the box is
    first detected.

    Args:
        box_detection: dict with 'keypoints' list; each keypoint has
                       'position' [x, y] in pixels (Vision y-axis, not flipped yet)
                       and optional 'confidence'.
        img_width, img_height: frame dimensions in pixels.

    Returns:
        (pt_left, pt_right): two (x, y) pixel tuples representing the delimiter
                             segment, or None if fewer than 3 keypoints exist.
    """
    keypoints = box_detection.get('keypoints', [])
    if len(keypoints) < 3:
        return None

    # Flip y to screen coordinates (same transform used during drawing)
    pts = []
    for kp in keypoints:
        x, y = kp.get('position', [0, 0])
        y_screen = img_height - y
        pts.append((x, y_screen))

    
    top1, pt_left, pt_right = pts[6], pts[8], pts[9]

    # Split the bottom segment at the x-coordinate of the top point
    split_x = top1[0]
    # Linearly interpolate y on the segment pt_left→pt_right at x=split_x
    if pt_right[0] != pt_left[0]:
        t = (split_x - pt_left[0]) / (pt_right[0] - pt_left[0])
        split_y = pt_left[1] + t * (pt_right[1] - pt_left[1])
    else:
        split_y = (pt_left[1] + pt_right[1]) / 2.0
    split_pt = (split_x, split_y)

    # Choose the half according to delimiter_side
    global delimiter_side
    if delimiter_side == 'right':
        # Right half: split_pt → pt_right
        return (split_pt, pt_right)
    else:
        # Left half (default): pt_left → split_pt
        return (pt_left, split_pt)


def rect_crosses_delimiter(cgRect, img_width, img_height):
    """Return True if a cgRect bounding box overlaps the x-range of the delimiter line.

    We only check the horizontal (x-axis) extent of the block against the
    x-span of the delimiter segment, consistent with the spec.

    Args:
        cgRect: [[x_norm, y_norm], [w_norm, h_norm]] — normalized Vision coords.
        img_width, img_height: frame dimensions.

    Returns:
        bool
    """
    global delimiter_line
    if delimiter_line is None:
        return False

    (x_norm, y_norm), (w_norm, h_norm) = cgRect
    block_x1 = x_norm * img_width
    block_x2 = (x_norm + w_norm) * img_width
    block_y_top_norm = y_norm + h_norm
    block_y1 = int((1 - block_y_top_norm) * img_height)
    block_y2 = int((1 - y_norm) * img_height)

    seg_x1 = min(delimiter_line[0][0], delimiter_line[1][0])
    seg_x2 = max(delimiter_line[0][0], delimiter_line[1][0])
    seg_y = min(delimiter_line[0][1], delimiter_line[1][1])

    # Overlap when intervals intersect
    return block_x1 <= seg_x2 and block_x2 >= seg_x1 and block_y1 <= seg_y <= block_y2

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


def draw_delimiter_line(bgr_image):
    """Overlay the computed delimiter line segment on the frame."""
    global delimiter_line
    if delimiter_line is None:
        return bgr_image
    annotated = bgr_image.copy()
    pt1 = (int(delimiter_line[0][0]), int(delimiter_line[0][1]))
    pt2 = (int(delimiter_line[1][0]), int(delimiter_line[1][1]))
    cv.line(annotated, pt1, pt2, (0, 255, 255), 2)  # yellow line
    return annotated


def visualize_frame(frame, frameResult: pd.Series):
    """Visualize all detections from a frame result on the input frame.
    
    Also updates the crossing counter based on blockDetections vs the delimiter line.
    
    Args:
        frame: BGR image from OpenCV
        frameResult: Detection result dictionary containing hands, faces, boxDetection, blockDetections
        
    Returns:
        Annotated BGR image with all detections drawn
    """
    global counter, hand_over_line, delimiter_line, delimiter_side

    annotated = frame.copy()
    height, width, _ = annotated.shape

    # --- Build / refresh the delimiter line from boxDetection keypoints ---
    if 'boxDetection' in frameResult and frameResult['boxDetection']:
        box = frameResult['boxDetection']
        annotated = draw_keypoints_on_image(annotated, box)
        if delimiter_line is None:
            # Compute for the first time
            delimiter_line = compute_delimiter_from_box(box, width, height)

    # --- Draw hand landmarks ---
    if 'hands' in frameResult and isinstance(frameResult['hands'], list):
        annotated = draw_landmarks_on_image(annotated, frameResult['hands'])

    # --- Draw face bounding boxes ---
    if 'faces' in frameResult and frameResult['faces']:
        for face in frameResult['faces']:
            if 'boundingBox' in face and face['boundingBox']:
                annotated = draw_cgrect_bboxes(annotated, face['boundingBox'])

    # --- Draw block detections and update counter ---
    block_crosses = False
    if 'blockDetections' in frameResult and frameResult['blockDetections']:
        for blockDetection in frameResult['blockDetections']:
            annotated = draw_cgrect_bboxes(annotated, blockDetection['boundingBox'],
                                           color=(255, 0, 255), thickness=2)
            rect = blockDetection['boundingBox'].get('cgRect')
            if rect and rect_crosses_delimiter(rect, width, height):
                block_crosses = True

    # Counter logic:
    # Increment when a block first crosses the delimiter (hand_over_line was False).
    # Reset hand_over_line when no hand landmark x is inside the delimiter x-range.
    if block_crosses and not hand_over_line:
        counter += 1
        hand_over_line = True
    elif hand_over_line and not frameResult['state'] == 'crossed':
        hand_over_line = False

    # --- Draw delimiter line ---
    annotated = draw_delimiter_line(annotated)

    return annotated

def main():
    if len(sys.argv) < 2:
        print("Error: Please provide video file path as command line argument")
        print("Usage: uv run ./main.py <video_path>")
        sys.exit(1)

    global delimiter_side
    delimiter_side = 'right'  # default: hand moving left→right

    # Open video file
    video_path = sys.argv[1]

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
            frame = visualize_frame(frame, frameResult)
            state_text = f"State: {frameResult['state']}"

        # --- All HUD text drawn here, same position as original Time/Frame line ---
        cv.putText(frame, f"Time: {time_ms:.6f}ms | Frame: {current_frame} | Counter = {counter}",
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