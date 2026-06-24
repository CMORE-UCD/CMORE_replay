from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2 as cv
from Config import *

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