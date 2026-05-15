from Temporary import Temporary
from VisUtils import draw_landmarks_on_image

import cv2 as cv
import pandas as pd
import numpy as np

from Block import Block
from Counter import Counter
from Config import *


class TemporaryManual(Temporary):

    def __init__(self, args):
        super().__init__(args)

    def process_video(self):
        print("Controls: A/D (±1 frame), W/S (±10 frames), Q (quit)")

        timestamps = self.df['presentationTime'].to_numpy() * 1000.0
       
        while True:
            self.cap.set(cv.CAP_PROP_POS_FRAMES, self.current_frame)
            ret, frame = self.cap.read()

            if not ret:
                break

            time_ms = self.cap.get(cv.CAP_PROP_POS_MSEC)
            print(f"Frame: {self.current_frame}, Time: {time_ms:.6f}ms")

            # Apply detections / update counter
            state_text = ""
            match_idx = np.where(np.abs(timestamps - time_ms) < self.tolerance)[0]
            if len(match_idx) > 0:
                print("Data frame indices: ", match_idx)
                frameResult = self.df.iloc[match_idx[0]]
                self.tracked = self.block_tracked.get(match_idx[0])
                self.counter.update_all(frame, frameResult['state'], tracked=self.tracked)
                frame = self.visualize_frame(frame, frameResult, tracked=self.tracked, counter=self.counter, target_zone=self.target_zone)
                state_text = f"State: {frameResult['state']}"

            # --- All HUD text drawn here, same position as original Time/Frame line ---
            cv.putText(frame, f"Time: {time_ms:.6f}ms | Frame: {self.current_frame} | Counter = {self.counter.counter}",
                    (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if state_text:
                cv.putText(frame, state_text,
                        (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv.imshow("Video Player", frame)

            key = cv.waitKey(0) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('a'):
                self.current_frame = max(0, self.current_frame - 1)
            elif key == ord('d'):
                self.current_frame = min(self.frame_count - 1, self.current_frame + 1)
            elif key == ord('w'):
                self.current_frame = max(0, self.current_frame - 10)
            elif key == ord('s'):
                self.current_frame = min(self.frame_count - 1, self.current_frame + 10)

        self.cap.release()
        cv.destroyAllWindows()

    def draw_keypoints_on_image(self, bgr_image, detection, point_color=(0, 255, 0), box_color=(255, 0, 0), radius=4):
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

    def draw_cgrect_bboxes(self, bgr_image, detection, color=(0, 0, 255), thickness=2):
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

    def draw_target_zone_lines(self, bgr_image, target_zone): 
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

    def visualize_frame(self, frame, frameResult: pd.Series, tracked: 'np.ndarray | None' = None, counter = None, target_zone = None):
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
            annotated = self.draw_keypoints_on_image(annotated, box)
            annotated = self.draw_target_zone_lines(annotated, target_zone)

        # --- Draw hand landmarks ---
        if 'hands' in frameResult and isinstance(frameResult['hands'], list):
            annotated = draw_landmarks_on_image(annotated, frameResult['hands'])

        # --- Draw face bounding boxes ---
        if 'faces' in frameResult and frameResult['faces']:
            for face in frameResult['faces']:
                if 'boundingBox' in face and face['boundingBox']:
                    annotated = self.draw_cgrect_bboxes(annotated, face['boundingBox'])

        # --- Draw block detections ---
        cur_detected_in_target = 0
        if 'blockDetections' in frameResult and frameResult['blockDetections']:
            for blockDetection in frameResult['blockDetections']:
                annotated = self.draw_cgrect_bboxes(annotated, blockDetection['boundingBox'],
                                            color=(255, 0, 255), thickness=2)

        # --- Draw tracker block detections ---
            # columns [x1,y1,x2,y2,track_id,...]
        if self.tracked is not None and len(self.tracked) > 0:
            for row in self.tracked:
                x1, y1, x2, y2, tid = row[0], row[1], row[2], row[3], int(row[4])
                px1, py1 = int(x1 * width), int(y1 * height)
                px2, py2 = int(x2 * width), int(y2 * height)
                cv.rectangle(annotated, (px1, py1), (px2, py2), TRACKER_COLOR, 2)
                cv.putText(annotated, f"T{tid}", (px1, max(py1 - 6, 10)),
                        cv.FONT_HERSHEY_SIMPLEX, 0.6, TRACKER_COLOR, 2, cv.LINE_AA)

        # --- Highlight possible new block added ---
        if self.counter is not None and self.counter.coords_last_block is not None:
            x1, y1, x2, y2 = self.counter.coords_last_block
            cv.rectangle(annotated, (int(x1 * width), int(y1 * height)), 
                (int(x2 * width), int(y2 * height)), COUNTER_COLOR, 2)
            cv.putText(annotated, f"T{tid}", (int(x1 * width), max(int(y1 * height) - 6, 10)),
                cv.FONT_HERSHEY_SIMPLEX, 0.6, COUNTER_COLOR, 2, cv.LINE_AA)

        return annotated

