import cv2 as cv
import pandas as pd
import numpy as np

from Detection import Detection
from VisUtils import draw_landmarks_on_image
from Config import *


class DetectionManual(Detection):

    def process_video(self):
        print("Controls: A/D (±1 frame), W/S (±10 frames), Q (quit)")

        timestamps = self.df['presentationTime'].to_numpy() * 1000.0

        while True:
            self.cap.set(cv.CAP_PROP_POS_FRAMES, self.current_frame)
            ret, self.frame = self.cap.read()
            if not ret:
                break

            time_ms = self.cap.get(cv.CAP_PROP_POS_MSEC)
            print(f"Frame: {self.current_frame}, Time: {time_ms:.6f}ms")

            frame = self.frame.copy()
            state_text = ""
            match_idx = np.where(np.abs(timestamps - time_ms) < self.tolerance)[0]
            if len(match_idx) > 0:
                row_idx = match_idx[0]
                print(f"Data frame index: {row_idx}")
                frame_result = self.df.iloc[row_idx]
                self._advance_tracker_to(row_idx)
                tracked = self.block_tracked.get(row_idx)
                self.counter.update_all(frame_result['state'], tracked=tracked)
                frame = self._visualize_frame(frame, frame_result, tracked=tracked)
                state_text = f"State: {frame_result['state']}"

            cv.putText(frame, f"Time: {time_ms:.6f}ms | Frame: {self.current_frame} | Counter = {self.counter.counter}",
                       (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if state_text:
                cv.putText(frame, state_text, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
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

    def _draw_keypoints(self, image, detection, color=(0, 255, 0), radius=4):
        annotated = image.copy()
        h, w, _ = annotated.shape
        for kp in detection.get('keypoints', []):
            conf = kp.get('confidence', 0)
            x, y = kp.get('position', [0, 0])
            y = h - y
            cv.circle(annotated, (int(x), int(y)), radius, color, thickness=-1)
            cv.putText(annotated, f"{conf:.3f}", (int(x) + 5, int(y) - 5),
                       cv.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv.LINE_AA)
        return annotated

    def _draw_cgrect(self, image, detection, color=(0, 0, 255), thickness=2):
        annotated = image.copy()
        h, w, _ = annotated.shape
        rect = detection.get('cgRect')
        if rect and len(rect) == 2:
            (x_n, y_n), (w_n, h_n) = rect
            x1, x2 = int(x_n * w), int((x_n + w_n) * w)
            y1, y2 = int((1 - y_n - h_n) * h), int((1 - y_n) * h)
            cv.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        return annotated

    def _draw_target_zone(self, image, target_zone):
        annotated = image.copy()
        tl = tuple(map(int, target_zone["top_left"]))
        bl = tuple(map(int, target_zone["bottom_left"]))
        tr = tuple(map(int, target_zone["top_right"]))
        br = tuple(map(int, target_zone["bottom_right"]))
        color = (0, 255, 255)
        cv.line(annotated, tl, tr, color, 2)
        cv.line(annotated, tl, bl, color, 2)
        cv.line(annotated, tr, br, color, 2)
        cv.line(annotated, bl, br, color, 2)
        return annotated

    def _visualize_frame(self, frame, frame_result: pd.Series, tracked: np.ndarray | None = None):
        annotated = frame.copy()
        h, w, _ = annotated.shape

        if frame_result.get('boxDetection'):
            annotated = self._draw_keypoints(annotated, frame_result['boxDetection'])
            annotated = self._draw_target_zone(annotated, self.target_zone)

        if isinstance(frame_result.get('hands'), list):
            annotated = draw_landmarks_on_image(annotated, frame_result['hands'])

        for face in frame_result.get('faces') or []:
            if face.get('boundingBox'):
                annotated = self._draw_cgrect(annotated, face['boundingBox'])

        for block in frame_result.get('blockDetections') or []:
            annotated = self._draw_cgrect(annotated, block['boundingBox'], color=(255, 0, 255))

        if tracked is not None and len(tracked) > 0:
            for row in tracked:
                x1, y1, x2, y2, tid = row[0], row[1], row[2], row[3], int(row[4])
                px1, py1 = int(x1 * w), int(y1 * h)
                px2, py2 = int(x2 * w), int(y2 * h)
                cv.rectangle(annotated, (px1, py1), (px2, py2), TRACKER_COLOR, 2)
                cv.putText(annotated, f"T{tid}", (px1, max(py1 - 6, 10)),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, TRACKER_COLOR, 2, cv.LINE_AA)

        if self.counter.coords_last_block is not None:
            x1, y1, x2, y2 = self.counter.coords_last_block
            cv.rectangle(annotated, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), COUNTER_COLOR, 2)

        return annotated
