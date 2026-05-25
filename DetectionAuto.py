import csv
import cv2 as cv
from pathlib import Path

import progressbar

from Detection import Detection
from DetectionRecord import DetectionRecord


class DetectionAuto(Detection):

    def __init__(self, args):
        super().__init__(args)
        self.setup_tracker()
        self.record = DetectionRecord(self.counter)

    def process_video(self):
        b = progressbar.ProgressBar(maxval=len(self.df)).start()

        for idx, row in self.df.iterrows():
            time_ms = row['presentationTime'] * 1000
            tracked = self.block_tracked.get(idx)
            self.counter.update_all(row['state'], tracked=tracked)
            self.record.update_record(row['state'], time_ms, self.current_frame)
            self.current_frame += 1
            b.update(self.current_frame)

        b.finish()

        timeTag = Path(self.video_path).stem.split('_')[2]
        with open(f'CMORE_Test_Results_{timeTag}_{self.args.mvmt_threshold}.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.record.keys)
            writer.writeheader()
            writer.writerows(self.record.total_record)

        self.cap.release()
