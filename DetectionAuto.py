from Detection import Detection
import cv2 as cv
import csv
import pandas as pd
import numpy as np
from pathlib import Path

from Block import Block
from Counter import Counter
from DetectionRecord import DetectionRecord
from Config import *
import progressbar
import time

record = None


class DetectionAuto(Detection):

    def __init__(self, args):
        super().__init__(args)
        self.record = DetectionRecord(self.counter)

    def process_video(self):  
        output = []
        timestamps = self.df['presentationTime'].to_numpy() * 1000.0 
    
        prev_count = 0
        b = progressbar.ProgressBar(maxval=self.frame_count)
        b.start()

        for idx, row in self.df.iterrows():
            frameResult = self.df.iloc[idx]
            time_ms = frameResult['presentationTime'] * 1000
            self.tracked = self.block_tracked.get(idx)
            self.counter.update_all(frameResult['state'], tracked=self.tracked)
            self.record.update_record(frameResult['state'], time_ms, self.current_frame)

            self.current_frame += 1
            b.update(self.current_frame) 

        timeTag = Path(self.video_path).stem.split('_')[2]
        with open(f'CMORE_Test_Results_{timeTag}.csv', 'w', newline='') as csvfile:
            fieldnames = self.record.keys
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.record.total_record)
            
        b.finish()

        self.cap.release()
        cv.destroyAllWindows()
