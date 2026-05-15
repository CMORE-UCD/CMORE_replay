from Temporary import Temporary
import cv2 as cv
import csv
import pandas as pd
import numpy as np
from pathlib import Path

from Block import Block
from Counter import Counter
from Config import *
import progressbar
import time


class TemporaryAuto(Temporary):

    def __init__(self, args):
        super().__init__(args)

    def process_video(self):  
        output = []
        timestamps = self.df['presentationTime'].to_numpy() * 1000.0 
    
        prev_count = 0
        # b = progressbar.ProgressBar(maxval=self.frame_count)
        b = progressbar.ProgressBar(maxval=500)
        b.start()
     
        # while self.current_frame <= self.frame_count:
        while self.current_frame < 500:
            self.cap.set(cv.CAP_PROP_POS_FRAMES, self.current_frame)
            ret, frame = self.cap.read()

            if not ret:
                break

            time_ms = self.cap.get(cv.CAP_PROP_POS_MSEC)
            match_idx = np.where(np.abs(timestamps - time_ms) < self.tolerance)[0]

            if len(match_idx) > 0:
                frameResult = self.df.iloc[match_idx[0]]
                self.tracked = self.block_tracked.get(match_idx[0])
                self.counter.update_all(frame, frameResult['state'], tracked=self.tracked)

                if self.counter.counter > prev_count:
                    prev_count += 1
                    curr_output = { #TO DO: figure out headers, reference google drive + old CMORE repo
                        'count_time': f"{time_ms}",
                        'count_frame': f"{self.current_frame}",
                        'current_count': f"{self.counter.counter}",
                        'valid_count': 'tbd'
                    }
                    output.append(curr_output)

            self.current_frame += 1
            b.update(self.current_frame) 

        timeTag = Path(self.video_path).stem.split('_')[2]
        with open(f'CMORE_Test_Results_{timeTag}.csv', 'w', newline='') as csvfile:
            fieldnames = ['count_time', 'count_frame', 'current_count', 'valid_count']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(output)
            
        b.finish()

        self.cap.release()
        cv.destroyAllWindows()
