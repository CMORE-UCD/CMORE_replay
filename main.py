import argparse
import numpy as np

from TemporaryManual import TemporaryManual
from TemporaryAuto import TemporaryAuto
from Config import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', 
                        help='Path to the video file')
    parser.add_argument('--tracker', choices=MOTION_TRACKERS, default='bytetrack',
                        help='Tracker to use (default: bytetrack)')
    parser.add_argument('--track_buffer', type=int, default=30,
                        help='Frames to keep a lost track alive (bytetrack only, default: 30)')
    parser.add_argument('--mode', choices=MODES, default='manual', 
                        help='Mode to enter (default: manual)')
    parser.add_argument('--mvmt_threshold', choices=MVMT_THRESHOLD, default='0.25', 
                        help='Movement threshold to enter (default: 0.25)')
    
    # add new CLI arg 
        # changing threshold for movement
        # automating process, writing out to data file

    args = parser.parse_args()
    processor = None

    

    match args.mode:
        case 'auto':
            processor = TemporaryAuto(args)
        case _:
            processor = TemporaryManual(args)

    if processor is not None:
        processor.process_video()


if __name__ == "__main__":
    main()