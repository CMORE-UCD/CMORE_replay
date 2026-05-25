import argparse

from Config import *
from DetectionAuto import DetectionAuto
from DetectionManual import DetectionManual


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
    parser.add_argument('--mvmt_threshold', type=float, default=0.25,
                        help='Movement threshold (default: 0.25)')

    args = parser.parse_args()

    match args.mode:
        case 'auto':
            processor = DetectionAuto(args)
        case _:
            processor = DetectionManual(args)

    processor.process_video()


if __name__ == "__main__":
    main()
