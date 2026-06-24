MOTION_TRACKERS = ['bytetrack', 'ocsort', 'sfsort', 'boosttrack']
MODES = ['manual', 'auto']

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green
TRACKER_COLOR = (255, 255, 0)  # cyan for tracker boxes
COUNTER_COLOR = (138,43,226) # violet for counting boxes

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