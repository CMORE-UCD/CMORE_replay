from Counter import Counter


class DetectionRecord:

    keys = [
        'attempt_number',
        'attempt_start_time',
        'attempt_end_time',
        'attempt_start_frame',
        'attempt_end_frame',
        'cross_time',
        'cross_frame',
        'block_drop_time',
        'block_drop_frame',
        'detected_block_drop',
        'block_drop_total',
    ]

    def __init__(self, counter: Counter):
        self.counter = counter
        self.attempt_number = -1
        self.prev_state = None
        self.prev_count = 0
        self.total_record: list[dict] = []
        self.current_record: dict = {}
        self._reset_current_record()

    def update_record(self, curr_state: str, time_ms: float, current_frame: int):
        if self.prev_state == 'crossedBack' and curr_state == 'free':
            self._end_and_add_entry(time_ms, current_frame)
        self._update_state(curr_state, time_ms, current_frame)
        self._update_block_count(time_ms, current_frame)

    def _end_and_add_entry(self, time_ms: float, current_frame: int):
        self.current_record['attempt_end_time'] = time_ms
        self.current_record['attempt_end_frame'] = current_frame
        self.total_record.append(self.current_record.copy())
        self._reset_current_record()

    def _update_state(self, curr_state: str, time_ms: float, current_frame: int):
        if self.prev_state == 'free' and curr_state == 'detecting':
            self.current_record['attempt_start_time'] = time_ms
            self.current_record['attempt_start_frame'] = current_frame
        if self.prev_state == 'detecting' and curr_state == 'crossed':
            self.current_record['cross_time'] = time_ms
            self.current_record['cross_frame'] = current_frame
        self.prev_state = curr_state

    def _update_block_count(self, time_ms: float, current_frame: int):
        curr_count = self.counter.counter
        if curr_count > self.prev_count:
            self.current_record['block_drop_time'] = time_ms
            self.current_record['block_drop_frame'] = current_frame
            self.current_record['detected_block_drop'] = 1
            self.current_record['block_drop_total'] = curr_count
        self.prev_count = curr_count

    def _reset_current_record(self):
        self.attempt_number += 1
        self.current_record = {
            'attempt_number': self.attempt_number,
            'detected_block_drop': 0,
            'block_drop_total': self.counter.counter,
        }
