from Counter import Counter

class DetectionRecord:
    attempt_number = -1
    prev_state = None
    prev_count = 0
    keys = ['attempt_number',
            'attempt_start_time',
            'attempt_end_time', # what is an end of an attempt, when crossed back? or when picked up another block
            'attempt_start_frame',
            'attempt_end_frame',
            'cross_time',
            'cross_frame',
            'block_drop_time',
            'block_drop_frame',
            'detected_block_drop',
            'block_drop_total']
    current_record = {}
    total_record = []
    counter = None

    def __init__(self, counter):
        self.counter = counter

        self.reset_current_record()

    def update_record(self, curr_state, time_ms, current_frame):
        #states: free, detecting, crossed, crossedBack

        # if end of an attempt, add to list and clear current entry
        if self.prev_state is not None and self.prev_state == 'crossedBack' and curr_state == 'free':
            self.end_and_add_entry(time_ms, current_frame)

        self.update_state(time_ms, current_frame, curr_state)
        self.update_block_count(time_ms, current_frame)
    
    def end_and_add_entry(self, time_ms, current_frame):
        self.current_record['attempt_end_time'] = time_ms
        self.current_record['attempt_end_frame'] = current_frame

        self.total_record.append(self.current_record.copy())

        self.reset_current_record()
        
        
    def update_state(self, time_ms, current_frame, curr_state):
        if self.prev_state is not None:

            if self.prev_state == 'free' and curr_state == 'detecting':
                self.current_record['attempt_start_time'] = time_ms
                self.current_record['attempt_start_frame'] = current_frame

            if self.prev_state == 'detecting' and curr_state == 'crossed':
                self.current_record['cross_time'] = time_ms
                self.current_record['cross_frame'] = current_frame
        
        self.prev_state = curr_state
    
    def update_block_count(self, time_ms, current_frame):
        curr_count = self.counter.counter

        if curr_count > self.prev_count:
            self.current_record['block_drop_time'] = time_ms
            self.current_record['block_drop_frame'] = current_frame

            self.current_record['detected_block_drop'] = 1

            self.current_record['block_drop_total'] = curr_count
        
        self.prev_count = curr_count

    def reset_current_record(self):
        # clear current_record
        self.current_record.clear()
        self.current_record.fromkeys(self.keys)

        # add attempt number to new record
        self.attempt_number += 1
        self.current_record['attempt_number'] = self.attempt_number

        # initialize necessary block fields
        self.current_record['detected_block_drop'] = 0
        self.current_record['block_drop_total'] = self.counter.counter


    
