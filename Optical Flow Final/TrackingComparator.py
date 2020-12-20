class TrackingComparator:
    # Class for storing information about how actual tracking compares to the desired result

    def __init__(self, frame, position, direction, desired_position, distance):
        self.frame = frame
        self.tracked_position = position
        self.tracked_direction = direction
        self.desired_position = desired_position
        self.distance = distance
