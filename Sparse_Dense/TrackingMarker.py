class TrackingMarker:

    #Number of positions the marker remembers
    frame_store_count = 5
    #Distance along either x or y that indicates if marker is moving
    movement_breakpoint = 2

    def __init__(self):
        self.positions = []
        self.is_Moving = False

    def getIsMoving(self):
        return self.is_Moving

    def getDirection(self):
        #How the marker has travelled from the first stored position to the last
        return (self.positions[-1][0] - self.positions[0][0], self.positions[-1][1] - self.positions[0][1])

    def addPosition(self, new_position):
        pos_x = new_position[0][0]
        pos_y = new_position[0][1]
        self.positions.append((pos_x, pos_y))

        #Remove the oldest position if past the limit
        if self.positions.__len__() > self.frame_store_count:
            self.positions.pop(0)

    def testSamePosition(self, test_position):
        pos_x = test_position[0][0]
        pos_y = test_position[0][1]

        return pos_x == self.positions[-1][0] and pos_y == self.positions[-1][1]

    def testMovement(self):
        if self.positions.__len__() > 1:
            #If difference is big enough, consider marker to be moving
            if abs(self.positions[0][0] - self.positions[-1][0]) > self.movement_breakpoint or \
                    abs(self.positions[0][1] - self.positions[-1][1]) > self.movement_breakpoint:
                self.is_Moving = True
            else:
                self.is_Moving = False
