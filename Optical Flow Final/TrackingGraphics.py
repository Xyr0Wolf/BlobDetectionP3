import cv2 as cv

class TrackingGraphics:

    # Colors used for graphics
    __color_still = (0, 255, 0)
    __color_moving = (255, 0, 0)
    __color_position_tracked = (0, 0, 255)
    __color_position_desired = (200, 0, 200)
    __color_direction = (0, 200, 200)
    __font = cv.FONT_HERSHEY_COMPLEX

    def drawMovementTracks(draw_mask, tracking_markers):
        # Draws the optical flow tracks
        for marker in tracking_markers:
            #Determine color
            if marker.getIsMoving():
                trackColor = TrackingGraphics.__color_moving
            else:
                trackColor = TrackingGraphics.__color_still

            #Draw a line for each set of positions
            for x in range(marker.positions.__len__() -1):
                point1 = marker.positions[x]
                point2 = marker.positions[x+1]
                cv.line(draw_mask, point1, point2, trackColor, 2)
            #Draw circle at current position
            cv.circle(draw_mask, marker.positions[-1], 3, trackColor, -1)

    def drawTrackedPos(position, draw_mask):
        cv.circle(draw_mask, position, 10, TrackingGraphics.__color_position_tracked, -1)

    def drawDesiredPos(position, draw_mask):
        cv.circle(draw_mask, position, 10, TrackingGraphics.__color_position_desired, -1)

    def drawTrackedDir(position, direction, draw_mask):
        offsetPos = (position[0] + direction[0], position[1] + direction[1])
        cv.line(draw_mask, position, offsetPos, TrackingGraphics.__color_direction, 2)

    # Text is written in the bottom left corner of the window
    def writeTrackedPosText(draw_mask, text, frame_height):
        cv.putText(draw_mask, text, (10, frame_height - 90),
                   TrackingGraphics.__font, 1, TrackingGraphics.__color_position_tracked, 2)

    def writeDesiredPosText(draw_mask, text, frame_height):
        cv.putText(draw_mask, text, (10, frame_height - 50),
                   TrackingGraphics.__font, 1, TrackingGraphics.__color_position_desired, 2)

    def writePosDifference(draw_mask, text, frame_height):
        cv.putText(draw_mask, text, (10, frame_height - 10),
                   TrackingGraphics.__font, 1, TrackingGraphics.__color_still, 2)