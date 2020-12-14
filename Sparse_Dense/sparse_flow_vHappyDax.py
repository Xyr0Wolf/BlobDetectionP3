import cv2 as cv
import numpy as np
from TrackingMarker import TrackingMarker

class SparseHappyDax():

    def __init__(self):
        print("using the modified sparse optical flow")

        # Parameters for Shi-Tomasi corner detection
        # Original parameters maxCorners = 300, qualityLevel = 0.2, minDistance = 2, blockSize = 7
        self.feature_params = dict(maxCorners=1000, qualityLevel=0.1, minDistance=1, blockSize=4)

        # Parameters for Lucas-Kanade optical flow
        # Original parameters winSize=(15, 15), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)
        self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                         criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

        # Colorcoding for marker graphics
        self.color_still = (0, 255, 0)
        self.color_moving = (255, 0, 0)
        self.color_pos_dir = (0, 0, 255)

        # Number of frames before the code will reset markers and find new tracking points
        self.no_movement_timer = 0
        self.no_movement_reset_time = 3
        # Number of active markers required to prevent reset
        self.moving_markers_lock = 2

        self.blur_power = 15

        self.tracking_markers = []
        self.moving_markers = []

        self.prev_points = None
        self.prev_gray_frame = None

        self.draw_mask = None

    def findTrackingPoints(self, video_frame_gray):

        self.tracking_markers.clear()

        # Finds the strongest corners in the first frame by Shi-Tomasi method - we will track the optical flow for these corners
        # https://docs.opencv.org/3.0-beta/modules/imgproc/doc/feature_detection.html#goodfeaturestotrack
        found_points = cv.goodFeaturesToTrack(video_frame_gray, mask=None, **self.feature_params)
        # Creates an image filled with zero intensities with the same dimensions as the frame - for later drawing purposes

        if found_points is not None:
            # Recreate the list of markers using the newly found tracking points
            for point in found_points:
                new_marker = TrackingMarker()
                new_marker.addPosition(point)
                self.tracking_markers.append(new_marker)

        return found_points

    def clearLostMarkers(self):
        for x in range(self.prev_points.__len__()):
            while self.tracking_markers[x].testSamePosition(self.prev_points[x]) == False:
                self.tracking_markers.remove(self.tracking_markers[x])
        # Above loop won't remove markers past the last index, do that here
        while self.tracking_markers.__len__() > self.prev_points.__len__():
            self.tracking_markers.remove(self.tracking_markers[-1])

    def updateTrackingMarkers(self, cur_points):
        # Add the new positions to our marker objects and test if they're moving
        for x in range(cur_points.__len__()):
            self.tracking_markers[x].addPosition(cur_points[x])
            self.tracking_markers[x].testMovement()

        # Reset list of moving markers and reassign
        self.moving_markers.clear()
        for marker in self.tracking_markers:
            if marker.getIsMoving():
                self.moving_markers.append(marker)

    def optimizeFrame(self, video_frame):
        video_frame = self.blurFrame(video_frame)
        return video_frame

    def blurFrame(self, video_frame):
        return cv.GaussianBlur(video_frame, (self.blur_power, self.blur_power), 0)

    def run(self, video_frame):

        if video_frame is None:
            return

        optimized_frame = video_frame.copy()
        optimized_frame = self.optimizeFrame(optimized_frame)

        # Optical flow requires 2 frames to compare, if we don't have a previous, simply generate and return
        # Note, this is expected to happen the first time, should never happen again
        if self.prev_points is None:
            self.prev_gray_frame = cv.cvtColor(optimized_frame, cv.COLOR_BGR2GRAY)
            self.prev_points = self.findTrackingPoints(self.prev_gray_frame)
            return

        cur_points = None
        cur_frame = optimized_frame
        cur_frame = self.optimizeFrame(cur_frame)
        cur_gray_frame = cv.cvtColor(cur_frame, cv.COLOR_BGR2GRAY)

        if self.prev_points is not None:
            # Calculates sparse optical flow by Lucas-Kanade method
            # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowpyrlk
            cur_points, status, error = cv.calcOpticalFlowPyrLK(self.prev_gray_frame, cur_gray_frame, self.prev_points, None, **self.lk_params)

            # If points were lost we need to adjust our markers appropriately
            if self.prev_points.__len__() != self.tracking_markers.__len__():
                self.clearLostMarkers()

        if cur_points is not None and self.no_movement_timer < self.no_movement_reset_time:
            # Selects good feature points for next position
            good_new = cur_points[status == 1]

            self.updateTrackingMarkers(cur_points)

            #If we have enough moving markers, reset the reset counter
            if self.moving_markers.__len__() > self.moving_markers_lock:
                self.no_movement_timer = 0
            else:
                self.no_movement_timer += 1
        else:
            #Reset
            self.prev_points = self.findTrackingPoints(cur_gray_frame)
            self.prev_gray_frame = cur_gray_frame
            self.no_movement_timer = 0
            return

        #Reset draw_mask
        self.draw_mask = np.zeros_like(cur_frame)

        #Draw the graphics for each marker, old way and new way
        self.drawMovementTracks()

        #Draw center position and average direction of moving markers
        if self.moving_markers.__len__() > 1:
            self.drawPosAndDir()

        # Overlays the optical flow tracks on the original frame
        output = cv.add(cur_frame, self.draw_mask)

        # Updates previous frame
        self.prev_gray_frame = cur_gray_frame.copy()
        self.prev_points = good_new.reshape(-1, 1, 2)
        # Not entirely sure why good_new couldn't just have been cur_points, but this works better?

        # Opens a new window and displays the output frame
        cv.imshow("sparse optical flow", output)
        cv.imshow("grey image", cur_gray_frame)

        # This is literally just here so I can run the code and have the images render, delete whenever
        if cv.waitKey(1):
            1+1

    def Release(self):
        # The following frees up resources and closes all windows
        cv.destroyAllWindows()

    def drawMovementTracks(self):
        # Draws the optical flow tracks
        for marker in self.tracking_markers:
            #Determine color
            if marker.getIsMoving():
                trackColor = self.color_moving
            else:
                trackColor = self.color_still

            #Draw a line for each set of positions
            for x in range(marker.positions.__len__() -1):
                point1 = marker.positions[x]
                point2 = marker.positions[x+1]
                self.draw_mask = cv.line(self.draw_mask, point1, point2, trackColor, 2)
            #Draw circle at current position
            self.draw_mask = cv.circle(self.draw_mask, marker.positions[-1], 3, trackColor, -1)

    def getCenterPoint(self):

        center_point = (0,0)
        if self.moving_markers.__len__() > 0:
            add_pos_x = 0
            add_pos_y = 0
            for marker in self.moving_markers:
                position = marker.positions[-1]
                add_pos_x += position[0]
                add_pos_y += position[1]

            average_x = add_pos_x / self.moving_markers.__len__()
            average_y = add_pos_y / self.moving_markers.__len__()
            center_point = (int(average_x), int(average_y))
        return center_point

    def getMoveDirection(self):
        move_direction = (0, 0)
        if self.moving_markers.__len__() > 0:
            add_dir_x = 0
            add_dir_y = 0
            for marker in self.moving_markers:
                direction = marker.getDirection()
                add_dir_x += direction[0]
                add_dir_y += direction[1]

            average_dir_x = add_dir_x / self.moving_markers.__len__()
            average_dir_y = add_dir_y / self.moving_markers.__len__()
            move_direction = (int(average_dir_x), int(average_dir_y))
        return move_direction

    def drawPosAndDir(self):
        self.drawPos()
        self.drawDir()

    def drawPos(self):
        position = self.getCenterPoint()
        self.draw_mask = cv.circle(self.draw_mask, position, 10, self.color_pos_dir, -1)

    def drawDir(self):
        position = self.getCenterPoint()
        direction = self.getMoveDirection()

        offsetPos = (position[0] + direction[0], position[1] + direction[1])
        self.draw_mask = cv.line(self.draw_mask, position, offsetPos, self.color_pos_dir, 2)