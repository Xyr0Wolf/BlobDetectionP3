import cv2 as cv
import numpy as np
from TrackingMarker import TrackingMarker
from TrackingGraphics import TrackingGraphics as Graphics
from TestDataProcessor import DataProcessor

class OpticalFlowSparseDataTest:

    def __init__(self):
        print("Using the accuracy test version")

        # Parameters for Shi-Tomasi corner detection
        # Original parameters maxCorners = 300, qualityLevel = 0.2, minDistance = 2, blockSize = 7
        self.__feature_params = dict(maxCorners=300, qualityLevel=0.1, minDistance=1, blockSize=2)

        # Parameters for Lucas-Kanade optical flow
        # Original parameters winSize=(15, 15), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)
        self.__lk_params = dict(winSize=(35, 35), maxLevel=2,
                                criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

        # Number of frames before the code will reset markers and find new tracking points
        self.__no_movement_timer = 0
        self.__no_movement_reset_time = 2
        # Number of active markers required to prevent reset, set this value really high to disable lock feature
        self.__moving_markers_lock = 4

        self.__blur_power = 15

        # Stored data from previous frame
        self.__prev_points = None
        self.__prev_gray_frame = None

        # Store optical flow markers
        self.__tracking_markers = []
        self.__moving_markers = []

        # The readable position and direction data
        self.__tracked_position = ()
        self.__tracked_direction = ()

        # Data test processing
        self.__data_processor = DataProcessor()

        cv.namedWindow("sparse optical flow")
        cv.setMouseCallback('sparse optical flow', self.__getClickPosition)

    def run(self, video_frame):

        if video_frame is None:
            return

        optimized_frame = video_frame.copy()
        optimized_frame = self.__optimizeFrame(optimized_frame)

        # Optical flow requires 2 frames to compare, if we don't have a previous, simply generate and return
        # Note, this is expected to happen the first time, should never happen again
        if self.__prev_points is None:
            self.__prev_gray_frame = cv.cvtColor(optimized_frame, cv.COLOR_BGR2GRAY)
            self.__prev_points = self.__findTrackingPoints(self.__prev_gray_frame)
            return

        #print("Frame begin")

        cur_frame = optimized_frame
        cur_gray_frame = cv.cvtColor(cur_frame, cv.COLOR_BGR2GRAY)

        # Calculates sparse optical flow by Lucas-Kanade method
        # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowpyrlk
        cur_points, status, error = cv.calcOpticalFlowPyrLK(self.__prev_gray_frame, cur_gray_frame, self.__prev_points, None, **self.__lk_params)

        # We need to actively be looking for new tracking point so we can recognize when the cat enter view
        if cur_points is not None and self.__no_movement_timer < self.__no_movement_reset_time:
            # Selects good feature points for next position
            self.__updateTrackingMarkers(cur_points)
            good_new = cur_points[status == 1]

            # If points were lost we need to adjust our markers appropriately
            if good_new.__len__() != self.__tracking_markers.__len__():
                self.__clearBadMarkers(good_new)

            #If we have enough moving markers, reset the reset counter
            if self.__moving_markers.__len__() > self.__moving_markers_lock:
                self.__no_movement_timer = 0
            else:
                self.__no_movement_timer += 1

            self.__calculateCenterPoint()
            self.__calculateMoveDirection()

            # Everything related to drawing is probably unnecessary for anyone using this code? just here for local testing
            self.__drawTracking(cur_frame)

            # Updates previous frame
            self.__prev_gray_frame = cur_gray_frame.copy()
            self.__prev_points = good_new.reshape(-1, 1, 2)
        else:
            #Reset
            self.__prev_points = self.__findTrackingPoints(cur_gray_frame)
            self.__prev_gray_frame = cur_gray_frame
            self.__no_movement_timer = 0

        #print("Frame end")

    def __findTrackingPoints(self, video_frame_gray):

        self.__tracking_markers.clear()

        # Finds the strongest corners in the first frame by Shi-Tomasi method - we will track the optical flow for these corners
        # https://docs.opencv.org/3.0-beta/modules/imgproc/doc/feature_detection.html#goodfeaturestotrack
        found_points = cv.goodFeaturesToTrack(video_frame_gray, mask=None, **self.__feature_params)
        # Creates an image filled with zero intensities with the same dimensions as the frame - for later drawing purposes

        if found_points is not None:
            # Recreate the list of markers using the newly found tracking points
            for point in found_points:
                new_marker = TrackingMarker()
                new_marker.addPosition(point)
                self.__tracking_markers.append(new_marker)

        return found_points

    def __clearBadMarkers(self, good_new):
        for x in range(good_new.__len__()):
            while self.__tracking_markers[x].testSamePosition(good_new[x]) == False:
                self.__tracking_markers.remove(self.__tracking_markers[x])
        # Above loop won't remove markers past the last index, do that here
        while self.__tracking_markers.__len__() > good_new.__len__():
            self.__tracking_markers.remove(self.__tracking_markers[-1])

    def __updateTrackingMarkers(self, cur_points):
        # Add the new positions to our marker objects and test if they're moving
        for x in range(cur_points.__len__()):
            self.__tracking_markers[x].addPosition(cur_points[x])
            self.__tracking_markers[x].testMovement()

        # Reset list of moving markers and reassign
        self.__moving_markers.clear()
        for marker in self.__tracking_markers:
            if marker.getIsMoving():
                self.__moving_markers.append(marker)

    def __optimizeFrame(self, video_frame):
        video_frame = self.__blurFrame(video_frame)
        return video_frame

    def __blurFrame(self, video_frame):
        return cv.GaussianBlur(video_frame, (self.__blur_power, self.__blur_power), 0)

    def __drawTracking(self, cur_frame):
        #Reset draw_mask
        draw_mask = np.zeros_like(cur_frame)

        #Draw the graphics for each marker, old way and new way
        Graphics.drawMovementTracks(draw_mask, self.__tracking_markers)

        #Draw center position and average direction of moving markers
        if self.__moving_markers.__len__() > 1:
            Graphics.drawTrackedPos(self.getPosition(), draw_mask)
            Graphics.drawTrackedDir(self.getPosition(), self.getDirection(), draw_mask)

        # Overlays the optical flow tracks on the original frame
        output = cv.add(cur_frame, draw_mask)

        # Opens a new window and displays the output frame
        cv.imshow("sparse optical flow", output)
        #cv.imshow("grey image", self.cur_gray_frame)

        # Pause each frame long enough for the user to put the desired location, advance manually by pressing 'q'
        if cv.waitKey(5000) & 0xFF == ord('q'):
            1+1

    def release(self):
        # The following frees up resources and closes all windows
        cv.destroyAllWindows()
        # This is kinda pointless when being used by another system?

        # Sneaking in the data display here so we don't need another function to call
        self.__data_processor.showPositionComparisons()
        self.__data_processor.showDataPlot()

    def __calculateCenterPoint(self):

        # Center point is calculated by averaging the positions of all moving markers
        center_point = (0,0)
        if self.__moving_markers.__len__() > 0:
            add_pos_x = 0
            add_pos_y = 0
            for marker in self.__moving_markers:
                position = marker.positions[-1]
                add_pos_x += position[0]
                add_pos_y += position[1]

            average_x = add_pos_x / self.__moving_markers.__len__()
            average_y = add_pos_y / self.__moving_markers.__len__()
            center_point = (int(average_x), int(average_y))
        self.__tracked_position = center_point

    def getPosition(self):
        return self.__tracked_position

    def __calculateMoveDirection(self):

        # Direction is calculated by averaging the directions of all moving markers
        move_direction = (0, 0)
        if self.__moving_markers.__len__() > 0:
            add_dir_x = 0
            add_dir_y = 0
            for marker in self.__moving_markers:
                direction = marker.getDirection()
                add_dir_x += direction[0]
                add_dir_y += direction[1]

            average_dir_x = add_dir_x / self.__moving_markers.__len__()
            average_dir_y = add_dir_y / self.__moving_markers.__len__()
            move_direction = (int(average_dir_x), int(average_dir_y))
        self.__tracked_direction = move_direction

    def getDirection(self):
        return self.__tracked_direction

    def __getClickPosition(self, event, x, y, flags, param):
        # This consistently happens AFTER the position has been calculated each frame
        if event == cv.EVENT_LBUTTONDOWN:
            self.__data_processor.addClickPosition(self.__cur_frame, self.getPosition(), self.getDirection(), (x, y))
