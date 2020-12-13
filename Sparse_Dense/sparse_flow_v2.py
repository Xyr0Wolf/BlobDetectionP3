import cv2 as cv
import numpy as np
import os
from Sparse_Dense.TrackingMarker import TrackingMarker

class SparseOpticalFlowMod():

    def Start(self):
        print("using the modified sparse optical flow")

        # Parameters for Shi-Tomasi corner detection
        # Original parameters maxCorners = 300, qualityLevel = 0.2, minDistance = 2, blockSize = 7
        self.feature_params = dict(maxCorners=300, qualityLevel=0.2, minDistance=2, blockSize=7)
        # Parameters for Lucas-Kanade optical flow
        # Original parameters winSize=(15, 15), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)
        self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                         criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

        # The video feed is read in as a VideoCapture object
        # video_name = "shibuya.mp4"
        # video_name = "unitytestclip.mp4"
        # video_name = "catwalk4.mp4"
        # video_name = "CatLaserVideoV2.mp4"
        video_name = "videoplayback.mp4"
        vidPath = os.path.abspath(os.path.join(os.path.dirname("BlobDetection"), '..', video_name))
        video_cap = cv.VideoCapture(vidPath)

        ret, firstFrame = video_cap.read()
        if ret is False:
            return

        self.trackProcessing(self, video_cap)

    feature_params = None
    lk_params = None

    #Used for cropping the image down, can be used to define area for markers we care about?
    frame_margin_x = 0
    frame_margin_y = 0

    #Intensity of blur effect
    blur_power = 15

    tracking_markers = []
    moving_markers = []

    def findTrackingPoints(self, video_frame_gray):

        self.tracking_markers.clear()

        # Finds the strongest corners in the first frame by Shi-Tomasi method - we will track the optical flow for these corners
        # https://docs.opencv.org/3.0-beta/modules/imgproc/doc/feature_detection.html#goodfeaturestotrack
        found_points = cv.goodFeaturesToTrack(video_frame_gray, mask=None, **self.feature_params)
        # Creates an image filled with zero intensities with the same dimensions as the frame - for later drawing purposes

        if found_points is not None:
            #Recreate the list of markers using the newly found tracking points
            for point in found_points:
                new_marker = TrackingMarker()
                new_marker.addPosition(point)
                self.tracking_markers.append(new_marker)

        return found_points

    def optimizeFrame(self, video_frame):
        #video_frame = self.cropFrame(self, video_frame)
        video_frame = self.blurFrame(self, video_frame)
        return video_frame

    def cropFrame(self, video_frame):
        #Bad, currently assumes 50% crop from bottom and 25% crop from both left and right
        return video_frame[0:self.frame_margin_x, self.frame_margin_y:video_frame.shape[1] - self.frame_margin_y]

    def blurFrame(self, video_frame):
        return cv.GaussianBlur(video_frame, (self.blur_power, self.blur_power), 0)

    def trackProcessing(self, video_cap):

        # ret = a boolean return value from getting the frame, first_frame = the first frame in the entire video sequence
        ret, first_frame = video_cap.read()

        if ret is False or first_frame is None:
            return

        #Assigns crop values, dependant on image being processed please adjust or delete this isn't smart
        self.frame_margin_x = int(first_frame.shape[0] / 2)
        self.frame_margin_y = int(first_frame.shape[1] / 4)

        first_frame = self.optimizeFrame(self, first_frame)
        first_frame_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)

        #Colorcoding for marker graphics
        color_still = (0, 255, 0)
        color_moving = (255, 0, 0)
        color_pos_dir = (0, 0, 255)

        #Number of frames before the code will reset markers and find new tracking points
        frameCountSession = 0
        sessionFrameLimit = 3
        #Number of active markers required to prevent reset
        movingMarkersLock = 2

        #Assign to previous frame, loop will essentially start on frame 2
        prev_points = self.findTrackingPoints(self, first_frame_gray)
        prev_gray_frame = first_frame_gray

        while(video_cap.isOpened()):
            # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
            ret, cur_frame = video_cap.read()

            if ret is False or cur_frame is None:
                break

            cur_points = None
            cur_frame = self.optimizeFrame(self, cur_frame)
            cur_gray_frame = cv.cvtColor(cur_frame, cv.COLOR_BGR2GRAY)

            if prev_points is not None:
                # Calculates sparse optical flow by Lucas-Kanade method
                # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowpyrlk
                cur_points, status, error = cv.calcOpticalFlowPyrLK(prev_gray_frame, cur_gray_frame, prev_points, None, **self.lk_params)

                #If points were lost we need to adjust our markers appropriately
                if prev_points.__len__() != self.tracking_markers.__len__():
                    for x in range(prev_points.__len__()):
                        while self.tracking_markers[x].testSamePosition(prev_points[x]) == False:
                            self.tracking_markers.remove(self.tracking_markers[x])
                    #Above loop won't remove markers past the last index of prev_points, do that here
                    while self.tracking_markers.__len__() > prev_points.__len__():
                        self.tracking_markers.remove(self.tracking_markers[-1])

                # Selects good feature points for previous position
                # Was used when drawing the old way, we don't do that anymore so it's unused
                good_prev = prev_points[status == 1]

            if cur_points is not None and frameCountSession < sessionFrameLimit:
                # Selects good feature points for next position
                # Still used to assign to prev_points, not sure if replaceable
                good_new = cur_points[status == 1]

                #Add the new positions to our marker objects and test if they're moving
                for x in range(cur_points.__len__()):
                    self.tracking_markers[x].addPosition(cur_points[x])
                    self.tracking_markers[x].testMovement()

                #Reset list of moving markers and reassign
                self.moving_markers.clear()
                for marker in self.tracking_markers:
                    if marker.getIsMoving():
                        self.moving_markers.append(marker)

                #If we have enough moving markers, reset the reset counter
                if self.moving_markers.__len__() > movingMarkersLock:
                    frameCountSession = 0
                else:
                    frameCountSession += 1
            else:
                #Reset
                prev_points = self.findTrackingPoints(self, cur_gray_frame)
                frameCountSession = 0
                continue

            draw_mask = np.zeros_like(cur_frame)
            #Draw the graphics for each marker, old way and new way
            #self.drawFlowTracks(self, cur_frame, draw_mask, good_new, good_prev, color_still)
            self.drawMovementTracks(self, draw_mask, color_still, color_moving)

            #Draw center position and average direction of moving markers
            if self.moving_markers.__len__() > 1:
                self.drawPosAndDir(self, draw_mask, color_pos_dir)

            # Overlays the optical flow tracks on the original frame
            output = cv.add(cur_frame, draw_mask)
            # Updates previous frame
            prev_gray_frame = cur_gray_frame.copy()
            # Updates previous good feature points
            prev_points = good_new.reshape(-1, 1, 2)
            # Opens a new window and displays the output frame
            cv.imshow("sparse optical flow", output)
            cv.imshow("grey image", cur_gray_frame)
            # Pause time between frames. Press the 'q' key to advance frames manually
            if cv.waitKey(30) & 0xFF == ord('q'):
                continue

        # The following frees up resources and closes all windows
        video_cap.release()
        cv.destroyAllWindows()

    def drawFlowTracks(self, cur_frame, draw_mask, good_new, good_prev, color_still):
        # Draws the optical flow tracks --- old method, kept for reference
        for i, (new, prev) in enumerate(zip(good_new, good_prev)):
            # Returns a contiguous flattened array as (x, y) coordinates for new point
            a, b = new.ravel()
            # Returns a contiguous flattened array as (x, y) coordinates for old point
            c, d = prev.ravel()
            # Draws line between new and old position with green color and 2 thickness
            draw_mask = cv.line(draw_mask, (a, b), (c, d), color_still, 2)
            # Draws filled circle (thickness of -1) at new position with green color and radius of 3
            cur_frame = cv.circle(cur_frame, (a, b), 3, color_still, -1)

    def drawMovementTracks(self, draw_mask, color_still, color_moving):
        # Draws the optical flow tracks
        for marker in self.tracking_markers:
            #Determine color
            if marker.getIsMoving():
                trackColor = color_moving
            else:
                trackColor = color_still

            #Draw a line for each set of positions
            for x in range(marker.positions.__len__() -1):
                point1 = marker.positions[x]
                point2 = marker.positions[x+1]
                draw_mask = cv.line(draw_mask, point1, point2, trackColor, 2)
            #Draw circle at current position
            draw_mask = cv.circle(draw_mask, marker.positions[-1], 3, trackColor, -1)

    def drawPosAndDir(self, draw_mask, color_pos_dir):
        add_pos_x = 0
        add_pos_y = 0
        for marker in self.moving_markers:
            position = marker.positions[-1]
            add_pos_x += position[0]
            add_pos_y += position[1]

        average_pos_x = int(add_pos_x / self.moving_markers.__len__())
        average_pos_y = int(add_pos_y / self.moving_markers.__len__())
        average_pos = (average_pos_x, average_pos_y)
        draw_mask = cv.circle(draw_mask, average_pos, 10, color_pos_dir, -1)

        add_dir_x = 0
        add_dir_y = 0
        for marker in self.moving_markers:
            direction = marker.getDirection()
            add_dir_x += direction[0]
            add_dir_y += direction[1]

        average_dir_x = int(add_dir_x / self.moving_markers.__len__())
        average_dir_y = int(add_dir_y / self.moving_markers.__len__())
        average_dir = (average_dir_x, average_dir_y)

        offsetPos = (average_pos[0] + average_dir[0], average_pos[1] + average_dir[1])
        draw_mask = cv.line(draw_mask, average_pos, offsetPos, color_pos_dir, 2)