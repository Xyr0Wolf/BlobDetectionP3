import numpy as np
import cv2 as cv
from Opticalflow import Opticalflow


class Camshift:

    def __init__(self, video):
        # Read the input video
        cap = cv.VideoCapture(video)
        cap.set(1, 270)

        cv.imshow('frame', cap.read()[1])
        print(cap.read()[1].shape)
        # take first frame of the
        # video
        ret, frame = cap.read()

        # setup initial region of
        # tracker
        x, y, width, height = 200, 60, 200, 200
        track_window = (x, y,
                        width, height)

        # set up the Region of
        # Interest for tracking
        roi = frame[y:y + height,
              x: x + width]
        # convert ROI from BGR to
        # HSV format
        hsv_roi = cv.cvtColor(roi,
                              cv.COLOR_BGR2HSV)

        roi_hist = cv.calcHist([hsv_roi],
                               [0], None,
                               [180],
                               [0, 180])

        cv.normalize(roi_hist, roi_hist,
                     0, 255, cv.NORM_MINMAX)

        # Setup the termination criteria,
        # either 15 iteration or move by
        # atleast 2 pt
        term_crit = (cv.TERM_CRITERIA_EPS |
                     cv.TERM_CRITERIA_COUNT, 10, 1)

        while True:

            ret, frame = cap.read()

            #Resize the video frames.
            frame = cv.resize(frame,
                              (720, 720),
                              fx=0, fy=0,
                              interpolation=cv.INTER_CUBIC)



            cv.imshow('Original', frame)

            # perform thresholding on
            # the video frames
            ret1, frame1 = cv.threshold(frame,
                                        180, 155,
                                        cv.THRESH_TOZERO_INV)

            # convert from BGR to HSV
            # format.
            hsv = cv.cvtColor(frame1, cv.COLOR_BGR2HSV)

            dst = cv.calcBackProject([hsv],
                                     [0],
                                     roi_hist,
                                     [0, 180], 1)

            # apply Camshift to get the
            # new location
            ret2, track_window = cv.CamShift(dst,
                                             track_window,
                                             term_crit)

            # Draw it on image
            pts = cv.boxPoints(ret2)

            # convert from floating
            # to integer
            pts = np.int0(pts)

            # Draw Tracking window on the
            # video frame.
            Result = cv.polylines(frame,
                                  [pts],
                                  True,
                                  (0, 255, 255),
                                  2)

            cv.imshow('Camshift', Result)

            # set ESC key as the
            # exit button.
            k = cv.waitKey(30) & 0xff

            if k == 27:
                break

        # Release the cap object
        cap.release()

        # close all opened windows
        cv.destroyAllWindows()


if __name__ == '__main__':

    Opticalflow('videoplayback.mp4', 0)
    Camshift('videoplayback.mp4')
