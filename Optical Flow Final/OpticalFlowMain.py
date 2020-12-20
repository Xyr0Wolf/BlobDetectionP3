from OpticalFlowSparse import OpticalFlowSparse
from OpticalFlowSparseDataTest import OpticalFlowSparseDataTest
import cv2 as cv

#https://nanonets.com/blog/optical-flow/

def doOpticalFlow():
    # Test video
    # video_cap = cv.VideoCapture("CatLaserVideoV4.mp4")
    # Cropped test video focusing on the cat
    video_cap = cv.VideoCapture("CatLaserVideoV4Crop.mp4")
    # Alternate test video
    # video_cap = cv.VideoCapture("catwalk.mp4")

    doDataTest = False
    if not doDataTest:
        opticalFlow = OpticalFlowSparse()
    else:
        # Data test will allow click input to compare the detected position against the tracked position
        opticalFlow = OpticalFlowSparseDataTest()

    while video_cap.isOpened():
        ret, new_frame = video_cap.read()

        if ret is False or new_frame is None:
            break

        opticalFlow.run(new_frame)
        # print(sparse.getPosition())
        # print(sparse.getDirection())
    opticalFlow.release()
    video_cap.release()

doOpticalFlow()

