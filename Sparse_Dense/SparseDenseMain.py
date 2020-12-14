from Sparse_Dense.sparse_flow import SparseOpticalFlow
from Sparse_Dense.dense_flow import DenseOpticalFlow
from Sparse_Dense.sparse_flow_v2 import SparseOpticalFlowMod
from Sparse_Dense.sparse_flow_vHappyDax import SparseHappyDax
import cv2 as cv
import os

#https://nanonets.com/blog/optical-flow/
sparse_or_dense = True
sparse_modified = True
makeDaxHappy = True

def doSparse():
    if sparse_modified:
        sparseMod = SparseOpticalFlowMod
        sparseMod.Start(sparseMod)
    else:
        sparse = SparseOpticalFlow
        sparse.Start(sparse)

def doDense():
    dense = DenseOpticalFlow
    dense.Start(dense)


if makeDaxHappy:

    video_name = "videoplayback.mp4"
    vidPath = os.path.abspath(os.path.join(os.path.dirname("BlobDetection"), '..', video_name))
    video_cap = cv.VideoCapture(vidPath)

    ret, firstFrame = video_cap.read()
    if ret is not False:
        sparse = SparseHappyDax()

        while video_cap.isOpened():
            ret, new_frame = video_cap.read()

            if ret is False or new_frame is None:
                break

            sparse.run(new_frame)
            print(sparse.getCenterPoint())
            print(sparse.getMoveDirection())
        sparse.Release()
        video_cap.release()
else:

    if sparse_or_dense:
        doSparse()
    else:
        doDense()

