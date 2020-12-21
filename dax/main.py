import cv2
import numpy as np
from particle import FollowParticle
from sparsedense.sparse_flow_vHappyDax import SparseHappyDax

video_cap = cv2.VideoCapture("http://192.168.0.47:8080/video")
print(video_cap)

ret, firstFrame = video_cap.read()
print(ret, firstFrame)
if ret is not False:
    sparse = SparseHappyDax()

    while video_cap.isOpened():
        ret, new_frame = video_cap.read()

        if ret is False or new_frame is None:
            break
        new_frame = cv2.resize(new_frame, (160, 90))

        sparse.run(new_frame)
        print(sparse.getPosition())
        print(sparse.getDirection())
        
        idk = tuple(sparse.getPosition())
        
        if (len(idk)!=0):
            new_frame = cv2.circle(new_frame, idk, 2, (255, 0, 0), -1)
        
        cv2.imshow("ree", new_frame)

    sparse.release()
    video_cap.release()