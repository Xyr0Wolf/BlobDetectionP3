from copy import deepcopy
import numpy as np

class FollowParticle:
    currentPos = np.array([0, 0])
    targetPos  = np.array([0, 0])
    moveSpeed = 100

    def __init__(targetPos: np.array):
        self.targetPos = targetPos
        self.currentPos = deepcopy(self.targetPos)

    def update():
        dir = (self.targetPos-self.currentPos)
        print(dir)
        normalizedDir = dir/np.linalg.norm(dir)
        print(normalizedDir)
        self.currentPos += normalizedDir*moveSpeed