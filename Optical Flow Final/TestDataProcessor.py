import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt
from TrackingComparator import TrackingComparator
from TrackingGraphics import TrackingGraphics as Graphics

class DataProcessor:

    def __init__(self):
        self.__tracking_comparisons = []

    def addClickPosition(self, frame, position, direction, clickPos):

        # Find the difference in x and y, then use pythagoras to calculate the distance
        difference_x = abs(clickPos[0] - position[0])
        difference_y = abs(clickPos[1] - position[1])
        a_squared = difference_x * difference_x
        b_squared = difference_y * difference_y
        distance = math.sqrt(a_squared + b_squared)

        tracking_comparison = TrackingComparator(frame, position, direction, clickPos, distance)
        self.__tracking_comparisons.append(tracking_comparison)

    def showPositionComparisons(self):
        for comparison in self.__tracking_comparisons:

            draw_mask = np.zeros_like(comparison.frame)
            Graphics.drawTrackedPos(comparison.tracked_position, draw_mask)
            Graphics.drawDesiredPos(comparison.desired_position, draw_mask)
            Graphics.drawTrackedDir(comparison.tracked_position, comparison.tracked_direction, draw_mask)

            tracked_text = f"Tracked {comparison.tracked_position}"
            desired_text = f"Desired {comparison.desired_position}"
            distance_text = f"Distance {comparison.distance}"

            frame_height = comparison.frame.shape[0]
            print(f"{tracked_text} - {distance_text} - {desired_text}")
            Graphics.writeTrackedPosText(draw_mask, tracked_text, frame_height)
            Graphics.writeDesiredPosText(draw_mask, desired_text, frame_height)
            Graphics.writePosDifference(draw_mask, distance_text, frame_height)

            # Overlays the optical flow tracks on the original frame
            output = cv.add(comparison.frame, draw_mask)

            # Opens a new window and displays the output frame
            cv.imshow("sparse optical flow", output)

            if cv.waitKey(1000) & 0xFF == ord('q'):
                1+1

    def showDataPlot(self):

        data = []
        for comparison in self.__tracking_comparisons:
            data.append(comparison.distance)

        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4), sharey=True)

        ax1.set_title('Accuracy box plot')
        ax1.set_ylabel('Observed values')
        ax1.boxplot(data)

        ax2.set_title('Accuracy violin plot')
        ax2.set_ylabel('Observed values')
        ax2.violinplot(data)

        plt.show()
