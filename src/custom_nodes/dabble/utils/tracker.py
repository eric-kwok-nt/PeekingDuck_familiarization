from typing import Union, Set
import numpy as np
import cv2
from ..sort_tracker.utils import iou
from copy import copy
import pdb


class Tracked_Obj:

    def __init__(self, current_bbox: Union[list, np.ndarray], iou_threshold: float, ma_window: int, look_back_period: int):
        self.prev_bbox = copy(current_bbox) # bbox before
        self.cur_bbox = None # bbox after
        self.iou_threshold = iou_threshold
        self.prev_centroid = self.find_centroid(self.prev_bbox)  
        self.cur_centroid = None
        self.height = self.prev_bbox[3] - self.prev_bbox[1]
        self.ma_window = ma_window
        self.ma_bbox = [self.prev_bbox for _ in range(self.ma_window)]
        self.count = 1
        self.look_back_period = look_back_period
        self.look_back_list = [self.prev_bbox for _ in range(self.look_back_period)]
        self.look_back_count = 0

    def update_pos(self, current_bbox: Union[list, np.ndarray]):
        # Get MA of previous bbox and add into the look back list
        self.prev_bbox = np.mean(self.ma_bbox, axis=0)
        if self.look_back_count == self.look_back_period:
            self.look_back_count = 0
        self.look_back_list[self.look_back_count] = self.prev_bbox
        self.look_back_count += 1

        # Get the MA of current bbox
        if self.count == self.ma_window:
            self.count = 0
        self.ma_bbox[self.count] = copy(current_bbox)
        self.count += 1
        self.cur_bbox = np.mean(self.ma_bbox, axis=0)

        self.height = self.cur_bbox[3] - self.cur_bbox[1]

        self.prev_centroid = self.find_centroid(self.prev_bbox)
        self.cur_centroid = self.find_centroid(self.cur_bbox)

    def _moved_beyond_threshold(self):
        assert self.cur_bbox is not None, "Please update the current bbox first!"
        prev_bbox = self.look_back_list[self.look_back_count - self.look_back_period]
        bbox_iou = iou(prev_bbox, self.cur_bbox)
        if bbox_iou <= self.iou_threshold:
            return True
        else:
            return False

    @staticmethod
    def find_centroid(bbox: Union[list, np.ndarray]) -> np.ndarray:
        x = (bbox[0] + bbox[2])/2
        y = (bbox[1] + bbox[3])/2
        return np.array([x, y])

    # @staticmethod
    # def centroid_distance(c1, c2):
    #     if not (isinstance(c1,np.ndarray) and isinstance(c2,np.ndarray)):
    #         c1 = np.array(c1)
    #         c2 = np.array(c2)
    #     return np.linalg.norm(c1-c2)


class Person(Tracked_Obj):

    def __init__(
        self, 
        current_bbox: Union[list, np.ndarray], 
        iou_threshold: float, 
        ma_window=2, 
        look_back_period=1
    ):
        super().__init__(current_bbox, iou_threshold, ma_window, look_back_period)

    def is_same_id(self) -> bool:
        return not self._moved_beyond_threshold()       
            # Return False if person bbox moved beyond threshold. i.e. tracking another person. To tackle id switching
        

class Bus(Tracked_Obj):

    def __init__(
        self, 
        current_bbox: Union[list, np.ndarray], 
        iou_threshold: float, 
        door_height_proportion: float,          # Proportion of height of door with respective to the height of bus
        ma_window=10,
        look_back_period=5
    ):
        super().__init__(current_bbox, iou_threshold, ma_window, look_back_period)
        self.door_height_proportion = door_height_proportion
        self.bus_door = None    # (x, (y1, y2))
        self.passengers: Set[Person] = set()
        self.stationary = False
        
    def is_stationary(self) -> bool:
        self.stationary = not self._moved_beyond_threshold()
        return self.stationary       

    def door_line(
            self,
            offset: float,
            rescale_function=None,
            draw_door=False, 
            image=None
        ) -> None:

        offset *= (self.cur_bbox[2] - self.cur_bbox[0]) # Assume door is vertically straight on the right side
        x = self.cur_bbox[2] + offset
        y2 = self.cur_bbox[3]
        y1 = y2 - self.door_height_proportion * self.height
            
        if draw_door:
            assert(image is not None), "Please provide image to draw the door on"
            assert(rescale_function is not None), "Please provide rescale function for drawing the door"
            rescaled_tracks = rescale_function([(x, y1, x, y2)])[0]
            cv2.line(
                image, 
                (rescaled_tracks[0], rescaled_tracks[1]), 
                (rescaled_tracks[2], rescaled_tracks[3]), 
                color=[255,255,255], 
                thickness=2
            )
        self.bus_door = (x, (y1, y2))
