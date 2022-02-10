from typing import Any, Dict, Union

from peekingduck.pipeline.nodes.node import AbstractNode

import os
import ntpath
import numpy as np
from .utils.tracker import Person, Bus
from .utils.draw_image import include_text
from copy import deepcopy, copy
import pandas as pd
from collections import defaultdict
import pdb


class Node(AbstractNode):
    """This node counts the number of passengers boarding each bus

    Args:
        config (:obj:`Dict[str, Any]` | :obj:`None`): Node configuration.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        # super().__init__(config, node_path=__name__, **kwargs)
        node_path = os.path.join(os.getcwd(), "src/custom_nodes/configs/dabble.passenger_counting")
        super().__init__(config, node_path=node_path, **kwargs)
        self.bus_dict = dict()
        self.person_dict = dict()
        self.bus_tracks = None
        self.rescaled_bus_tracks = None
        self.bus_ids = None
        self.person_tracks = None
        self.person_ids = None
        self.rescale_function = None
        self.image_ = None
        self.frame = 0
        self.fps_ = None
        self.buses_records = dict()


    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
        """
        Args:
            inputs (dict): Dictionary with keys "img", "bus_tracks", "person_tracks",
            "bus_ids", "person_ids", "rescale_function" and "saved_video_fps".

        Returns:
            outputs (dict): Dictionary with keys "buses".
        """
        self.frame += 1
        self.bus_tracks = deepcopy(inputs["bus_tracks"])
        self.bus_ids = copy(inputs["bus_ids"])
        self.person_tracks = deepcopy(inputs["person_tracks"])
        self.person_ids = copy(inputs["person_ids"])
        self.rescale_function = inputs['rescale_function']
        self.image_ = inputs['img']
        self.fps_ = inputs["saved_video_fps"]

        self._update_bus()
        self._update_person()
        self._count_passenger()
        if (len(self.bus_dict) != 0) and (self.indicate_num_passengers):
            for bus_id, bus in self.bus_dict.items():
                num_passengers = len(bus.passengers)
                if not isinstance(self.bus_ids, np.ndarray):
                    self.bus_ids = np.array(self.bus_ids)
                bus_bbox = self.rescaled_bus_tracks[np.where(self.bus_ids==bus_id)[0][0]]
                text = f"n_passengers: {num_passengers}"
                include_text(self.image_, bus_bbox, text, [255,255,255], 'top_left_upper')

        return {"buses": self.bus_dict}

    def save_data(self, mode='w'):
        """Saves the number of passengers in each bus, the timing in the video when the bus appears to a csv file

        Args:
            mode (str, optional): Whether to write to a new file or append to an existing file. 
            Defaults to 'w'.
        """
        records_to_save = defaultdict(list)
        indices = []
        if self.record_to_csv:
            self.logger.info("Saving bus record to CSV file!")
            base, _ = ntpath.split(self.csv_path)
            # Create make new directories if the path does not exist
            if not os.path.exists(base):
                os.makedirs(base)
                self.logger.info(f"Created folder '{base}'")
            # Iterate each bus object and save to a DataFrame
            for i, bus in enumerate(self.buses_records.keys()):
                records_to_save["Number of Passengers"].append(len(bus.passengers))
                min, sec = self.buses_records[bus]
                records_to_save["Recorded Time"].append(f"{min}:{sec}")
                indices.append(str(i))
            df = pd.DataFrame(data=records_to_save, index=indices)
            df.to_csv(self.csv_path, mode=mode)

    def _update_bus(self):
        """Updates the bus objects. If bus object already exists, update the existing object.
        Otherwise, create a new object.
        """
        self.rescaled_bus_tracks = self.rescale_function(self.bus_tracks)
        bus_dict = dict()
        for idx, id in enumerate(self.bus_ids):
            if id in self.bus_dict:
                bus_obj = self.bus_dict[id]
                bus_obj.update_pos(self.bus_tracks[idx])
                bus_dict[id] = bus_obj
                # Draw the virtual door line if the bus is stationary
                if bus_obj.is_stationary():
                    text = 'stationary'
                    bus_dict[id].door_line(
                        offset=self.bus_tracker['boundary_offset'],
                        image=self.image_,
                        rescale_function=self.rescale_function,
                        draw_door=self.bus_tracker['draw_boundary'],
                    )
                else:
                    text='moving'
                # Indicates in the image whether a bus is moving
                include_text(
                    self.image_, 
                    self.rescaled_bus_tracks[idx], 
                    text, 
                    colour=[255,255,255], 
                    pos='top_left'
                )
            else:
                # Creates a new object if bus object does not exist
                bus_dict[id] = Bus(
                    current_bbox=copy(self.bus_tracks[idx]), 
                    iou_threshold=self.bus_tracker['iou_threshold'],
                    door_height_proportion=self.bus_tracker['door_height_proportion'],
                    door_offset_height=self.bus_tracker['door_offset_height'],
                    ma_window=self.bus_tracker['ma_window'],
                    look_back_period=self.bus_tracker['look_back_period']
                )
                min, sec = int(self.frame//self.fps_//60), int(self.frame//self.fps_)%60
                # buses_records dict have keys that are bus object and value being their appearance timing
                self.buses_records[bus_dict[id]] = (min, sec)
        # Copy the bus_dict into the class attribute, objects that do not appear in current frame are removed
        self.bus_dict = bus_dict

    def _update_person(self):
        """Updates the person objects. If person object already exists, update the existing object.
        Otherwise, create a new object.
        """
        person_dict = dict()
        for idx, id in enumerate(self.person_ids):
            # Check whether the person object already exists in record
            if id in self.person_dict:
                person_obj = self.person_dict[id]
                person_obj.update_pos(self.person_tracks[idx])
                # Check whether ID switch has occurred to a particular person ID 
                # by looking at the IOU of previous and current bbox
                if person_obj.is_same_id():
                    person_dict[id] = person_obj
                else:
                    person_dict[id] = Person(
                        current_bbox=copy(self.person_tracks[idx]),
                        iou_threshold=self.person_tracker['iou_threshold'],
                        ma_window=self.person_tracker['ma_window'],
                        look_back_period=self.person_tracker['look_back_period']
                    )
            else:
                person_dict[id] = Person(
                        current_bbox=copy(self.person_tracks[idx]),
                        iou_threshold=self.person_tracker['iou_threshold'],
                        ma_window=self.person_tracker['ma_window'],
                        look_back_period=self.person_tracker['look_back_period']
                    )

        self.person_dict = person_dict
    
    def _count_passenger(self):
        """Count the number of passengers based on the heuristics
        """
        img_rows, img_cols, _ = self.image_.shape
        for _, bus_obj in self.bus_dict.items():
            if bus_obj.stationary:
                for _, person_obj in self.person_dict.items():
                    person_bus_height_r = person_obj.height / (bus_obj.width*img_cols/img_rows)
                    if (person_obj.prev_centroid[0] < bus_obj.bus_door[0]) or \
                        (person_obj.prev_centroid[1] < bus_obj.bus_door[1][0]) or \
                        (person_obj.prev_centroid[1] > bus_obj.bus_door[1][1]) or \
                        (person_bus_height_r < self.person_to_bus_ratio[0]) or \
                        (person_bus_height_r > self.person_to_bus_ratio[1]) or \
                        (person_obj in bus_obj.passengers):
                        # If person was on left side or above or below the bus door,
                        # or if the person is too small compared to bus bbox, 
                        # or if the person has already been counted
                        continue
                    elif (person_obj.cur_centroid is not None) and \
                        (person_obj.cur_centroid[0] < bus_obj.bus_door[0]) and \
                        (person_obj.cur_centroid[1] > bus_obj.bus_door[1][0]) and \
                        (person_obj.cur_centroid[1] < bus_obj.bus_door[1][1]):
                        # If the person was not just initialized and previously con correct position, 
                        # and is currently on the left side and within the height of door
                        bus_obj.passengers.add(person_obj)


    def __del__(self):
        # Saves the data before the passenger counting instance is destroyed
        self.save_data(mode=self.write_mode)
