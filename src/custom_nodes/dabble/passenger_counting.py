"""
Node template for creating custom nodes.
"""

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
    """This is a template class of how to write a node for PeekingDuck.

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
        """This node does ___.

        Args:
            inputs (dict): Dictionary with keys "__", "__".

        Returns:
            outputs (dict): Dictionary with keys "__".
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
        records_to_save = defaultdict(list)
        indices = []
        if self.record_to_csv:
            self.logger.info("Saving bus record to CSV file!")
            base, _ = ntpath.split(self.csv_path)
            if not os.path.exists(base):
                os.makedirs(base)
                self.logger.info(f"Created folder '{base}'")
            for i, bus in enumerate(self.buses_records.keys()):
                records_to_save["Number of Passengers"].append(len(bus.passengers))
                min, sec = self.buses_records[bus]
                records_to_save["Recorded Time"].append(f"{min}:{sec}")
                indices.append(str(i))
            series = pd.DataFrame(data=records_to_save, index=indices)
            series.to_csv(self.csv_path, mode=mode)

    def _update_bus(self):
        self.rescaled_bus_tracks = self.rescale_function(self.bus_tracks)
        bus_dict = dict()
        for idx, id in enumerate(self.bus_ids):
            if id in self.bus_dict:
                bus_obj = self.bus_dict[id]
                bus_obj.update_pos(self.bus_tracks[idx])
                bus_dict[id] = bus_obj
                
                if bus_obj.is_stationary():
                    text = 'stationary'
                    bus_dict[id].door_line(
                        offset=self.bus_tracker['boundary_offset'],
                        rescale_function=self.rescale_function,
                        draw_door=self.bus_tracker['draw_boundary'],
                        image=self.image_
                    )
                else:
                    text='moving'

                include_text(
                    self.image_, 
                    self.rescaled_bus_tracks[idx], 
                    text, 
                    colour=[255,255,255], 
                    pos='top_left'
                )
            else:
                bus_dict[id] = Bus(
                    current_bbox=copy(self.bus_tracks[idx]), 
                    iou_threshold=self.bus_tracker['iou_threshold'],
                    door_height_proportion=self.bus_tracker['door_height_proportion'],
                    ma_window=self.bus_tracker['ma_window'],
                    look_back_period=self.bus_tracker['look_back_period']
                )
                min, sec = int(self.frame//self.fps_//60), int(self.frame//self.fps_)%60
                self.buses_records[bus_dict[id]] = (min, sec)
            
        self.bus_dict = bus_dict

    def _update_person(self):
        person_dict = dict()
        for idx, id in enumerate(self.person_ids):
            if id in self.person_dict:
                person_obj = self.person_dict[id]
                person_obj.update_pos(self.person_tracks[idx])

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
        for _, bus_obj in self.bus_dict.items():
            if bus_obj.stationary:
                for _, person_obj in self.person_dict.items():
                    person_bus_height_r = person_obj.height / bus_obj.height
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
        self.save_data(mode=self.write_mode)
