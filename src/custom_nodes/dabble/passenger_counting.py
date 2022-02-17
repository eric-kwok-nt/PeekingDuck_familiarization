from typing import Any, Dict
from peekingduck.pipeline.nodes.node import AbstractNode
import os
import numpy as np
from .utils.tracker import Person, Bus
from .utils.draw_image import include_text, bboxes_rescaling
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
        node_path = os.path.join(
            os.getcwd(), "src/custom_nodes/configs/dabble.passenger_counting"
        )
        super().__init__(config, node_path=node_path, **kwargs)
        self.bus_dict = dict()
        self.person_dict = dict()
        self.bus_tracks = None
        self.rescaled_bus_tracks = None
        self.bus_ids = None
        self.person_tracks = None
        self.person_ids = None
        self.image_ = None
        self.frame = 0
        self.fps_ = None
        self.buses_records = dict()
        self.draw_pipeline = []
        self.write_now = False
        self.bus_record_df = None

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
        """
        Args:
            inputs (dict):
                - "img"
                - "bus_tracks"
                - "person_tracks"
                - "bus_ids"
                - "person_ids"
                - "saved_video_fps"
                - "draw_pipeline"
                - "pipeline_end"

        Returns:
            outputs (dict):
                - "df_records"
                - "draw_pipeline"
                - "write_now"
        """
        self.frame += 1
        self.draw_pipeline = inputs["draw_pipeline"]
        self.bus_tracks = deepcopy(inputs["bus_tracks"])
        self.bus_ids = copy(inputs["bus_ids"])
        self.person_tracks = deepcopy(inputs["person_tracks"])
        self.person_ids = copy(inputs["person_ids"])
        self.image_ = inputs["img"]
        self.fps_ = inputs["saved_video_fps"]
        if self.image_ is not None:
            self._update_bus()
            self._update_person()
            self._count_passenger()
            # To indicate number of passengers on the respective bus
            if (len(self.bus_dict) != 0) and (self.indicate_num_passengers):
                for bus_id, bus in self.bus_dict.items():
                    num_passengers = len(bus.passengers)
                    if not isinstance(self.bus_ids, np.ndarray):
                        self.bus_ids = np.array(self.bus_ids)
                    bus_bbox = self.rescaled_bus_tracks[
                        np.where(self.bus_ids == bus_id)[0][0]
                    ]
                    text = f"n_passengers: {num_passengers}"
                    include_text_kwargs = {
                        "image": self.image_,
                        "bbox": bus_bbox,
                        "tag": text,
                        "colour": [255, 255, 255],
                        "pos": "top_left_upper",
                    }
                    self.draw_pipeline.append((include_text, include_text_kwargs))
        if inputs["pipeline_end"]:
            self.bus_record_df = self.save_data()

        return {
            "df_records": self.bus_record_df,
            "draw_pipeline": self.draw_pipeline,
            "write_now": self.write_now,
        }

    def _update_bus(self):
        """Updates the bus objects. If bus object already exists, update the existing object.
        Otherwise, create a new object.
        """
        self.rescaled_bus_tracks = bboxes_rescaling(self.bus_tracks, self.image_)
        bus_dict = dict()
        for idx, id in enumerate(self.bus_ids):
            if id in self.bus_dict:
                bus_obj = self.bus_dict[id]
                bus_obj.update_pos(self.bus_tracks[idx])
                bus_dict[id] = bus_obj
                # Draw the virtual door line if the bus is stationary
                if bus_obj.is_stationary():
                    text = "stationary"
                    bus_dict[id].door_line(
                        offset=self.bus_tracker["boundary_offset"],
                        image=self.image_,
                        draw_door=self.bus_tracker["draw_boundary"],
                        draw_pipeline=self.draw_pipeline,
                    )
                else:
                    text = "moving"
                # Indicates in the image whether a bus is moving
                include_text_kwargs = {
                    "image": self.image_,
                    "bbox": self.rescaled_bus_tracks[idx],
                    "tag": text,
                    "colour": [255, 255, 255],
                    "pos": "top_left",
                }
                self.draw_pipeline.append((include_text, include_text_kwargs))
            else:
                # Creates a new object if bus object does not exist
                bus_dict[id] = Bus(
                    current_bbox=copy(self.bus_tracks[idx]),
                    iou_threshold=self.bus_tracker["iou_threshold"],
                    door_height_proportion=self.bus_tracker["door_height_proportion"],
                    door_offset_height=self.bus_tracker["door_offset_height"],
                    ma_window=self.bus_tracker["ma_window"],
                    look_back_period=self.bus_tracker["look_back_period"],
                )
                min, sec = (
                    int(self.frame // self.fps_ // 60),
                    int(self.frame // self.fps_) % 60,
                )
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
                        iou_threshold=self.person_tracker["iou_threshold"],
                        ma_window=self.person_tracker["ma_window"],
                        look_back_period=self.person_tracker["look_back_period"],
                    )
            else:
                person_dict[id] = Person(
                    current_bbox=copy(self.person_tracks[idx]),
                    iou_threshold=self.person_tracker["iou_threshold"],
                    ma_window=self.person_tracker["ma_window"],
                    look_back_period=self.person_tracker["look_back_period"],
                )

        self.person_dict = person_dict

    def _count_passenger(self):
        """Count the number of passengers based on the heuristics"""
        img_rows, img_cols, _ = self.image_.shape
        for _, bus_obj in self.bus_dict.items():
            if bus_obj.stationary:
                for _, person_obj in self.person_dict.items():
                    person_bus_height_r = person_obj.height / (
                        bus_obj.width * img_cols / img_rows
                    )
                    if (
                        (person_obj.prev_centroid[0] < bus_obj.bus_door[0])
                        or (person_obj.prev_centroid[1] < bus_obj.bus_door[1][0])
                        or (person_obj.prev_centroid[1] > bus_obj.bus_door[1][1])
                        or (person_bus_height_r < self.person_to_bus_ratio[0])
                        or (person_bus_height_r > self.person_to_bus_ratio[1])
                        or (person_obj in bus_obj.passengers)
                    ):
                        # If person was on left side or above or below the bus door,
                        # or if the person is too small compared to bus bbox,
                        # or if the person has already been counted
                        continue
                    elif (
                        (person_obj.cur_centroid is not None)
                        and (person_obj.cur_centroid[0] < bus_obj.bus_door[0])
                        and (person_obj.cur_centroid[1] > bus_obj.bus_door[1][0])
                        and (person_obj.cur_centroid[1] < bus_obj.bus_door[1][1])
                    ):
                        # If the person was not just initialized and previously con correct position,
                        # and is currently on the left side and within the height of door
                        bus_obj.passengers.add(person_obj)

    def save_data(self):
        """Saves the number of passengers in each bus, the timing in the video when the bus appears to a csv file"""
        self.logger.info("Saving Number of passengers to DataFrame...")
        records_to_save = defaultdict(list)
        indices = []
        # Iterate each bus object and save to a DataFrame
        for i, bus in enumerate(self.buses_records.keys()):
            records_to_save["Number of Passengers"].append(len(bus.passengers))
            min, sec = self.buses_records[bus]
            records_to_save["Recorded Time"].append(f"{min}:{sec}")
            indices.append(str(i))
        df = pd.DataFrame(data=records_to_save, index=indices)
        return df
