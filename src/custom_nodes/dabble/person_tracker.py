"""
Node template for creating custom nodes.
"""

from typing import Any, Dict

from peekingduck.pipeline.nodes.node import AbstractNode
from .sort_tracker.sort import Sort
import numpy as np
import cv2
import threading
from queue import Queue
import pdb


class Node(AbstractNode):
    """This is a template class of how to write a node for PeekingDuck.

    Args:
        config (:obj:`Dict[str, Any]` | :obj:`None`): Node configuration.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self.mot_person_tracker = Sort(max_age=self.sort_person_tracker['DEFAULT_MAX_AGE'],
                       min_hits=self.sort_person_tracker["DEFAULT_MIN_HITS"],
                       use_time_since_update=self.sort_person_tracker['DEFAULT_USE_TIME_SINCE_UPDATE'],
                       iou_threshold=self.sort_person_tracker['DEFAULT_IOU_THRESHOLD'],
                       tracker_type=self.sort_person_tracker['TRACKER_TYPE'])
        
        self.mot_bus_tracker = Sort(max_age=self.sort_bus_tracker['DEFAULT_MAX_AGE'],
                min_hits=self.sort_bus_tracker["DEFAULT_MIN_HITS"],
                use_time_since_update=self.sort_bus_tracker['DEFAULT_USE_TIME_SINCE_UPDATE'],
                iou_threshold=self.sort_bus_tracker['DEFAULT_IOU_THRESHOLD'],
                tracker_type=self.sort_bus_tracker['TRACKER_TYPE'])
        self.image_ = None
        self.img_n_rows = None 
        self.img_n_cols = None
        self.frame = 0

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
        """This node does ___.

        Args:
            inputs (dict): Dictionary with keys "__", "__".

        Returns:
            outputs (dict): Dictionary with keys "__".
        """
        bboxes = []
        obj_tags = []
        bbox_labels = []
        try:
            if inputs['bboxes'].shape[1] == 4:
                que_person = Queue()
                que_bus = Queue()
                self.image_ = inputs['img']
                self.img_n_rows, self.img_n_cols, _ = self.image_.shape
                # pdb.set_trace()
                person_bboxes = inputs["bboxes"][inputs["bbox_labels"]=='person']
                bus_bboxes = inputs["bboxes"][inputs["bbox_labels"]=='bus']

                if self.multithread:
                    t_person = threading.Thread(
                        target=lambda q , args: q.put(self._track(*args)), 
                        args=(que_person, [self.mot_person_tracker, person_bboxes]), 
                        daemon=True
                        )
                    t_bus = threading.Thread(
                        target=lambda q , args: q.put(self._track(*args)), 
                        args=(que_bus, [self.mot_bus_tracker, bus_bboxes]), 
                        daemon=True
                        )
                    t_person.start()
                    t_bus.start()
                    t_person.join()
                    t_bus.join()

                    person_tracks, person_tracks_ids = que_person.get()
                    bus_tracks, bus_tracks_ids = que_bus.get()
                else:
                    person_tracks, person_tracks_ids = self._track(self.mot_person_tracker, person_bboxes)
                    bus_tracks, bus_tracks_ids = self._track(self.mot_bus_tracker, bus_bboxes)

                bbox_labels = np.array(["person" for _ in person_tracks]+["bus" for _ in bus_tracks])
                
                if self.show_class_in_tag:
                    obj_tags = [f"person_{id}" for id in person_tracks_ids] + [f"bus_{id}" for id in bus_tracks_ids]
                else:
                    obj_tags = [f"{id}" for id in person_tracks_ids] + [f"{id}" for id in bus_tracks_ids]

                bboxes = np.concatenate((person_tracks, bus_tracks))
        except IndexError:
            pass
        
        outputs = {
            "bboxes": bboxes,
            "obj_tags": obj_tags,
            "bbox_labels": bbox_labels
            }
        self.frame += 1
        # if self.frame == 11:
        #     pdb.set_trace()
        return outputs

    def _track(self, mot_tracker, bboxes):
        bboxes_rescaled = self._bboxes_rescaling(bboxes)
        
        tracks, tracks_ids = mot_tracker.update_and_get_tracks(bboxes_rescaled, self.image_)
        tracks, tracks_ids = np.array(tracks), np.array(tracks_ids)
        tracks[:,[0,2]] /= self.img_n_cols
        tracks[:,[1,3]] /= self.img_n_rows
        return tracks, tracks_ids


    def _draw_rectangle(self, bboxes, color=[255,255,255], thickness=4):
        bboxes_rescaled = self._bboxes_rescaling(bboxes)
        for box in bboxes_rescaled:
            self.image_ = cv2.rectangle(
                self.image_, 
                pt1=(int(box[0]), int(box[1])), 
                pt2=(int(box[2]), int(box[3])), 
                color=color, 
                thickness=thickness
                )

    def _bboxes_rescaling(self, bboxes):
        bboxes_rescaled = []
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox
            x_min = int(x_min*self.img_n_cols)
            x_max = int(x_max*self.img_n_cols)
            y_min = int(y_min*self.img_n_rows)
            y_max = int(y_max*self.img_n_rows)
            bboxes_rescaled.append([x_min, y_min, x_max, y_max])
        
        return bboxes_rescaled

    # def get_sorted_idx(self, array: np.ndarray):
    #     processed_list = []
    #     for i, item in enumerate(array):
    #         processed_list.append((i, tuple(item)))
    #     sorted_list = sorted(processed_list, key=lambda tup: tup[1])
    #     return np.array([item[0] for item in sorted_list])
