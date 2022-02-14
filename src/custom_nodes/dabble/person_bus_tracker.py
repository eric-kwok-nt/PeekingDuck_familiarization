from typing import Any, Dict, List, Union, Callable, Tuple
from peekingduck.pipeline.nodes.node import AbstractNode
from .sort_tracker.sort import Sort
from .utils.draw_image import include_text
import numpy as np
import cv2
import threading
from queue import Queue
from copy import deepcopy, copy
import os
from .deep_sort import nn_matching
from .deep_sort.detection import Detection
from .deep_sort.tracker import Tracker
from .deep_sort.tools import generate_detections as gdet
import pdb


class Node(AbstractNode):
    """This node tracks person and bus objects

    Args:
        config (:obj:`Dict[str, Any]` | :obj:`None`): Node configuration.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        node_path = os.path.join(
            os.getcwd(), "src/custom_nodes/configs/dabble.person_bus_tracker"
        )
        super().__init__(config, node_path=node_path, **kwargs)
        # super().__init__(config, node_path=__name__, **kwargs)
        if not self.deep_sort:
            self.mot_person_tracker = Sort(
                max_age=self.sort_person_tracker["DEFAULT_MAX_AGE"],
                min_hits=self.sort_person_tracker["DEFAULT_MIN_HITS"],
                use_time_since_update=self.sort_person_tracker[
                    "DEFAULT_USE_TIME_SINCE_UPDATE"
                ],
                iou_threshold=self.sort_person_tracker["DEFAULT_IOU_THRESHOLD"],
                tracker_type=self.sort_person_tracker["TRACKER_TYPE"],
            )

            self.mot_bus_tracker = Sort(
                max_age=self.sort_bus_tracker["DEFAULT_MAX_AGE"],
                min_hits=self.sort_bus_tracker["DEFAULT_MIN_HITS"],
                use_time_since_update=self.sort_bus_tracker[
                    "DEFAULT_USE_TIME_SINCE_UPDATE"
                ],
                iou_threshold=self.sort_bus_tracker["DEFAULT_IOU_THRESHOLD"],
                tracker_type=self.sort_bus_tracker["TRACKER_TYPE"],
            )
        else:
            person_metric = nn_matching.NearestNeighborDistanceMetric(
                "cosine",
                self.deep_sort_person_tracker["max_cosine_distance"],
                self.deep_sort_person_tracker["nn_budget"],
            )
            self.mot_person_tracker = Tracker(person_metric)

            bus_metric = nn_matching.NearestNeighborDistanceMetric(
                "cosine",
                self.deep_sort_bus_tracker["max_cosine_distance"],
                self.deep_sort_bus_tracker["nn_budget"],
            )
            self.mot_bus_tracker = Tracker(bus_metric)

            model_filename = (
                "src/custom_nodes/dabble/deep_sort/model_data/mars-small128.pb"
            )
            self.encoder = gdet.create_box_encoder(model_filename, batch_size=1)

        self.image_ = None
        self.img_n_rows = None
        self.img_n_cols = None
        self.frame = 0
        self.draw_pipeline = None

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
        """
        Args:
            inputs (dict): 
                - "img"
                - "bboxes"
                - "bbox_labels"
                - "bbox_scores"

        Returns:
            outputs (dict): 
                - "obj_tags"
                - "bboxes"
                - "bbox_labels"
                - "bus_tracks"
                - "person_tracks"
                - "bus_ids"
                - "person_ids"
                - "rescale_function"
                - "draw_pipeline"
        """
        bboxes = []
        obj_tags = []
        bbox_labels = []
        self.draw_pipeline = []
        que_person = Queue()
        que_bus = Queue()
        self.image_ = inputs["img"]
        self.img_n_rows, self.img_n_cols, _ = self.image_.shape

        # Separates person bboxes with bus bboxes and tracks them separately
        person_bboxes = inputs["bboxes"][inputs["bbox_labels"] == "person"]
        person_scores = inputs["bbox_scores"][inputs["bbox_labels"] == "person"]
        bus_bboxes = inputs["bboxes"][inputs["bbox_labels"] == "bus"]
        bus_scores = inputs["bbox_scores"][inputs["bbox_labels"] == "bus"]

        # Filtering the detection bboxes separately
        person_bboxes = person_bboxes[person_scores >= self.person_score_threshold]
        bus_bboxes = bus_bboxes[bus_scores >= self.bus_score_threshold]
        person_scores = person_scores[person_scores >= self.person_score_threshold]
        bus_scores = bus_scores[bus_scores >= self.bus_score_threshold]

        # In puts to tracker is different for SORT and Deep SORT algorithm
        if not self.deep_sort:
            kwargs_person = {
                "bboxes": person_bboxes,
                "mot_tracker": self.mot_person_tracker,
            }
            kwargs_bus = {"bboxes": bus_bboxes, "mot_tracker": self.mot_bus_tracker}
        else:
            kwargs_person = {
                "bboxes": person_bboxes,
                "mot_tracker": self.mot_person_tracker,
                "names": ["person" for _ in range(len(person_bboxes))],
                "scores": person_scores,
                "default_max_age": self.deep_sort_person_tracker["default_max_age"],
            }
            kwargs_bus = {
                "bboxes": bus_bboxes,
                "mot_tracker": self.mot_bus_tracker,
                "names": ["bus" for _ in range(len(bus_bboxes))],
                "scores": bus_scores,
                "default_max_age": self.deep_sort_bus_tracker["default_max_age"],
            }

        # Performs tracking and obtain various track bboxes and IDs
        if self.multithread:
            t_person = threading.Thread(
                target=lambda q, kwargs: q.put(self._track(**kwargs)),
                args=(que_person, kwargs_person),
                daemon=True,
            )
            t_bus = threading.Thread(
                target=lambda q, kwargs: q.put(self._track(**kwargs)),
                args=(que_bus, kwargs_bus),
                daemon=True,
            )
            t_person.start()
            t_bus.start()
            t_person.join()
            t_bus.join()

            person_tracks, person_tracks_ids = que_person.get()
            bus_tracks, bus_tracks_ids = que_bus.get()
        else:
            person_tracks, person_tracks_ids = self._track(**kwargs_person)
            bus_tracks, bus_tracks_ids = self._track(**kwargs_bus)

        # Whether to show the class name in the tag
        if self.show_class_in_tag:
            obj_tags = [f"person_{id}" for id in person_tracks_ids] + [
                f"bus_{id}" for id in bus_tracks_ids
            ]
        else:
            obj_tags = [f"{id}" for id in person_tracks_ids] + [
                f"{id}" for id in bus_tracks_ids
            ]

        # Combines tracks and bbox labels for the draw bbox node downstream
        if (len(bus_tracks) > 0) and (len(person_tracks) > 0):
            bbox_labels = np.array(
                ["person" for _ in person_tracks] + ["bus" for _ in bus_tracks]
            )
            bboxes = np.concatenate((person_tracks, bus_tracks))
        elif len(person_tracks) > 0:
            bbox_labels = np.array(["person" for _ in person_tracks])
            bboxes = person_tracks
        else:
            bbox_labels = np.array(["bus" for _ in bus_tracks])
            bboxes = bus_tracks

        # Whether to draw detection bboxes
        if self.detection["draw_bbox"]:
            draw_person_kwargs = {"bboxes": person_bboxes, "scores": person_scores}
            draw_bus_kwargs = {"bboxes": bus_bboxes, "scores": bus_scores}
            self._draw_rectangle(**draw_person_kwargs)
            self._draw_rectangle(**draw_bus_kwargs)
            # self.draw_pipeline.append((self._draw_rectangle, draw_person_kwargs))
            # self.draw_pipeline.append((self._draw_rectangle, draw_bus_kwargs))

        outputs = {
            "bboxes": deepcopy(bboxes),
            "obj_tags": obj_tags,
            "bbox_labels": bbox_labels,
            "bus_tracks": deepcopy(bus_tracks),
            "person_tracks": deepcopy(person_tracks),
            "bus_ids": copy(bus_tracks_ids),
            "person_ids": copy(person_tracks_ids),
            "rescale_function": self.bboxes_rescaling,
            "draw_pipeline": self.draw_pipeline,
        }
        self.frame += 1

        return outputs

    def _track(
        self,
        bboxes: Union[list, np.ndarray],
        mot_tracker: Callable,
        names=[],
        scores=[],
        default_max_age=1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Performs tracking with either SORT or Deep SORT algorithm

        Args:
            bboxes (Union[list, np.ndarray]): List of Bounding Boxes
            mot_tracker (Callable): The respective MOT tracker 
            names (list, optional): Respective class names. Only applicable to DeepSORT. Defaults to [].
            scores (list, optional): Respective bbox scores. Only applicable to DeepSORT. Defaults to [].
            default_max_age (int, optional): Max age of the tracker bbox. Only applicable to DeepSORT. Defaults to 1.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The tracked bboxes and the respective track IDs.
        """
        tracks = []
        tracks_ids = []

        bboxes_rescaled = np.array(self.bboxes_rescaling(bboxes))

        if not self.deep_sort:
            tracks, tracks_ids = mot_tracker.update_and_get_tracks(
                bboxes_rescaled, self.image_
            )
            tracks, tracks_ids = np.array(tracks), np.array(tracks_ids)
        else:
            if len(bboxes_rescaled) != 0:
                bboxes_rescaled[:, [2, 3]] = (
                    bboxes_rescaled[:, [2, 3]] - bboxes_rescaled[:, [0, 1]]
                )

            features = self.encoder(self.image_, bboxes_rescaled)
            detections = [
                Detection(bbox, score, class_name, feature)
                for bbox, score, class_name, feature in zip(
                    bboxes_rescaled, scores, names, features
                )
            ]
            mot_tracker.predict()
            mot_tracker.update(detections)
            for track in mot_tracker.tracks:
                if (
                    not track.is_confirmed()
                    or track.time_since_update > default_max_age
                ):
                    continue
                tracks.append(track.to_tlbr())
                tracks_ids.append(track.track_id)

            tracks, tracks_ids = np.array(tracks), np.array(tracks_ids)
        if len(tracks) > 0:
            tracks[:, [0, 2]] /= self.img_n_cols
            tracks[:, [1, 3]] /= self.img_n_rows
        return tracks, tracks_ids

    def _draw_rectangle(
        self,
        bboxes: Union[list, np.ndarray],
        color=[255, 255, 255],
        thickness=2,
        scores=None,
    ):
        """Draws the bboxes on image

        Args:
            bboxes (Union[list, np.ndarray]): List or array of bounding boxes
            color (list, optional): Colour of the bbox. Defaults to [255,255,255].
            thickness (int, optional): Thickness of bbox. Defaults to 2.
        """
        bboxes_rescaled = self.bboxes_rescaling(bboxes)
        for box, score in zip(bboxes_rescaled, scores):
            draw_rect_kwargs = {
                "img": self.image_,
                "pt1": (int(box[2]), int(box[3])),
                "pt2": (int(box[2]), int(box[3])),
                "color": color,
                "thickness": thickness,
            }
            self.draw_pipeline.append((cv2.rectangle, draw_rect_kwargs))
            # cv2.rectangle(**draw_rect_kwargs)
            if self.detection["include_tag"]:
                text = "Det"
                if self.detection["include_score"]:
                    assert (
                        scores is not None
                    ), "Please input detection scores if include_score is True"
                    text += f": {score:.2f}"

                det_text_kwargs = {
                    "image": self.image_,
                    "bbox": box,
                    "tag": text,
                    "colour": color,
                    "pos": "bottom",
                }
                self.draw_pipeline.append((include_text, det_text_kwargs))
                # include_text(**det_text_kwargs)

    def bboxes_rescaling(
        self, bboxes: List[Union[list, tuple]]
    ) -> List[Union[list, tuple]]:
        """Rescale the normalized bboxes to the original scale.

        Args:
            bboxes (List[Union[list, tuple]]): Bounding boxes to resize

        Returns:
            List[Union[list, tuple]]: Rescaled bboxes
        """
        bboxes_rescaled = []
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox
            x_min = int(x_min * self.img_n_cols)
            x_max = int(x_max * self.img_n_cols)
            y_min = int(y_min * self.img_n_rows)
            y_max = int(y_max * self.img_n_rows)
            bboxes_rescaled.append([x_min, y_min, x_max, y_max])

        return bboxes_rescaled

    # def get_sorted_idx(self, array: np.ndarray):
    #     processed_list = []
    #     for i, item in enumerate(array):
    #         processed_list.append((i, tuple(item)))
    #     sorted_list = sorted(processed_list, key=lambda tup: tup[1])
    #     return np.array([item[0] for item in sorted_list])
