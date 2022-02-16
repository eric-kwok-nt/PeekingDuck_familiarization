from re import L
from typing import Any, Dict, Union
import os
from copy import deepcopy
import cv2
import numpy as np
from peekingduck.pipeline.nodes.node import AbstractNode
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from .utils.draw_image import bboxes_rescaling
from .utils.tracker import Tracked_Obj
from .utils.publishers import StringPublisher
import rclpy
from rclpy.node import Node


class Node(AbstractNode):
    """This is a template class of how to write a node for PeekingDuck.

    Args:
        config (:obj:`Dict[str, Any]` | :obj:`None`): Node configuration.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        node_path = os.path.join(
            os.getcwd(), "src/custom_nodes/configs/dabble.bus_stop_count_pub"
        )
        super().__init__(config, node_path=node_path, **kwargs)
        # super().__init__(config, node_path=__name__, **kwargs)
        self.image_ = None
        self.bboxes = None
        self.ros_node_name = "bus_stop_counter"
        self.ros_topic_name = "number_of_people"
        self.Publisher = StringPublisher(
            node_name_=self.ros_node_name, topic_name=self.ros_topic_name
        )

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
        """This node does ___.

        Args:
            inputs (dict): Dictionary with keys "__", "__".

        Returns:
            outputs (dict): Dictionary with keys "__".
        """
        # self.image_ = deepcopy(inputs["img"])
        self.image_ = deepcopy(inputs["img"])
        self.bboxes = inputs["person_tracks"]
        count = self._count_people()
        if self.show_bus_stop_zone:
            self._draw_rectangle([np.array(self.bus_stop_zone)[[0, 2], :].ravel()])
        text = f"{count}"
        self.Publisher.publish(text)
        if inputs["pipeline_end"]:
            self.Publisher.shutdown()
        outputs = {}
        return outputs

    def _count_people(self):
        count = 0
        scaled_bboxes = bboxes_rescaling(self.bboxes, self.image_)
        polygon = Polygon(self.bus_stop_zone)
        for bbox in scaled_bboxes:
            centroid = Tracked_Obj.find_centroid(bbox)
            point = Point(centroid)
            if polygon.contains(point):
                count += 1
        return count

    def _draw_rectangle(
        self,
        bboxes: Union[list, np.ndarray],
        color=[255, 255, 255],
        thickness=2,
    ):
        """Draws the bboxes on image

        Args:
            bboxes (Union[list, np.ndarray]): List or array of bounding boxes
            color (list, optional): Colour of the bbox. Defaults to [255,255,255].
            thickness (int, optional): Thickness of bbox. Defaults to 2.
        """
        for box in bboxes:
            draw_rect_kwargs = {
                "img": self.image_,
                "pt1": (int(box[0]), int(box[1])),
                "pt2": (int(box[2]), int(box[3])),
                "color": color,
                "thickness": thickness,
            }
            cv2.rectangle(**draw_rect_kwargs)
