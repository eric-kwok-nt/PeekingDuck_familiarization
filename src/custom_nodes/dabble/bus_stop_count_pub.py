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
from .utils.publishers import ROSPublisher
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class Node(AbstractNode):
    """This Node counts the number of people at the pre-defined bus stop bounding box
    and publishes ROS String and Image messages onto the ROS2 topic

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
        self.br = CvBridge()
        self.ros_node_name = "bus_stop_counter"
        self.Publisher = ROSPublisher(self.ros_node_name)
        self.Publisher.create_publishers(String, "/number_of_people", queue_size=1)
        if self.publish_footage:
            self.Publisher.create_publishers(Image, "/video_footage", queue_size=10)
        self.msg_list = []

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
        """
        Args:
            inputs (dict): Dictionary with keys "img", "person_tracks", "pipeline_end".

        Returns:
            outputs (dict): {}.
        """
        self.msg_list = []
        self.image_ = deepcopy(inputs["img"])
        self.bboxes = inputs["person_tracks"]
        count = self._count_people()
        if self.show_bus_stop_zone:
            self._draw_rectangle([np.array(self.bus_stop_zone)[[0, 2], :].ravel()])
        # Create messages
        text = String()
        text.data = f"{count}"
        self.msg_list.append(text)
        if self.publish_footage:
            self.msg_list.append(self.br.cv2_to_imgmsg(self.image_))
        # Publish messages
        self.Publisher.publish(self.msg_list)
        # Make sure to stop the ROS node when the pipeline ends
        if inputs["pipeline_end"]:
            self.Publisher.shutdown()

        return {}

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
