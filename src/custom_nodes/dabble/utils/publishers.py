import rclpy
from rclpy.node import Node


class ROSPublisher(Node):
    rclpy.init()

    def __init__(self, node_name_: str):
        super().__init__(node_name_)
        self.publisher_list = []

    def create_publishers(self, msg_type, topic_name, queue_size):
        self.publisher_list.append(
            self.create_publisher(msg_type, topic_name, queue_size)
        )

    def publish(self, msg_list: list):
        for publisher, msg in zip(self.publisher_list, msg_list):
            publisher.publish(msg)

    def shutdown(self):
        self.destroy_node()
        rclpy.shutdown()
