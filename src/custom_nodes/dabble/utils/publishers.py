import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from abc import ABCMeta, abstractmethod


class PublisherTemplate(Node, metaclass=ABCMeta):
    rclpy.init()

    def __init__(self, node_name_: str, topic_name: str, queue_size: int):
        super().__init__(node_name_)
        self.publisher_ = self.create_publisher(String, topic_name, queue_size)

    @abstractmethod
    def publish(self):
        raise NotImplementedError("This method needs to be implemented")

    def shutdown(self):
        self.destroy_node()
        rclpy.shutdown()


class StringPublisher(PublisherTemplate):
    def __init__(self, node_name_: str, topic_name: str, queue_size=1):
        super().__init__(node_name_, topic_name, queue_size)

    def publish(self, text: str):
        msg = String()
        msg.data = text
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)


