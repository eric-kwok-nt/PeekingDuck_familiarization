import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge


class Subscriber(Node):
    def __init__(self):
        super().__init__("person_count_subscriber")
        self.br = CvBridge()
        self.count_sub = self.create_subscription(
            String, "/number_of_people", self.listener_callback, 1
        )
        self.image_sub = self.create_subscription(
            Image, "/video_footage", self.image_callback, 10
        )

    def listener_callback(self, msg):
        self.get_logger().info("%s at the bus stop." % msg.data)

    def image_callback(self, img):
        img_cv = self.br.imgmsg_to_cv2(img)
        cv2.imshow("Video_footage", img_cv)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            return


def main(args=None):
    rclpy.init(args=args)

    person_count_subscriber = Subscriber()

    rclpy.spin(person_count_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    person_count_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
