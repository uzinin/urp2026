import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge

from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy

import sys
import cv2
import os

#---------------Variable Setting---------------
# 후면 카메라 Publish 토픽 이름
PUB_TOPIC_NAME = 'rear/image_raw'

# 데이터 입력 소스: 'camera', 'image', 또는 'video'
DATA_SOURCE = 'camera'

# 후면 카메라 장치 번호
# 터미널에서 ls /dev/video* 로 확인 후 수정
CAM_NUM = 4

# 이미지 데이터 디렉토리 경로
IMAGE_DIRECTORY_PATH = 'src/camera_perception_pkg/camera_perception_pkg/lib/Collected_Datasets/sample_dataset'

# 비디오 파일 경로
VIDEO_FILE_PATH = 'src/camera_perception_pkg/camera_perception_pkg/lib/Collected_Datasets/driving_simulation.mp4'

# 화면 출력 여부
SHOW_IMAGE = True

# 이미지 발행 주기
TIMER = 0.03

# 후면 카메라 frame_id
FRAME_ID = 'rear_camera_frame'
#----------------------------------------------


class RearImagePublisherNode(Node):
    def __init__(
        self,
        data_source=DATA_SOURCE,
        cam_num=CAM_NUM,
        img_dir=IMAGE_DIRECTORY_PATH,
        video_path=VIDEO_FILE_PATH,
        pub_topic=PUB_TOPIC_NAME,
        logger=SHOW_IMAGE,
        timer=TIMER,
        frame_id=FRAME_ID
    ):
        super().__init__('rear_image_publisher_node')

        self.declare_parameter('data_source', data_source)
        self.declare_parameter('cam_num', cam_num)
        self.declare_parameter('img_dir', img_dir)
        self.declare_parameter('video_path', video_path)
        self.declare_parameter('pub_topic', pub_topic)
        self.declare_parameter('logger', logger)
        self.declare_parameter('timer', timer)
        self.declare_parameter('frame_id', frame_id)

        self.data_source = self.get_parameter('data_source').get_parameter_value().string_value
        self.cam_num = self.get_parameter('cam_num').get_parameter_value().integer_value
        self.img_dir = self.get_parameter('img_dir').get_parameter_value().string_value
        self.video_path = self.get_parameter('video_path').get_parameter_value().string_value
        self.pub_topic = self.get_parameter('pub_topic').get_parameter_value().string_value
        self.logger = self.get_parameter('logger').get_parameter_value().bool_value
        self.timer_period = self.get_parameter('timer').get_parameter_value().double_value
        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value

        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        self.br = CvBridge()
        self.cap = None

        if self.data_source == 'camera':
            self.cap = cv2.VideoCapture(self.cam_num)

            if not self.cap.isOpened():
                self.get_logger().error(f'Cannot open rear camera: /dev/video{self.cam_num}')
                rclpy.shutdown()
                sys.exit(1)

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        elif self.data_source == 'video':
            self.cap = cv2.VideoCapture(self.video_path)

            if not self.cap.isOpened():
                self.get_logger().error(f'Cannot open video file: {self.video_path}')
                rclpy.shutdown()
                sys.exit(1)

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        elif self.data_source == 'image':
            if os.path.isdir(self.img_dir):
                self.img_list = sorted(os.listdir(self.img_dir))
                self.img_num = 0
            else:
                self.get_logger().error(f'Not a directory: {self.img_dir}')
                rclpy.shutdown()
                sys.exit(1)

        else:
            self.get_logger().error(
                f"Wrong data source: {self.data_source}\n"
                "Check that data_source is 'camera', 'image', or 'video'."
            )
            rclpy.shutdown()
            sys.exit(1)

        self.publisher = self.create_publisher(Image, self.pub_topic, self.qos_profile)
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        self.get_logger().info(f'Rear camera publisher started')
        self.get_logger().info(f'Publish topic: {self.pub_topic}')
        self.get_logger().info(f'Frame ID: {self.frame_id}')
        self.get_logger().info(f'Data source: {self.data_source}')

    def make_image_msg(self, frame):
        image_msg = self.br.cv2_to_imgmsg(frame, encoding='bgr8')
        image_msg.header = Header()
        image_msg.header.stamp = self.get_clock().now().to_msg()
        image_msg.header.frame_id = self.frame_id
        return image_msg

    def timer_callback(self):
        if self.data_source == 'camera':
            ret, frame = self.cap.read()

            if not ret:
                self.get_logger().warn('Failed to read frame from rear camera')
                return

            frame = cv2.resize(frame, (640, 480))
            image_msg = self.make_image_msg(frame)
            self.publisher.publish(image_msg)

            if self.logger:
                cv2.imshow('Rear Camera Image', frame)
                cv2.waitKey(1)

        elif self.data_source == 'image':
            while self.img_num < len(self.img_list):
                img_file = self.img_list[self.img_num]
                img_path = os.path.join(self.img_dir, img_file)
                img = cv2.imread(img_path)

                if img is None:
                    self.get_logger().warn(f'Skipping non-image file: {img_file}')
                else:
                    img = cv2.resize(img, (640, 480))
                    image_msg = self.make_image_msg(img)
                    self.publisher.publish(image_msg)

                    if self.logger:
                        self.get_logger().info(f'Published rear image: {img_file}')
                        cv2.imshow('Rear Saved Image', img)
                        cv2.waitKey(1)

                self.img_num += 1
                break
            else:
                self.img_num = 0

        elif self.data_source == 'video':
            ret, frame = self.cap.read()

            if ret:
                frame = cv2.resize(frame, (640, 480))
                image_msg = self.make_image_msg(frame)
                self.publisher.publish(image_msg)

                if self.logger:
                    cv2.imshow('Rear Video Frame', frame)
                    cv2.waitKey(1)
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)


def main(args=None):
    rclpy.init(args=args)
    node = RearImagePublisherNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('\n\nrear camera node shutdown\n\n')

    node.destroy_node()

    if node.cap is not None and node.cap.isOpened():
        node.cap.release()

    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == '__main__':
    main()