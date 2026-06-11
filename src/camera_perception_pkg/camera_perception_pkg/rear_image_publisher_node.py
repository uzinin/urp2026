
#!/usr/bin/env python3
"""
rear_image_publisher_node.py

굴절 차량 후방 카메라 영상 발행 노드.
기존 yolov8_node.py가 기본적으로 "image_raw"를 구독하므로,
기본 publish 토픽도 "image_raw"로 설정되어 있다.

지원 입력:
- camera: USB 후방 카메라
- video : 녹화 영상 반복 재생
- image : 폴더 내 이미지 순차 반복 발행
"""

from pathlib import Path
from typing import List, Optional

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import Image


class RearImagePublisherNode(Node):
    def __init__(self) -> None:
        super().__init__("rear_image_publisher_node")

        # 기존 yolov8_node.py와 연결되는 기본 토픽
        self.pub_topic = self.declare_parameter("pub_topic", "image_raw").value

        # Input settings
        self.data_source = str(
            self.declare_parameter("data_source", "camera").value
        ).lower()
        self.cam_num = int(self.declare_parameter("cam_num", 0).value)
        self.image_directory = str(
            self.declare_parameter("image_directory", "").value
        )
        self.video_file = str(self.declare_parameter("video_file", "").value)

        # Output / display settings
        self.width = int(self.declare_parameter("width", 640).value)
        self.height = int(self.declare_parameter("height", 480).value)
        self.timer_period = float(self.declare_parameter("timer", 0.03).value)
        self.show_image = bool(self.declare_parameter("show_image", True).value)
        self.frame_id = str(
            self.declare_parameter("frame_id", "rear_camera_frame").value
        )

        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )

        self.bridge = CvBridge()
        self.cap: Optional[cv2.VideoCapture] = None
        self.image_files: List[Path] = []
        self.image_index = 0

        self._configure_source()

        self.publisher = self.create_publisher(
            Image, self.pub_topic, self.qos_profile
        )
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        self.get_logger().info(
            f"Rear image publisher started: source={self.data_source}, "
            f"topic={self.pub_topic}, size={self.width}x{self.height}"
        )

    def _configure_source(self) -> None:
        if self.data_source == "camera":
            self.cap = cv2.VideoCapture(self.cam_num)
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open rear camera: /dev/video{self.cam_num}")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        elif self.data_source == "video":
            if not self.video_file:
                raise ValueError("video_file parameter is required for data_source:=video")
            self.cap = cv2.VideoCapture(self.video_file)
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open video file: {self.video_file}")

        elif self.data_source == "image":
            if not self.image_directory:
                raise ValueError(
                    "image_directory parameter is required for data_source:=image"
                )
            directory = Path(self.image_directory)
            if not directory.is_dir():
                raise ValueError(f"Image directory does not exist: {directory}")

            supported = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            self.image_files = sorted(
                path for path in directory.iterdir()
                if path.suffix.lower() in supported
            )
            if not self.image_files:
                raise ValueError(f"No image files found in: {directory}")

        else:
            raise ValueError(
                "Invalid data_source. Choose one of: camera, video, image."
            )

    def _read_frame(self) -> Optional[object]:
        if self.data_source in ("camera", "video"):
            if self.cap is None:
                return None
            ret, frame = self.cap.read()
            if not ret and self.data_source == "video":
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
            return frame if ret else None

        image_path = self.image_files[self.image_index]
        frame = cv2.imread(str(image_path))
        self.image_index = (self.image_index + 1) % len(self.image_files)
        if frame is None:
            self.get_logger().warn(f"Could not read image: {image_path}")
        return frame

    def timer_callback(self) -> None:
        frame = self._read_frame()
        if frame is None:
            self.get_logger().warn("Rear frame acquisition failed.")
            return

        frame = cv2.resize(frame, (self.width, self.height))
        image_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        image_msg.header.stamp = self.get_clock().now().to_msg()
        image_msg.header.frame_id = self.frame_id
        self.publisher.publish(image_msg)

        if self.show_image:
            cv2.imshow("Rear Camera Image", frame)
            cv2.waitKey(1)

    def close(self) -> None:
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = RearImagePublisherNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as error:
        if node is not None:
            node.get_logger().error(str(error))
        else:
            print(f"rear_image_publisher_node startup failed: {error}")
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
