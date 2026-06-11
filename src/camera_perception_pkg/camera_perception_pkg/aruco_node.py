#!/usr/bin/env python3
"""
aruco_yaw_node.py

이미지를 구독하여 아루코 마커를 검출한 뒤,
차량과 마커 사이의 수직축 회전각(Yaw, Degree)만을 
단일 Float32 토픽으로 아주 가볍게 발행하는 노드.
"""

from typing import Optional
import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from scipy.spatial.transform import Rotation as R


class ArucoYawNode(Node):
    def __init__(self) -> None:
        super().__init__("aruco_yaw_node")

        # 파라미터 선언
        self.sub_topic = self.declare_parameter("sub_topic", "image_raw").value
        # 토픽 이름을 더 직관적으로 변경
        self.pub_topic = self.declare_parameter("pub_topic", "aruco_yaw").value
        self.marker_size = float(self.declare_parameter("marker_size", 0.4).value)
        self.show_image = bool(self.declare_parameter("show_image", True).value)

        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.bridge = CvBridge()

        # OpenCV ArUco (버전 호환성)
        try:
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            self.aruco_params = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            self.is_new_api = True
        except AttributeError:
            self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
            self.aruco_params = cv2.aruco.DetectorParameters_create()
            self.is_new_api = False

        # TODO: 실제 사용 시 카메라 캘리브레이션 값으로 교체 필요
        self.camera_matrix = np.array([
            [463.07145, 0.00000, 302.24553],
            [0.00000, 467.14171, 275.52168],
            [0.00000, 0.00000, 1.00000]
        ], dtype=float)

        self.dist_coeffs = np.array([[0.00294, -0.09816, 0.00912, -0.00769, 0.09115]], dtype=float)

        self.subscription = self.create_subscription(
            Image, self.sub_topic, self.image_callback, self.qos_profile
        )
        
        # 배열 대신 단일 Float32 토픽 발행
        self.publisher = self.create_publisher(Float32, self.pub_topic, 10)

        self.get_logger().info(f"ArUco Yaw Node started. Publishes to: {self.pub_topic}")

    def image_callback(self, msg: Image) -> None:
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")
            return

        if self.is_new_api:
            corners, ids, _ = self.detector.detectMarkers(cv_image)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(cv_image, self.aruco_dict, parameters=self.aruco_params)

        if ids is not None and len(ids) > 0:
            # 1. 마커의 실제 3D 모서리 좌표 정의 (순서: 좌상, 우상, 우하, 좌하)
            half_size = self.marker_size / 2.0
            marker_points = np.array([
                [-half_size,  half_size, 0],
                [ half_size,  half_size, 0],
                [ half_size, -half_size, 0],
                [-half_size, -half_size, 0]
            ], dtype=np.float32)

            # 2. 첫 번째로 인식된 마커의 2D 픽셀 좌표
            image_points = corners[0][0]

            # 3. cv2.solvePnP를 사용하여 자세 추정 (최신 OpenCV 표준 방식)
            success, rvec, tvec = cv2.solvePnP(
                marker_points,
                image_points,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE # 정사각형 마커에 최적화된 알고리즘
            )

            if success:
                # 회전 벡터(rvec)를 1차원 배열로 평탄화 후 오일러 각도(Degree)로 변환
                # 1. OpenCV 3D 축 기준 회전값 추출 (X: 상하 기울기, Y: 좌우 틀어짐, Z: 핸들 회전)
                rotation = R.from_rotvec(rvec.flatten())
                rot_x, rot_y, rot_z = rotation.as_euler('xyz', degrees=True)

                # 2. 우리가 진짜 필요한 '주차 평행 판단용 Yaw'는 Y축 회전값(rot_y)입니다!
                real_yaw = rot_y

                # 3. 이 진짜 Yaw 값을 발행
                yaw_msg = Float32()
                yaw_msg.data = float(real_yaw)
                self.publisher.publish(yaw_msg)

                # 화면 시각화
                if self.show_image:
                    cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)
                    try:
                        cv2.drawFrameAxes(cv_image, self.camera_matrix, self.dist_coeffs, rvec, tvec, self.marker_size * 0.5)
                    except AttributeError:
                        # 하위 버전 호환용
                        cv2.aruco.drawAxis(cv_image, self.camera_matrix, self.dist_coeffs, rvec, tvec, self.marker_size * 0.5)

                    cx = int(image_points[0][0])
                    cy = int(image_points[0][1])
                    # 화면 표시 부분 수정
                    cv2.putText(cv_image, f"Yaw: {real_yaw:.1f} deg", (cx, cy - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if self.show_image:
            cv2.imshow("ArUco Yaw Tracking", cv_image)
            cv2.waitKey(1)

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ArucoYawNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()