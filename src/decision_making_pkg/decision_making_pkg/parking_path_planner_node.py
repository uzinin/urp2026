#!/usr/bin/env python3
"""
parking_path_planner_node.py

기존 yolov8_node.py가 발행하는 DetectionArray(detections)를 사용하여
parking / left / right segmentation mask로 주차 경로를 생성하는 노드.

- 경로 알고리즘: 3-Point Scipy CubicSpline
- [카메라 하단 중앙(자차) - 좌/우 마커 중앙(트레일러) - 주차 구역 중앙(목표)] 3개의 점을 관통하는 부드러운 스플라인 곡선을 생성합니다.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import Image
from std_msgs.msg import Bool

from interfaces_pkg.msg import DetectionArray, PathPlanningResult
from scipy.interpolate import CubicSpline


class ParkingPathPlannerNode(Node):
    def __init__(self) -> None:
        super().__init__("parking_path_planner_node")

        self.detection_topic = self.declare_parameter("detection_topic", "detections").value
        self.image_topic = self.declare_parameter("image_topic", "image_raw").value
        self.path_topic = self.declare_parameter("path_topic", "path_planning_result").value
        self.debug_topic = self.declare_parameter(
            "debug_image_topic", "parking_path/debug_image"
        ).value
        self.completion_topic = self.declare_parameter(
            "completion_topic", "/parking/complete"
        ).value

        self.minimum_score = float(self.declare_parameter("minimum_score", 0.50).value)
        self.path_point_count = int(self.declare_parameter("path_point_count", 50).value)

        self.show_debug_image = bool(self.declare_parameter("show_debug_image", True).value)
        self.completion_inside_ratio = float(
            self.declare_parameter("completion_inside_ratio", 0.90).value
        )
        self.completion_confirmation_frames = int(
            self.declare_parameter("completion_confirmation_frames", 3).value
        )

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )

        self.bridge = CvBridge()
        self.latest_image: Optional[np.ndarray] = None
        self.latest_header = None
        self.missing_counter = 0
        self.completion_counter = 0
        self.parking_complete_latched = False

        # Debug drawing을 위한 3개의 기착지 저장 변수
        self.last_points: List[np.ndarray] = []

        self.detection_sub = self.create_subscription(
            DetectionArray, self.detection_topic, self.detection_callback, qos
        )
        self.image_sub = self.create_subscription(
            Image, self.image_topic, self.image_callback, qos
        )
        self.path_pub = self.create_publisher(PathPlanningResult, self.path_topic, qos)
        self.debug_pub = self.create_publisher(Image, self.debug_topic, qos)
        self.completion_pub = self.create_publisher(Bool, self.completion_topic, qos)

        self.get_logger().info(
            f"Parking path planner started: subscribe={self.detection_topic}, "
            f"publish={self.path_topic}, complete={self.completion_topic} "
            "(3-Point Spline Path Enabled)"
        )

    def image_callback(self, msg: Image) -> None:
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.latest_header = msg.header
        except CvBridgeError as error:
            self.get_logger().warn(f"Image conversion failed: {error}")

    @staticmethod
    def centroid(contour: np.ndarray) -> Optional[np.ndarray]:
        moments = cv2.moments(contour.astype(np.float32))
        if abs(moments["m00"]) < 1e-6:
            return None
        return np.array(
            [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]],
            dtype=np.float32,
        )

    @staticmethod
    def detection_to_contour(detection) -> Optional[np.ndarray]:
        if len(detection.mask.data) >= 3:
            return np.array(
                [[float(point.x), float(point.y)] for point in detection.mask.data],
                dtype=np.float32,
            )
        return None

    def select_contours(self, msg: DetectionArray) -> Dict[str, np.ndarray]:
        candidates: Dict[str, List[Tuple[float, np.ndarray]]] = {
            "parking": [], "left": [], "right": []
        }

        for detection in msg.detections:
            name = str(detection.class_name).strip().lower()
            if name not in candidates or float(detection.score) < self.minimum_score:
                continue

            contour = self.detection_to_contour(detection)
            if contour is not None:
                candidates[name].append((float(detection.score), contour))

        selected: Dict[str, np.ndarray] = {}
        for name, items in candidates.items():
            if items:
                selected[name] = max(
                    items, key=lambda item: (cv2.contourArea(item[1]), item[0])
                )[1]
        return selected

    @staticmethod
    def contour_inside_ratio(marker: np.ndarray, parking: np.ndarray) -> float:
        all_points = np.vstack((marker, parking))
        x_min = int(np.floor(np.min(all_points[:, 0]))) - 2
        y_min = int(np.floor(np.min(all_points[:, 1]))) - 2
        x_max = int(np.ceil(np.max(all_points[:, 0]))) + 2
        y_max = int(np.ceil(np.max(all_points[:, 1]))) + 2

        width = max(1, x_max - x_min + 1)
        height = max(1, y_max - y_min + 1)

        marker_shifted = marker.copy()
        parking_shifted = parking.copy()
        marker_shifted[:, 0] -= x_min
        marker_shifted[:, 1] -= y_min
        parking_shifted[:, 0] -= x_min
        parking_shifted[:, 1] -= y_min

        marker_mask = np.zeros((height, width), dtype=np.uint8)
        parking_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(marker_mask, [marker_shifted.astype(np.int32)], 1)
        cv2.fillPoly(parking_mask, [parking_shifted.astype(np.int32)], 1)

        marker_area = int(np.count_nonzero(marker_mask))
        if marker_area == 0:
            return 0.0

        intersection = int(np.count_nonzero((marker_mask == 1) & (parking_mask == 1)))
        return intersection / marker_area

    def update_completion(self, contours: Dict[str, np.ndarray]) -> Tuple[bool, float, float]:
        left_ratio = self.contour_inside_ratio(contours["left"], contours["parking"])
        right_ratio = self.contour_inside_ratio(contours["right"], contours["parking"])

        inside_now = (
            left_ratio >= self.completion_inside_ratio
            and right_ratio >= self.completion_inside_ratio
        )

        if not self.parking_complete_latched:
            if inside_now:
                self.completion_counter += 1
            else:
                self.completion_counter = 0

            if self.completion_counter >= max(1, self.completion_confirmation_frames):
                self.parking_complete_latched = True
                self.get_logger().info(
                    "PARKING COMPLETE: left/right markers are inside the parking area. "
                    f"left={left_ratio:.2f}, right={right_ratio:.2f}"
                )

        msg = Bool()
        msg.data = self.parking_complete_latched
        self.completion_pub.publish(msg)
        return self.parking_complete_latched, left_ratio, right_ratio

    def publish_completion_false_if_not_latched(self) -> None:
        msg = Bool()
        msg.data = self.parking_complete_latched
        self.completion_pub.publish(msg)

    def generate_path(self, contours: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        if self.latest_image is None:
            return None

        # 이미지 크기를 바탕으로 카메라 하단 중앙 좌표 도출
        img_height, img_width = self.latest_image.shape[:2]

        left = self.centroid(contours["left"])
        right = self.centroid(contours["right"])
        target = self.centroid(contours["parking"])
        
        if left is None or right is None or target is None:
            return None

        # 1. 카메라 맨 아래 중점 (현재 차량 본체의 위치)
        camera_bottom = np.array([img_width / 2.0, img_height], dtype=np.float32)
        
        # 2. Left / Right 마커의 중점 (트레일러 후미의 위치)
        trailer_mid = (left + right) / 2.0

        # 3. 주차 구역의 중점 (최종 목적지)
        # target 변수 그대로 사용

        # 디버깅 시각화를 위해 저장
        self.last_points = [camera_bottom, trailer_mid, target]

        points = self.last_points
        y_points = [p[1] for p in points]
        x_points = [p[0] for p in points]

        # Y좌표가 증가하는 순서로 정렬 (CubicSpline 필수 조건)
        sorted_pairs = sorted(zip(y_points, x_points), key=lambda pair: pair[0])
        
        unique_y, unique_x = [], []
        for y, x in sorted_pairs:
            # Y값이 겹치면 스플라인 보간이 실패하므로 중복 방지
            if not unique_y or abs(y - unique_y[-1]) > 1e-3:
                unique_y.append(y)
                unique_x.append(x)

        if len(unique_y) < 2:
            self.get_logger().warn("Not enough distinct Y points for spline generation.")
            return None

        try:
            # 점이 3개이므로 안정적인 자연 경계(natural) 조건 적용
            cs = CubicSpline(unique_y, unique_x, bc_type='natural')
            y_new = np.linspace(min(unique_y), max(unique_y), self.path_point_count)
            x_new = cs(y_new)
        except Exception as e:
            self.get_logger().warn(f"Spline interpolation failed: {e}")
            return None

        path = np.column_stack((x_new, y_new)).astype(np.float32)

        # ROS2 배열 순서를 차량(카메라 하단)에서 주차 구역(타겟)으로 향하도록 재정렬
        if np.linalg.norm(path[0] - camera_bottom) > np.linalg.norm(path[-1] - camera_bottom):
            path = path[::-1]

        return path

    def publish_path(self, path: np.ndarray) -> None:
        msg = PathPlanningResult()
        msg.x_points = [float(point[0]) for point in path]
        msg.y_points = [float(point[1]) for point in path]
        self.path_pub.publish(msg)

    def publish_debug(
        self,
        contours: Dict[str, np.ndarray],
        path: Optional[np.ndarray],
        complete: bool = False,
        left_ratio: Optional[float] = None,
        right_ratio: Optional[float] = None,
    ) -> None:
        if self.latest_image is None:
            return

        debug = self.latest_image.copy()
        colors = {"parking": (0, 255, 0), "left": (255, 0, 0), "right": (0, 0, 255)}

        # 마커 및 폴리곤 그리기
        for name, contour in contours.items():
            cv2.polylines(debug, [contour.astype(np.int32)], True, colors[name], 2)
            center = self.centroid(contour)
            if center is not None:
                point = tuple(center.astype(int))
                cv2.circle(debug, point, 5, colors[name], -1)
                cv2.putText(debug, name, (point[0] + 5, point[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, colors[name], 2)

        # 경로 및 3개의 기착지 그리기
        if path is not None:
            cv2.polylines(debug, [path.astype(np.int32)], False, (0, 255, 255), 3)
            
            # 스플라인을 생성할 때 사용한 핵심 3포인트 명시
            if len(self.last_points) == 3:
                cam_pt = tuple(self.last_points[0].astype(int))
                trl_pt = tuple(self.last_points[1].astype(int))
                prk_pt = tuple(self.last_points[2].astype(int))

                cv2.circle(debug, cam_pt, 7, (0, 165, 255), -1)
                cv2.circle(debug, trl_pt, 7, (0, 165, 255), -1)
                cv2.circle(debug, prk_pt, 7, (0, 165, 255), -1)

                cv2.putText(debug, "Car(Cam Bottom)", (cam_pt[0]-50, cam_pt[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                cv2.putText(debug, "Trailer Mid", (trl_pt[0]+10, trl_pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                cv2.putText(debug, "Parking Mid", (prk_pt[0]+10, prk_pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

            cv2.putText(
                debug,
                "Simple 3-Point Spline Path",
                (15, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )

        if left_ratio is not None and right_ratio is not None:
            cv2.putText(
                debug,
                f"inside L:{left_ratio * 100:.0f}% R:{right_ratio * 100:.0f}%",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                (0, 255, 255),
                2,
            )
        if complete:
            cv2.putText(
                debug,
                "PARKING COMPLETE - STOP",
                (15, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (0, 0, 255),
                3,
            )

        debug_msg = self.bridge.cv2_to_imgmsg(debug, encoding="bgr8")
        if self.latest_header is not None:
            debug_msg.header = self.latest_header
        self.debug_pub.publish(debug_msg)

        if self.show_debug_image:
            cv2.imshow("Parking Path Debug", debug)
            cv2.waitKey(1)

    def detection_callback(self, msg: DetectionArray) -> None:
        contours = self.select_contours(msg)
        required = {"parking", "left", "right"}

        if not required.issubset(contours):
            self.missing_counter += 1
            self.publish_completion_false_if_not_latched()
            if self.missing_counter % 20 == 1:
                self.get_logger().warn(
                    f"Path not published; missing: {sorted(required - set(contours))}"
                )
            self.publish_debug(contours, None, self.parking_complete_latched)
            return

        complete, left_ratio, right_ratio = self.update_completion(contours)

        path = self.generate_path(contours)
        if path is None:
            self.get_logger().warn("Path geometry calculation failed.")
            self.publish_debug(contours, None, complete, left_ratio, right_ratio)
            return

        self.missing_counter = 0
        self.publish_path(path)
        self.publish_debug(contours, path, complete, left_ratio, right_ratio)

    def close(self) -> None:
        if self.show_debug_image:
            cv2.destroyAllWindows()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ParkingPathPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()