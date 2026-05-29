#!/usr/bin/env python3
"""
parking_path_planner_node.py

기존 yolov8_node.py가 발행하는 DetectionArray(detections)를 사용하여
parking / left / right segmentation mask로 주차 경로를 생성하는 노드.

이 노드는 YOLO 모델(best.pt)을 직접 로드하지 않는다.
best.pt는 yolov8_node.py에서만 사용한다.
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

from interfaces_pkg.msg import DetectionArray, PathPlanningResult


class ParkingPathPlannerNode(Node):
    def __init__(self) -> None:
        super().__init__("parking_path_planner_node")

        self.detection_topic = self.declare_parameter("detection_topic", "detections").value
        self.image_topic = self.declare_parameter("image_topic", "image_raw").value
        self.path_topic = self.declare_parameter("path_topic", "path_planning_result").value
        self.debug_topic = self.declare_parameter(
            "debug_image_topic", "parking_path/debug_image"
        ).value

        self.minimum_score = float(self.declare_parameter("minimum_score", 0.50).value)
        self.path_point_count = int(self.declare_parameter("path_point_count", 30).value)
        self.control_distance_ratio = float(
            self.declare_parameter("control_distance_ratio", 0.35).value
        )
        self.minimum_control_distance_px = float(
            self.declare_parameter("minimum_control_distance_px", 20.0).value
        )
        self.show_debug_image = bool(
            self.declare_parameter("show_debug_image", True).value
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

        self.detection_sub = self.create_subscription(
            DetectionArray, self.detection_topic, self.detection_callback, qos
        )
        self.image_sub = self.create_subscription(
            Image, self.image_topic, self.image_callback, qos
        )
        self.path_pub = self.create_publisher(PathPlanningResult, self.path_topic, qos)
        self.debug_pub = self.create_publisher(Image, self.debug_topic, qos)

        self.get_logger().info(
            f"Parking path planner started: subscribe={self.detection_topic}, "
            f"publish={self.path_topic} (YOLO inference is handled by yolov8_node)"
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
    def unit(vector: np.ndarray) -> Optional[np.ndarray]:
        norm = float(np.linalg.norm(vector))
        if norm < 1e-6:
            return None
        return vector.astype(np.float32) / norm

    @staticmethod
    def detection_to_contour(detection) -> Optional[np.ndarray]:
        # yolov8_node.py의 parse_masks()가 넣어주는 segmentation polygon 사용
        if len(detection.mask.data) >= 3:
            return np.array(
                [[float(point.x), float(point.y)] for point in detection.mask.data],
                dtype=np.float32,
            )
        return None

    def select_contours(self, msg: DetectionArray) -> Dict[str, np.ndarray]:
        candidates: Dict[str, List[Tuple[float, np.ndarray]]] = {
            "parking": [],
            "left": [],
            "right": [],
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
                    items,
                    key=lambda item: (cv2.contourArea(item[1]), item[0]),
                )[1]
        return selected

    def parking_axis(self, parking: np.ndarray, vehicle_center: np.ndarray) -> np.ndarray:
        rect = cv2.minAreaRect(parking.astype(np.float32))
        box = cv2.boxPoints(rect).astype(np.float32)
        edge_a = box[1] - box[0]
        edge_b = box[2] - box[1]
        axis = self.unit(edge_a if np.linalg.norm(edge_a) >= np.linalg.norm(edge_b) else edge_b)
        if axis is None:
            return np.array([0.0, -1.0], dtype=np.float32)

        parking_center = np.array(rect[0], dtype=np.float32)
        if float(np.dot(axis, vehicle_center - parking_center)) < 0.0:
            axis = -axis
        return axis

    def generate_path(self, contours: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        left = self.centroid(contours["left"])
        right = self.centroid(contours["right"])
        target = self.centroid(contours["parking"])
        if left is None or right is None or target is None:
            return None

        vehicle = (left + right) / 2.0
        lateral_axis = self.unit(right - left)
        if lateral_axis is None:
            return None

        vehicle_axis = np.array([-lateral_axis[1], lateral_axis[0]], dtype=np.float32)
        if float(np.dot(vehicle_axis, target - vehicle)) < 0.0:
            vehicle_axis = -vehicle_axis

        target_axis = self.parking_axis(contours["parking"], vehicle)
        distance = float(np.linalg.norm(target - vehicle))
        control_distance = max(
            self.minimum_control_distance_px,
            distance * self.control_distance_ratio,
        )

        p0 = vehicle
        p1 = vehicle + vehicle_axis * control_distance
        p2 = target + target_axis * control_distance
        p3 = target

        t = np.linspace(0.0, 1.0, max(3, self.path_point_count), dtype=np.float32)[:, None]
        return (
            (1.0 - t) ** 3 * p0
            + 3.0 * (1.0 - t) ** 2 * t * p1
            + 3.0 * (1.0 - t) * t ** 2 * p2
            + t ** 3 * p3
        )

    def publish_path(self, path: np.ndarray) -> None:
        msg = PathPlanningResult()
        msg.x_points = [float(point[0]) for point in path]
        msg.y_points = [float(point[1]) for point in path]
        self.path_pub.publish(msg)

    def publish_debug(self, contours: Dict[str, np.ndarray], path: Optional[np.ndarray]) -> None:
        if self.latest_image is None:
            return

        debug = self.latest_image.copy()
        colors = {"parking": (0, 255, 0), "left": (255, 0, 0), "right": (0, 0, 255)}

        for name, contour in contours.items():
            cv2.polylines(debug, [contour.astype(np.int32)], True, colors[name], 2)
            center = self.centroid(contour)
            if center is not None:
                point = tuple(center.astype(int))
                cv2.circle(debug, point, 5, colors[name], -1)
                cv2.putText(debug, name, (point[0] + 5, point[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, colors[name], 2)

        if path is not None:
            cv2.polylines(debug, [path.astype(np.int32)], False, (0, 255, 255), 3)
            cv2.circle(debug, tuple(path[0].astype(int)), 6, (0, 255, 255), -1)
            cv2.circle(debug, tuple(path[-1].astype(int)), 7, (0, 255, 0), -1)

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
            if self.missing_counter % 20 == 1:
                self.get_logger().warn(
                    f"Path not published; missing: {sorted(required - set(contours))}"
                )
            self.publish_debug(contours, None)
            return

        path = self.generate_path(contours)
        if path is None:
            self.get_logger().warn("Path geometry calculation failed.")
            self.publish_debug(contours, None)
            return

        self.missing_counter = 0
        self.publish_path(path)
        self.publish_debug(contours, path)

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
