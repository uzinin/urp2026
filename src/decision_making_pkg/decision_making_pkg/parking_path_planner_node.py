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
from std_msgs.msg import Bool

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
        self.completion_topic = self.declare_parameter(
            "completion_topic", "/parking/complete"
        ).value

        self.minimum_score = float(self.declare_parameter("minimum_score", 0.50).value)
        self.path_point_count = int(self.declare_parameter("path_point_count", 50).value)

        # Normal-intersection path shape:
        # The left/right midpoint normal and the parking long-edge normal are
        # extended until they intersect. That intersection becomes one quadratic
        # Bézier control point, producing a single-bend path with no S-shaped inflection.
        self.maximum_control_distance_ratio = float(
            self.declare_parameter("maximum_control_distance_ratio", 3.0).value
        )
        self.parallel_axis_tolerance_px = float(
            self.declare_parameter("parallel_axis_tolerance_px", 8.0).value
        )
        self.show_debug_image = bool(
            self.declare_parameter("show_debug_image", True).value
        )
        # Marker mask가 parking mask 안에 포함되는 비율로 주차 완료를 판정한다.
        # 1.0으로 두면 segmentation 흔들림으로 완료 판정이 어려울 수 있어 0.90을 기본값으로 사용한다.
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

        # Last successfully generated path geometry for debug drawing.
        self.last_control_point: Optional[np.ndarray] = None
        self.last_lr_normal: Optional[np.ndarray] = None
        self.last_entry_direction: Optional[np.ndarray] = None

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
            "(YOLO inference is handled by yolov8_node)"
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

    @staticmethod
    def contour_inside_ratio(marker: np.ndarray, parking: np.ndarray) -> float:
        """
        Return how much of the marker mask area lies within the parking mask.
        Both polygons are rasterized only in their local bounding rectangle.
        """
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

    def update_completion(
        self, contours: Dict[str, np.ndarray]
    ) -> Tuple[bool, float, float]:
        """
        Publish /parking/complete.
        Once completion is confirmed, it is latched True until the node is restarted.
        """
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

    def parking_axis(self, parking: np.ndarray, vehicle_center: np.ndarray) -> np.ndarray:
        """
        Return the approach axis normal to the long edge of the parking polygon.

        parking 영역이 화면에서 가로로 보일 때, 기존처럼 긴 변 방향을 쓰면
        경로의 마지막 부분이 좌우 방향으로 정렬된다.
        여기서는 긴 변에 수직인 법선 방향을 사용하여, 차량 쪽(화면 아래쪽)에서
        parking 중심으로 직선 진입하는 경로를 만든다.
        """
        rect = cv2.minAreaRect(parking.astype(np.float32))
        box = cv2.boxPoints(rect).astype(np.float32)

        edge_a = box[1] - box[0]
        edge_b = box[2] - box[1]
        long_edge = (
            edge_a if np.linalg.norm(edge_a) >= np.linalg.norm(edge_b) else edge_b
        )

        # parking bounding box의 긴 변에 수직인 진입축
        normal_axis = np.array([-long_edge[1], long_edge[0]], dtype=np.float32)
        axis = self.unit(normal_axis)
        if axis is None:
            return np.array([0.0, 1.0], dtype=np.float32)

        parking_center = np.array(rect[0], dtype=np.float32)
        toward_vehicle = vehicle_center - parking_center

        # 두 법선 방향 중, 현재 후방 차량(left/right)이 있는 쪽을 선택한다.
        # 현재 영상처럼 차량이 parking 아래에 있으면 아래쪽을 향하는 축이 된다.
        if float(np.dot(axis, toward_vehicle)) < 0.0:
            axis = -axis
        return axis

    @staticmethod
    def cross_2d(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
        return float(vector_a[0] * vector_b[1] - vector_a[1] * vector_b[0])

    @staticmethod
    def sample_quadratic_bezier(
        p0: np.ndarray,
        control: np.ndarray,
        p2: np.ndarray,
        count: int,
    ) -> np.ndarray:
        """
        Sample one quadratic Bézier path.

        A quadratic Bézier has one control point; therefore its curvature does
        not reverse direction as an S-shaped cubic path can.
        """
        t = np.linspace(0.0, 1.0, max(3, count), dtype=np.float32)[:, None]
        return (
            (1.0 - t) ** 2 * p0
            + 2.0 * (1.0 - t) * t * control
            + t ** 2 * p2
        )

    def generate_path(self, contours: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        """
        Generate a single-bend path by connecting two measured normal directions.

        P0:
            Midpoint of the rear trailer left/right markers.

        Start tangent:
            Normal vector of the left-right marker line, selected toward parking.

        P2:
            Center of the parking mask.

        End tangent:
            Normal vector of the parking long edge, selected as the direction
            entering into parking from the rear-trailer side.

        Control point:
            Intersection of the P0 start-normal ray and the reverse P2
            entry-normal ray.

        This constructs one quadratic Bézier curve whose tangent at P0 follows
        the rear trailer normal and whose tangent at P2 follows the parking
        entry normal. When the normals cannot form a forward one-bend curve,
        no path is published rather than generating an unsafe S-shaped route.
        """
        left = self.centroid(contours["left"])
        right = self.centroid(contours["right"])
        target = self.centroid(contours["parking"])
        if left is None or right is None or target is None:
            return None

        vehicle = (left + right) / 2.0
        displacement = target - vehicle
        total_distance = float(np.linalg.norm(displacement))
        if total_distance < 1.0:
            return None

        # Normal of the line connecting rear-trailer left/right markers.
        marker_line = self.unit(right - left)
        if marker_line is None:
            return None
        lr_normal = np.array([-marker_line[1], marker_line[0]], dtype=np.float32)
        if float(np.dot(lr_normal, displacement)) < 0.0:
            lr_normal = -lr_normal

        # parking_axis() returns the parking normal pointing toward the vehicle.
        # Reverse it to obtain the final direction entering the parking region.
        parking_toward_vehicle = self.parking_axis(contours["parking"], vehicle)
        entry_direction = -parking_toward_vehicle

        determinant = self.cross_2d(lr_normal, entry_direction)

        # If both normals are parallel and already aligned with the target,
        # the correct path is effectively a straight path.
        if abs(determinant) < 1.0e-5:
            lateral_error = abs(self.cross_2d(lr_normal, displacement))
            same_direction = float(np.dot(lr_normal, entry_direction)) > 0.98
            forward_target = float(np.dot(lr_normal, displacement)) > 0.0

            if (
                lateral_error <= self.parallel_axis_tolerance_px
                and same_direction
                and forward_target
            ):
                control = (vehicle + target) / 2.0
            else:
                self.last_control_point = None
                self.get_logger().warn(
                    "Path not published: LR normal and parking normal are parallel "
                    "but laterally offset, so a one-bend path cannot match both."
                )
                return None
        else:
            # vehicle + a*lr_normal == target - b*entry_direction
            matrix = np.column_stack((lr_normal, entry_direction))
            try:
                a, b = np.linalg.solve(matrix, displacement)
            except np.linalg.LinAlgError:
                self.last_control_point = None
                return None

            if a < 0.0 or b < 0.0:
                self.last_control_point = None
                self.get_logger().warn(
                    "Path not published: detected normals point away from a "
                    "valid forward single-bend approach."
                )
                return None

            maximum_distance = total_distance * self.maximum_control_distance_ratio
            if a > maximum_distance or b > maximum_distance:
                self.last_control_point = None
                self.get_logger().warn(
                    "Path not published: normal intersection is too far away "
                    "for a stable parking path."
                )
                return None

            control = vehicle + lr_normal * float(a)

        self.last_control_point = control.astype(np.float32)
        self.last_lr_normal = lr_normal.astype(np.float32)
        self.last_entry_direction = entry_direction.astype(np.float32)

        return self.sample_quadratic_bezier(
            vehicle,
            self.last_control_point,
            target,
            self.path_point_count,
        )

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

            # Show the measured normal-vector intersection used as the single control point.
            if self.last_control_point is not None:
                control_i = tuple(self.last_control_point.astype(int))
                cv2.line(debug, tuple(path[0].astype(int)), control_i, (255, 255, 0), 1)
                cv2.line(debug, control_i, tuple(path[-1].astype(int)), (255, 0, 255), 1)
                cv2.circle(debug, control_i, 5, (0, 165, 255), -1)

            cv2.putText(
                debug,
                "LR normal - one bend - parking normal",
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

# !/usr/bin/env python3
# """
# parking_path_planner_node.py

# 기존 yolov8_node.py가 발행하는 DetectionArray(detections)를 사용하여
# parking / left / right segmentation mask로 주차 경로를 생성하는 노드.

# 이 노드는 YOLO 모델(best.pt)을 직접 로드하지 않는다.
# best.pt는 yolov8_node.py에서만 사용한다.
# """

# from typing import Dict, List, Optional, Tuple

# import cv2
# import numpy as np
# import rclpy
# from cv_bridge import CvBridge, CvBridgeError
# from rclpy.node import Node
# from rclpy.qos import (
#     QoSDurabilityPolicy,
#     QoSHistoryPolicy,
#     QoSProfile,
#     QoSReliabilityPolicy,
# )
# from sensor_msgs.msg import Image
# from std_msgs.msg import Bool

# from interfaces_pkg.msg import DetectionArray, PathPlanningResult


# class ParkingPathPlannerNode(Node):
#     def __init__(self) -> None:
#         super().__init__("parking_path_planner_node")

#         self.detection_topic = self.declare_parameter("detection_topic", "detections").value
#         self.image_topic = self.declare_parameter("image_topic", "image_raw").value
#         self.path_topic = self.declare_parameter("path_topic", "path_planning_result").value
#         self.debug_topic = self.declare_parameter(
#             "debug_image_topic", "parking_path/debug_image"
#         ).value
#         self.completion_topic = self.declare_parameter(
#             "completion_topic", "/parking/complete"
#         ).value

#         self.minimum_score = float(self.declare_parameter("minimum_score", 0.50).value)
#         self.path_point_count = int(self.declare_parameter("path_point_count", 50).value)

#         # Both-end perpendicular path shape:
#         # One cubic Bézier curve is used. Its first handle is placed vertically above
#         # the rear-trailer midpoint and its second handle is placed below the parking
#         # target along the parking normal. Therefore both endpoints have vertical
#         # tangents while the lateral movement occurs smoothly in the middle.
#         self.start_handle_ratio = float(
#             self.declare_parameter("start_handle_ratio", 0.30).value
#         )
#         self.end_handle_ratio = float(
#             self.declare_parameter("end_handle_ratio", 0.30).value
#         )
#         self.minimum_handle_px = float(
#             self.declare_parameter("minimum_handle_px", 25.0).value
#         )
#         self.maximum_handle_ratio = float(
#             self.declare_parameter("maximum_handle_ratio", 0.55).value
#         )
#         self.show_debug_image = bool(
#             self.declare_parameter("show_debug_image", True).value
#         )
#         # Marker mask가 parking mask 안에 포함되는 비율로 주차 완료를 판정한다.
#         # 1.0으로 두면 segmentation 흔들림으로 완료 판정이 어려울 수 있어 0.90을 기본값으로 사용한다.
#         self.completion_inside_ratio = float(
#             self.declare_parameter("completion_inside_ratio", 0.90).value
#         )
#         self.completion_confirmation_frames = int(
#             self.declare_parameter("completion_confirmation_frames", 3).value
#         )

#         qos = QoSProfile(
#             reliability=QoSReliabilityPolicy.RELIABLE,
#             history=QoSHistoryPolicy.KEEP_LAST,
#             durability=QoSDurabilityPolicy.VOLATILE,
#             depth=1,
#         )

#         self.bridge = CvBridge()
#         self.latest_image: Optional[np.ndarray] = None
#         self.latest_header = None
#         self.missing_counter = 0
#         self.completion_counter = 0
#         self.parking_complete_latched = False

#         self.detection_sub = self.create_subscription(
#             DetectionArray, self.detection_topic, self.detection_callback, qos
#         )
#         self.image_sub = self.create_subscription(
#             Image, self.image_topic, self.image_callback, qos
#         )
#         self.path_pub = self.create_publisher(PathPlanningResult, self.path_topic, qos)
#         self.debug_pub = self.create_publisher(Image, self.debug_topic, qos)
#         self.completion_pub = self.create_publisher(Bool, self.completion_topic, qos)

#         self.get_logger().info(
#             f"Parking path planner started: subscribe={self.detection_topic}, "
#             f"publish={self.path_topic}, complete={self.completion_topic} "
#             "(YOLO inference is handled by yolov8_node)"
#         )

#     def image_callback(self, msg: Image) -> None:
#         try:
#             self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
#             self.latest_header = msg.header
#         except CvBridgeError as error:
#             self.get_logger().warn(f"Image conversion failed: {error}")

#     @staticmethod
#     def centroid(contour: np.ndarray) -> Optional[np.ndarray]:
#         moments = cv2.moments(contour.astype(np.float32))
#         if abs(moments["m00"]) < 1e-6:
#             return None
#         return np.array(
#             [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]],
#             dtype=np.float32,
#         )

#     @staticmethod
#     def unit(vector: np.ndarray) -> Optional[np.ndarray]:
#         norm = float(np.linalg.norm(vector))
#         if norm < 1e-6:
#             return None
#         return vector.astype(np.float32) / norm

#     @staticmethod
#     def detection_to_contour(detection) -> Optional[np.ndarray]:
#         # yolov8_node.py의 parse_masks()가 넣어주는 segmentation polygon 사용
#         if len(detection.mask.data) >= 3:
#             return np.array(
#                 [[float(point.x), float(point.y)] for point in detection.mask.data],
#                 dtype=np.float32,
#             )
#         return None

#     def select_contours(self, msg: DetectionArray) -> Dict[str, np.ndarray]:
#         candidates: Dict[str, List[Tuple[float, np.ndarray]]] = {
#             "parking": [],
#             "left": [],
#             "right": [],
#         }

#         for detection in msg.detections:
#             name = str(detection.class_name).strip().lower()
#             if name not in candidates or float(detection.score) < self.minimum_score:
#                 continue

#             contour = self.detection_to_contour(detection)
#             if contour is not None:
#                 candidates[name].append((float(detection.score), contour))

#         selected: Dict[str, np.ndarray] = {}
#         for name, items in candidates.items():
#             if items:
#                 selected[name] = max(
#                     items,
#                     key=lambda item: (cv2.contourArea(item[1]), item[0]),
#                 )[1]
#         return selected

#     @staticmethod
#     def contour_inside_ratio(marker: np.ndarray, parking: np.ndarray) -> float:
#         """
#         Return how much of the marker mask area lies within the parking mask.
#         Both polygons are rasterized only in their local bounding rectangle.
#         """
#         all_points = np.vstack((marker, parking))
#         x_min = int(np.floor(np.min(all_points[:, 0]))) - 2
#         y_min = int(np.floor(np.min(all_points[:, 1]))) - 2
#         x_max = int(np.ceil(np.max(all_points[:, 0]))) + 2
#         y_max = int(np.ceil(np.max(all_points[:, 1]))) + 2

#         width = max(1, x_max - x_min + 1)
#         height = max(1, y_max - y_min + 1)

#         marker_shifted = marker.copy()
#         parking_shifted = parking.copy()
#         marker_shifted[:, 0] -= x_min
#         marker_shifted[:, 1] -= y_min
#         parking_shifted[:, 0] -= x_min
#         parking_shifted[:, 1] -= y_min

#         marker_mask = np.zeros((height, width), dtype=np.uint8)
#         parking_mask = np.zeros((height, width), dtype=np.uint8)
#         cv2.fillPoly(marker_mask, [marker_shifted.astype(np.int32)], 1)
#         cv2.fillPoly(parking_mask, [parking_shifted.astype(np.int32)], 1)

#         marker_area = int(np.count_nonzero(marker_mask))
#         if marker_area == 0:
#             return 0.0

#         intersection = int(np.count_nonzero((marker_mask == 1) & (parking_mask == 1)))
#         return intersection / marker_area

#     def update_completion(
#         self, contours: Dict[str, np.ndarray]
#     ) -> Tuple[bool, float, float]:
#         """
#         Publish /parking/complete.
#         Once completion is confirmed, it is latched True until the node is restarted.
#         """
#         left_ratio = self.contour_inside_ratio(contours["left"], contours["parking"])
#         right_ratio = self.contour_inside_ratio(contours["right"], contours["parking"])

#         inside_now = (
#             left_ratio >= self.completion_inside_ratio
#             and right_ratio >= self.completion_inside_ratio
#         )

#         if not self.parking_complete_latched:
#             if inside_now:
#                 self.completion_counter += 1
#             else:
#                 self.completion_counter = 0

#             if self.completion_counter >= max(1, self.completion_confirmation_frames):
#                 self.parking_complete_latched = True
#                 self.get_logger().info(
#                     "PARKING COMPLETE: left/right markers are inside the parking area. "
#                     f"left={left_ratio:.2f}, right={right_ratio:.2f}"
#                 )

#         msg = Bool()
#         msg.data = self.parking_complete_latched
#         self.completion_pub.publish(msg)
#         return self.parking_complete_latched, left_ratio, right_ratio

#     def publish_completion_false_if_not_latched(self) -> None:
#         msg = Bool()
#         msg.data = self.parking_complete_latched
#         self.completion_pub.publish(msg)

#     def parking_axis(self, parking: np.ndarray, vehicle_center: np.ndarray) -> np.ndarray:
#         """
#         Return the approach axis normal to the long edge of the parking polygon.

#         parking 영역이 화면에서 가로로 보일 때, 기존처럼 긴 변 방향을 쓰면
#         경로의 마지막 부분이 좌우 방향으로 정렬된다.
#         여기서는 긴 변에 수직인 법선 방향을 사용하여, 차량 쪽(화면 아래쪽)에서
#         parking 중심으로 직선 진입하는 경로를 만든다.
#         """
#         rect = cv2.minAreaRect(parking.astype(np.float32))
#         box = cv2.boxPoints(rect).astype(np.float32)

#         edge_a = box[1] - box[0]
#         edge_b = box[2] - box[1]
#         long_edge = (
#             edge_a if np.linalg.norm(edge_a) >= np.linalg.norm(edge_b) else edge_b
#         )

#         # parking bounding box의 긴 변에 수직인 진입축
#         normal_axis = np.array([-long_edge[1], long_edge[0]], dtype=np.float32)
#         axis = self.unit(normal_axis)
#         if axis is None:
#             return np.array([0.0, 1.0], dtype=np.float32)

#         parking_center = np.array(rect[0], dtype=np.float32)
#         toward_vehicle = vehicle_center - parking_center

#         # 두 법선 방향 중, 현재 후방 차량(left/right)이 있는 쪽을 선택한다.
#         # 현재 영상처럼 차량이 parking 아래에 있으면 아래쪽을 향하는 축이 된다.
#         if float(np.dot(axis, toward_vehicle)) < 0.0:
#             axis = -axis
#         return axis

#     @staticmethod
#     def sample_cubic_bezier(
#         p0: np.ndarray,
#         p1: np.ndarray,
#         p2: np.ndarray,
#         p3: np.ndarray,
#         count: int,
#     ) -> np.ndarray:
#         """
#         Sample one cubic Bézier curve.
#         The curve remains one continuous path while independently controlling
#         the initial and final tangent directions.
#         """
#         t = np.linspace(0.0, 1.0, max(4, count), dtype=np.float32)[:, None]
#         return (
#             (1.0 - t) ** 3 * p0
#             + 3.0 * (1.0 - t) ** 2 * t * p1
#             + 3.0 * (1.0 - t) * t ** 2 * p2
#             + t ** 3 * p3
#         )

#     def generate_path(self, contours: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
#         """
#         Generate one continuous lane-change-like curve for the rear trailer.

#         P0: midpoint of rear-trailer left/right markers.
#         P1: vertically above P0 toward the parking side, so departure is vertical.
#         P2: on the parking normal below P3, so entry is perpendicular/vertical.
#         P3: center of the parking mask.

#         Because the rear trailer and parking target are laterally offset while both
#         end tangents are vertical, the path naturally shifts sideways in its middle.
#         """
#         left = self.centroid(contours["left"])
#         right = self.centroid(contours["right"])
#         target = self.centroid(contours["parking"])
#         if left is None or right is None or target is None:
#             return None

#         vehicle = (left + right) / 2.0
#         total_distance = float(np.linalg.norm(target - vehicle))
#         if total_distance < 1.0:
#             return None

#         # Direction from parking center toward the visible rear trailer.
#         # For the shown rear-camera layout, this points downward from parking.
#         target_axis = self.parking_axis(contours["parking"], vehicle)

#         # Start departure direction must point from the trailer toward parking.
#         # Use the opposite of the parking-to-vehicle normal so that the beginning
#         # tangent is parallel to, and enters toward, the parking normal direction.
#         start_direction = -target_axis

#         start_requested = max(
#             self.minimum_handle_px,
#             total_distance * self.start_handle_ratio,
#         )
#         end_requested = max(
#             self.minimum_handle_px,
#             total_distance * self.end_handle_ratio,
#         )
#         max_handle = total_distance * self.maximum_handle_ratio
#         start_handle = min(start_requested, max_handle)
#         end_handle = min(end_requested, max_handle)

#         p0 = vehicle
#         p1 = vehicle + start_direction * start_handle
#         p2 = target + target_axis * end_handle
#         p3 = target

#         return self.sample_cubic_bezier(
#             p0, p1, p2, p3, self.path_point_count
#         )

#     def publish_path(self, path: np.ndarray) -> None:
#         msg = PathPlanningResult()
#         msg.x_points = [float(point[0]) for point in path]
#         msg.y_points = [float(point[1]) for point in path]
#         self.path_pub.publish(msg)

#     def publish_debug(
#         self,
#         contours: Dict[str, np.ndarray],
#         path: Optional[np.ndarray],
#         complete: bool = False,
#         left_ratio: Optional[float] = None,
#         right_ratio: Optional[float] = None,
#     ) -> None:
#         if self.latest_image is None:
#             return

#         debug = self.latest_image.copy()
#         colors = {"parking": (0, 255, 0), "left": (255, 0, 0), "right": (0, 0, 255)}

#         for name, contour in contours.items():
#             cv2.polylines(debug, [contour.astype(np.int32)], True, colors[name], 2)
#             center = self.centroid(contour)
#             if center is not None:
#                 point = tuple(center.astype(int))
#                 cv2.circle(debug, point, 5, colors[name], -1)
#                 cv2.putText(debug, name, (point[0] + 5, point[1] - 5),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.55, colors[name], 2)

#         if path is not None:
#             cv2.polylines(debug, [path.astype(np.int32)], False, (0, 255, 255), 3)
#             cv2.circle(debug, tuple(path[0].astype(int)), 6, (0, 255, 255), -1)
#             cv2.circle(debug, tuple(path[-1].astype(int)), 7, (0, 255, 0), -1)
#             cv2.putText(
#                 debug,
#                 "vertical start - smooth shift - vertical entry",
#                 (15, 95),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.55,
#                 (0, 255, 255),
#                 2,
#             )

#         if left_ratio is not None and right_ratio is not None:
#             cv2.putText(
#                 debug,
#                 f"inside L:{left_ratio * 100:.0f}% R:{right_ratio * 100:.0f}%",
#                 (15, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.70,
#                 (0, 255, 255),
#                 2,
#             )
#         if complete:
#             cv2.putText(
#                 debug,
#                 "PARKING COMPLETE - STOP",
#                 (15, 65),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.85,
#                 (0, 0, 255),
#                 3,
#             )

#         debug_msg = self.bridge.cv2_to_imgmsg(debug, encoding="bgr8")
#         if self.latest_header is not None:
#             debug_msg.header = self.latest_header
#         self.debug_pub.publish(debug_msg)

#         if self.show_debug_image:
#             cv2.imshow("Parking Path Debug", debug)
#             cv2.waitKey(1)

#     def detection_callback(self, msg: DetectionArray) -> None:
#         contours = self.select_contours(msg)
#         required = {"parking", "left", "right"}

#         if not required.issubset(contours):
#             self.missing_counter += 1
#             self.publish_completion_false_if_not_latched()
#             if self.missing_counter % 20 == 1:
#                 self.get_logger().warn(
#                     f"Path not published; missing: {sorted(required - set(contours))}"
#                 )
#             self.publish_debug(contours, None, self.parking_complete_latched)
#             return

#         complete, left_ratio, right_ratio = self.update_completion(contours)

#         path = self.generate_path(contours)
#         if path is None:
#             self.get_logger().warn("Path geometry calculation failed.")
#             self.publish_debug(contours, None, complete, left_ratio, right_ratio)
#             return

#         self.missing_counter = 0
#         self.publish_path(path)
#         self.publish_debug(contours, path, complete, left_ratio, right_ratio)

#     def close(self) -> None:
#         if self.show_debug_image:
#             cv2.destroyAllWindows()


# def main(args=None) -> None:
#     rclpy.init(args=args)
#     node = ParkingPathPlannerNode()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.close()
#         node.destroy_node()
#         rclpy.shutdown()


# if __name__ == "__main__":
#     main()