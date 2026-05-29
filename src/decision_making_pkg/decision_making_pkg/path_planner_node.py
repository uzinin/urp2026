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

        # Path shape:
        # - start/end portions are explicit straight segments.
        # - steering curvature is concentrated in the middle Bézier segment.
        self.straight_section_ratio = float(
            self.declare_parameter("straight_section_ratio", 0.18).value
        )
        self.minimum_straight_length_px = float(
            self.declare_parameter("minimum_straight_length_px", 12.0).value
        )
        self.maximum_straight_section_ratio = float(
            self.declare_parameter("maximum_straight_section_ratio", 0.28).value
        )
        self.middle_handle_ratio = float(
            self.declare_parameter("middle_handle_ratio", 0.45).value
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
    def sample_line(start: np.ndarray, end: np.ndarray, count: int) -> np.ndarray:
        """Sample a straight segment, including start and end points."""
        t = np.linspace(0.0, 1.0, max(2, count), dtype=np.float32)[:, None]
        return (1.0 - t) * start + t * end

    @staticmethod
    def sample_cubic_bezier(
        p0: np.ndarray,
        p1: np.ndarray,
        p2: np.ndarray,
        p3: np.ndarray,
        count: int,
    ) -> np.ndarray:
        """Sample a cubic Bézier middle turning segment."""
        t = np.linspace(0.0, 1.0, max(3, count), dtype=np.float32)[:, None]
        return (
            (1.0 - t) ** 3 * p0
            + 3.0 * (1.0 - t) ** 2 * t * p1
            + 3.0 * (1.0 - t) * t ** 2 * p2
            + t ** 3 * p3
        )

    def generate_path(self, contours: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        """
        Generate a three-part parking path for the rear trailer:
          1) short straight departure from the left/right marker midpoint,
          2) central cubic Bézier turn where most curvature occurs,
          3) straight final alignment into the parking target.

        This makes the line visually and geometrically straighter near both ends,
        instead of bending immediately at the marker or parking center.
        """
        left = self.centroid(contours["left"])
        right = self.centroid(contours["right"])
        target = self.centroid(contours["parking"])
        if left is None or right is None or target is None:
            return None

        vehicle = (left + right) / 2.0
        lateral_axis = self.unit(right - left)
        if lateral_axis is None:
            return None

        # left-right marker line is lateral; its perpendicular is the rear trailer axis.
        vehicle_axis = np.array([-lateral_axis[1], lateral_axis[0]], dtype=np.float32)
        if float(np.dot(vehicle_axis, target - vehicle)) < 0.0:
            vehicle_axis = -vehicle_axis

        # Axis points from parking center toward the approaching vehicle.
        # Therefore final movement into parking follows -target_axis.
        target_axis = self.parking_axis(contours["parking"], vehicle)

        total_distance = float(np.linalg.norm(target - vehicle))
        if total_distance < 1.0:
            return None

        requested_straight = max(
            self.minimum_straight_length_px,
            total_distance * self.straight_section_ratio,
        )
        straight_length = min(
            requested_straight,
            total_distance * self.maximum_straight_section_ratio,
        )

        # Exact straight sections at both ends.
        start_straight_end = vehicle + vehicle_axis * straight_length
        final_straight_start = target + target_axis * straight_length

        middle_vector = final_straight_start - start_straight_end
        middle_distance = float(np.linalg.norm(middle_vector))

        # When the points are already too close, fall back to one aligned Bézier.
        if middle_distance < 5.0:
            handle = total_distance * 0.35
            return self.sample_cubic_bezier(
                vehicle,
                vehicle + vehicle_axis * handle,
                target + target_axis * handle,
                target,
                max(3, self.path_point_count),
            )

        # Tangents of the central curve continue from / into the straight segments.
        handle_length = max(10.0, middle_distance * self.middle_handle_ratio)
        curve_p0 = start_straight_end
        curve_p1 = curve_p0 + vehicle_axis * handle_length
        curve_p3 = final_straight_start
        curve_p2 = curve_p3 + target_axis * handle_length

        total_count = max(15, self.path_point_count)
        straight_count = max(3, int(round(total_count * self.straight_section_ratio)))
        middle_count = max(7, total_count - 2 * straight_count + 2)

        start_line = self.sample_line(vehicle, start_straight_end, straight_count)
        middle_curve = self.sample_cubic_bezier(
            curve_p0, curve_p1, curve_p2, curve_p3, middle_count
        )
        end_line = self.sample_line(final_straight_start, target, straight_count)

        # Remove duplicate junction points before concatenating.
        return np.vstack((start_line, middle_curve[1:], end_line[1:]))

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
            cv2.putText(
                debug,
                "straight - central turn - perpendicular entry",
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

# import rclpy
# from rclpy.node import Node
# from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, QoSReliabilityPolicy
# from interfaces_pkg.msg import LaneInfo, PathPlanningResult
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.interpolate import CubicSpline

# #---------------Variable Setting---------------
# SUB_LANE_TOPIC_NAME = "yolov8_lane_info"  # lane_info_extractor 노드에서 퍼블리시하는 타겟 지점 토픽
# PUB_TOPIC_NAME = "path_planning_result"   # 경로 계획 결과 퍼블리시 토픽
# CENTER_X = 320
# CAR_CENTER_POINT = (CENTER_X, 179) # 이미지 상에서 차량 앞 범퍼의 중심이 위치한 픽셀 좌표

# #----------------------------------------------
# class PathPlannerNode(Node):
#     def __init__(self):
#         super().__init__('path_planner_node')

#         # 파라미터 선언
#         self.sub_lane_topic = self.declare_parameter('sub_lane_topic', SUB_LANE_TOPIC_NAME).value
#         self.pub_topic = self.declare_parameter('pub_topic', PUB_TOPIC_NAME).value
#         self.car_center_point = self.declare_parameter('car_center_point', CAR_CENTER_POINT).value
        
#         # QoS 설정
#         self.qos_profile = QoSProfile(
#             reliability=QoSReliabilityPolicy.RELIABLE,
#             history=QoSHistoryPolicy.KEEP_LAST,
#             durability=QoSDurabilityPolicy.VOLATILE,
#             depth=1
#         )

#         # 변수 초기화
#         self.target_points = []  # 차선의 타겟 지점들 (차선 중앙)

#         # 8조 추가변수
#         # self.cnt_dead = 0

#         # 서브스크라이버 설정 (타겟 지점 구독)
#         self.lane_sub = self.create_subscription(LaneInfo, self.sub_lane_topic, self.lane_callback, self.qos_profile)

#         # 퍼블리셔 설정 (경로 계획 결과 퍼블리시)
#         self.publisher = self.create_publisher(PathPlanningResult, self.pub_topic, self.qos_profile)

#     def lane_callback(self, msg: LaneInfo):
        
#         # 타겟 지점 받아오기
#         self.target_points = msg.target_points
        
#         # 타겟 지점이 3개 이상 모이면 경로 계획 시작
#         if len(self.target_points) >= 3:
#             self.plan_path()
#             # self.cnt_dead = 0
#         # else:
#         #     self.cnt_dead += 1
#         #     if self.cnt_dead >= 30:
#         #         self.target_points[0].target_x = CENTER_X 
#         #         self.target_points[0].target_y = 5
#         #         self.target_points[1].target_x = CENTER_X 
#         #         self.target_points[1].target_y = 55
#         #         self.target_points[2].target_x = CENTER_X 
#         #         self.target_points[2].target_y = 105
#         #         self.target_points[3].target_x = CENTER_X 
#         #         self.target_points[3].target_y = 155

#     def plan_path(self):
#         # self.target_points가 TargetPoint 객체들의 리스트라고 가정
#         if not self.target_points:
#             self.get_logger().warn("No target points available")
#             return
        
#         # TargetPoint 객체에서 x, y 값 추출
#         x_points, y_points = zip(*[(tp.target_x, tp.target_y) for tp in self.target_points])

#         #차량 앞 범퍼의 중심이 위치한 픽셀 좌표 추가
#         y_points_list, x_points_list = list(y_points), list(x_points) 
#         y_points_list.append(self.car_center_point[1])
#         x_points_list.append(self.car_center_point[0])
#         y_points, x_points = tuple(y_points_list), tuple(x_points_list)
        
#         # y 값을 기준으로 정렬 (y가 증가하는 순서로 정렬)
#         sorted_points = sorted(zip(y_points, x_points), key=lambda point: point[0])

#         # 정렬된 y, x 값을 다시 분리
#         y_points, x_points = zip(*sorted_points)
        
#         # 몇개의 점으로 경로 계획을 하는지 확인
#         self.get_logger().info(f"Planning path with {len(y_points)} points")

#         # 스플라인 보간법을 사용하여 경로 생성
#         cs = CubicSpline(y_points, x_points, bc_type='natural')

#         # 생성된 경로 점들 (추가적인 점들을 생성하여 부드러운 경로를 얻음)
#         y_new = np.linspace(min(y_points), max(y_points), 100)
#         x_new = cs(y_new)

#         # 경로를 따라가는 정보 (PathPlanningResult 메시지로 발행)
#         path_msg = PathPlanningResult()
#         path_msg.x_points = list(x_new)
#         path_msg.y_points = list(y_new)

#         # print(f'path_msg_x :  {path_msg.x_points[0]}')
#         # print(f'path_msg_y :  {path_msg.y_points[0]}')

#         # 경로 퍼블리시
#         self.publisher.publish(path_msg)

#         # 타겟 지점 초기화 (다음 경로 계산을 위해)
#         self.target_points.clear()


# def main(args=None):
#     rclpy.init(args=args)
#     node = PathPlannerNode()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         print("\n\nshutdown\n\n")
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()

# if __name__ == '__main__':
#     main()
