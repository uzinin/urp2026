#!/usr/bin/env python3
"""
curved_trailer_parking_motion_planner_node.py

State-machine motion planner for curved reverse parking with an articulated
trailer vehicle. (Upgraded with variable proportional steering & axis alignment checks)

Inputs:
  - detections (interfaces_pkg/DetectionArray): YOLO detections for left/right/parking
  - path_planning_result (interfaces_pkg/PathPlanningResult): parking path points
  - /articulation/angle (std_msgs/Float32): articulation angle from serial_sender

Output:
  - topic_control_signal (interfaces_pkg/MotionCommand): steering and motor command

Firmware steering convention follows the existing nodes:
  steering -7..7, positive is right, negative is left.
  speed < 0 is reverse.
"""

import time
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from std_msgs.msg import Float32

from interfaces_pkg.msg import DetectionArray, MotionCommand, PathPlanningResult


Point = Tuple[float, float]

SUB_DETECTION_TOPIC_NAME = "detections"
SUB_PATH_TOPIC_NAME = "path_planning_result"
SUB_HITCH_TOPIC_NAME = "/articulation/angle"
PUB_TOPIC_NAME = "topic_control_signal"

CLS_LEFT = "left"
CLS_RIGHT = "right"
CLS_PARKING = "parking"

IMG_W = 640.0
IMG_CX = IMG_W / 2.0

TIMER = 0.1
MAX_STEP = 7
REVERSE_SPEED = -40
STOP_SPEED = 0


class ParkingState:
    WAIT_PATH = "WAIT_PATH"
    STATE1_INITIAL_TURN = "STATE1_INITIAL_TURN"
    STATE2_COUNTER_TURN = "STATE2_COUNTER_TURN"
    STATE2_NEUTRAL_FORWARD = "STATE2_NEUTRAL_FORWARD"
    STATE3_NEUTRAL_REVERSE = "STATE3_NEUTRAL_REVERSE"
    STATE4_MISSING_CONFIRM = "STATE4_MISSING_CONFIRM"
    JACKKNIFE_STOP = "JACKKNIFE_STOP"
    JACKKNIFE_FORWARD_RECOVERY = "JACKKNIFE_FORWARD_RECOVERY"
    DONE = "DONE"


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class CurvedTrailerParkingMotionPlannerNode(Node):
    def __init__(self) -> None:
        super().__init__("curved_trailer_parking_motion_planner_node")

        self.sub_detection_topic = self.declare_parameter(
            "sub_detection_topic", SUB_DETECTION_TOPIC_NAME
        ).value
        self.sub_path_topic = self.declare_parameter(
            "sub_lane_topic", SUB_PATH_TOPIC_NAME
        ).value
        self.sub_hitch_topic = self.declare_parameter(
            "sub_hitch_topic", SUB_HITCH_TOPIC_NAME
        ).value
        self.pub_topic = self.declare_parameter("pub_topic", PUB_TOPIC_NAME).value
        self.timer_period = float(self.declare_parameter("timer", TIMER).value)

        # ========================= TUNING PARAMETERS =========================
        self.reverse_speed = int(
            self.declare_parameter("reverse_speed", REVERSE_SPEED).value
        )
        self.forward_speed = int(
            self.declare_parameter("forward_speed", abs(60)).value
        )

        # 조향 가변형 제어 및 정렬을 위한 최대 한계 조향값으로 유지 사용
        self.initial_turn_step = int(
            self.declare_parameter("initial_turn_step", MAX_STEP).value
        )
        self.counter_turn_step = int(
            self.declare_parameter("counter_turn_step", 7).value
        )

        # 🌟 [새로 추가] 가변 조향 비례 게인 (축 정렬 오차각도 수치를 조향 스텝으로 변환)
        # 틀어진 각도에 곱해지는 값이므로 실험을 통해 민감도를 조절하세요.
        self.k_angle_state1 = float(self.declare_parameter("k_angle_state1", 0.35).value)
        self.k_angle_state2 = float(self.declare_parameter("k_angle_state2", 0.50).value)

        # 🌟 [새로 추가] 주차 구역 축과 차량 축의 정렬 허용 오차각 (도 단위)
        # 이 각도 이내로 진입 각도가 좁혀져야 평행으로 인정합니다.
        self.heading_tolerance_deg = float(
            self.declare_parameter("heading_tolerance_deg", 6.0).value
        )

        self.steering_sign = int(self.declare_parameter("steering_sign", -1).value)
        self.max_step_delta = int(self.declare_parameter("max_step_delta", 2).value)

        self.state1_hitch_trigger_deg = float(
            self.declare_parameter("state1_hitch_trigger_deg", 25.0).value
        )
        self.center_tolerance_px = float(
            self.declare_parameter("center_tolerance_px", 35.0).value
        )
        self.parking_horizontal_tolerance_deg = float(
            self.declare_parameter("parking_horizontal_tolerance_deg", 20.0).value
        )
        self.parking_orientation_min_aspect_ratio = float(
            self.declare_parameter("parking_orientation_min_aspect_ratio", 1.2).value
        )
        self.hitch_zero_tolerance_deg = float(
            self.declare_parameter("hitch_zero_tolerance_deg", 2.0).value
        )
        self.neutral_forward_duration_sec = float(
            self.declare_parameter("neutral_forward_duration_sec", 4.0).value
        )
        self.parking_missing_complete_sec = float(
            self.declare_parameter("parking_missing_complete_sec", 2.0).value
        )
        self.parking_detection_timeout_sec = float(
            self.declare_parameter("parking_detection_timeout_sec", 0.30).value
        )
        self.jackknife_limit_deg = float(
            self.declare_parameter("jackknife_limit_deg", 40.0).value
        )
        self.jackknife_stop_duration_sec = float(
            self.declare_parameter("jackknife_stop_duration_sec", 1.0).value
        )
        self.jackknife_detection_duration_sec = float(
            self.declare_parameter("jackknife_detection_duration_sec", 0.2).value
        )
        self.jackknife_recovery_target_deg = float(
            self.declare_parameter(
                "jackknife_recovery_target_deg",
                self.state1_hitch_trigger_deg,
            ).value
        )
        self.minimum_score = float(self.declare_parameter("minimum_score", 0.50).value)
        self.show_log = bool(self.declare_parameter("show_log", True).value)
        # =====================================================================

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )

        self.detection_sub = self.create_subscription(
            DetectionArray, self.sub_detection_topic, self.detection_callback, qos
        )
        self.path_sub = self.create_subscription(
            PathPlanningResult, self.sub_path_topic, self.path_callback, qos
        )
        self.hitch_sub = self.create_subscription(
            Float32, self.sub_hitch_topic, self.hitch_callback, qos
        )
        self.publisher = self.create_publisher(MotionCommand, self.pub_topic, qos)

        self.path_data: Optional[List[Point]] = None
        self.last_parking_center_x: Optional[float] = None
        self.last_marker_midpoint_x: Optional[float] = None
        self.last_alignment_seen_time: Optional[float] = None
        self.last_parking_is_horizontal = False
        self.last_parking_angle_deg: Optional[float] = None
        self.last_parking_seen_time: Optional[float] = None
        self.current_hitch_angle = 0.0

        self.parking_side = 1
        self.state = ParkingState.WAIT_PATH
        self.last_steering = 0
        self.missing_start_time: Optional[float] = None
        self.neutral_forward_start_time: Optional[float] = None
        self.jackknife_stop_start_time: Optional[float] = None
        self.jackknife_over_limit_start_time: Optional[float] = None

        # 🌟 실시간 기하학적 연산을 위한 변수 추가
        self.current_heading_error = 0.0

        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        self.get_logger().info(
            "curved_trailer_parking_motion_planner_node started with Variable Steering Control"
        )

    def detection_callback(self, msg: DetectionArray) -> None:
        perception = self.extract_perception(msg)
        now = time.time()

        parking = perception.get(CLS_PARKING)
        if parking is not None:
            self.last_parking_center_x = parking["center_x"]
            self.last_parking_seen_time = now
            self.missing_start_time = None
            self.parking_side = self.side_from_x(parking["center_x"])
            (
                self.last_parking_is_horizontal,
                self.last_parking_angle_deg,
            ) = self.parking_horizontal_alignment(parking.get("contour"))

        left = perception.get(CLS_LEFT)
        right = perception.get(CLS_RIGHT)
        if parking is not None and left is not None and right is not None:
            self.last_marker_midpoint_x = (
                float(left["center_x"]) + float(right["center_x"])
            ) / 2.0
            self.last_alignment_seen_time = now

        if self.state in (
            ParkingState.STATE3_NEUTRAL_REVERSE,
            ParkingState.STATE4_MISSING_CONFIRM,
        ):
            if self.markers_inside_parking(perception):
                self.complete("left/right marker centers reached parking boundary")

    def path_callback(self, msg: PathPlanningResult) -> None:
        if len(msg.x_points) != len(msg.y_points) or len(msg.x_points) < 2:
            return

        self.path_data = list(zip(msg.x_points, msg.y_points))
        if self.last_parking_center_x is None:
            self.parking_side = self.side_from_path(self.path_data)

        if self.state == ParkingState.WAIT_PATH:
            self.state = ParkingState.STATE1_INITIAL_TURN
            self.log("path received -> state1")

    def hitch_callback(self, msg: Float32) -> None:
        self.current_hitch_angle = float(msg.data)

    @staticmethod
    def distance(point_a: Point, point_b: Point) -> float:
        return math.hypot(point_b[0] - point_a[0], point_b[1] - point_a[1])

    @staticmethod
    def signed_angle_between(vector_a: Point, vector_b: Point) -> float:
        """두 2차원 벡터 사이의 오차각을 부호 포함 도(Degree) 단위로 계산"""
        cross = vector_a[0] * vector_b[1] - vector_a[1] * vector_b[0]
        dot = vector_a[0] * vector_b[0] + vector_a[1] * vector_b[1]
        return math.degrees(math.atan2(cross, dot))

    def timer_callback(self) -> None:
        if self.state == ParkingState.DONE:
            self.publish_command(0, STOP_SPEED)
            return

        if self.path_data is None and self.state == ParkingState.WAIT_PATH:
            self.publish_command(0, STOP_SPEED)
            self.log("waiting for first path")
            return

        # 잭나이프 예외 제어 (기존 원본 구조 계승)
        if self.state == ParkingState.JACKKNIFE_STOP:
            self.publish_command(0, STOP_SPEED)
            if self.jackknife_stop_elapsed() >= self.jackknife_stop_duration_sec:
                self.state = ParkingState.JACKKNIFE_FORWARD_RECOVERY
                self.jackknife_stop_start_time = None
                self.log("jackknife stop done -> forward recovery")
            return

        if self.state == ParkingState.JACKKNIFE_FORWARD_RECOVERY:
            self.publish_command(0, self.forward_speed)
            if self.hitch_ready_for_state2():
                self.state = ParkingState.STATE2_COUNTER_TURN
                self.log("jackknife recovered to state2 angle -> state2")
            return

        if abs(self.current_hitch_angle) >= self.jackknife_limit_deg:
            now = time.time()
            if self.jackknife_over_limit_start_time is None:
                self.jackknife_over_limit_start_time = now
            if now - self.jackknife_over_limit_start_time >= self.jackknife_detection_duration_sec:
                self.state = ParkingState.JACKKNIFE_STOP
                self.jackknife_stop_start_time = now
                self.jackknife_over_limit_start_time = None
                self.publish_command(0, STOP_SPEED)
                self.log(f"jackknife limit -> 1s stop: hitch={self.current_hitch_angle:.1f}")
                return
            return
        self.jackknife_over_limit_start_time = None

        if self.state == ParkingState.WAIT_PATH:
            self.state = ParkingState.STATE1_INITIAL_TURN

        # 🌟 [핵심 개선 알고리즘]: 실시간 차량 진행 축과 주차장 축 간의 헤딩 오차각 산출
        if self.path_data is not None and len(self.path_data) >= 3:
            # 차량 축 벡터: 베지에 곡선 패스의 극초반 시점 방향 활용
            v_veh = (self.path_data[2][0] - self.path_data[0][0], self.path_data[0][1] - self.path_data[2][1])
            # 주차 목표지점 축 벡터: 베지에 곡선 패스의 종단 주차 공간 방향 활용
            v_park = (self.path_data[-1][0] - self.path_data[-2][0], self.path_data[-2][1] - self.path_data[-1][1])
            
            if math.hypot(*v_veh) > 1e-6 and math.hypot(*v_park) > 1e-6:
                self.current_heading_error = self.signed_angle_between(v_veh, v_park)

        # ----------------------------------------------------------------------
        # 상태 기계 분기 처리 (가변 조향 및 엄격한 축 동기화 기반 코드 분기)
        # ----------------------------------------------------------------------

        # [STATE 1]: 주차 영역 쪽으로 트레일러 뒷머리 꺾어 넣기 (가변 조향)
        if self.state == ParkingState.STATE1_INITIAL_TURN:
            if self.ready_for_state3_strict(self.current_heading_error):
                self.state = ParkingState.STATE3_NEUTRAL_REVERSE
                self.log("🎯 [축 정렬 일치] 사각형 축 완전 동기화 성공 -> state3")
                return

            # 오차각에 게인을 곱해 실시간 비례형(P) 조향 단계를 연산
            # 오차가 커질수록 크게 꺾고 가깝게 정렬될수록 핸들을 풀게 됩니다.
            calculated_steering = int(round(self.current_heading_error * self.k_angle_state1))
            steering = clamp(calculated_steering, -self.initial_turn_step, self.initial_turn_step)
            
            self.publish_command(int(steering), self.reverse_speed)

            # 트레일러가 목표 트리거 각만큼 꺾였다면 카운터 조향 단계로 이전
            if abs(self.current_hitch_angle) >= self.state1_hitch_trigger_deg:
                self.state = ParkingState.STATE2_COUNTER_TURN
                self.log(f"hitch trigger -> state2: hitch={self.current_hitch_angle:.1f}, error={self.current_heading_error:.1f}°")
            return

        # [STATE 2]: 트레일러 역조향을 풀고 일직선 축 정렬 유도 (가변 조향)
        if self.state == ParkingState.STATE2_COUNTER_TURN:
            if self.ready_for_state3_strict(self.current_heading_error):
                self.state = ParkingState.STATE3_NEUTRAL_REVERSE
                self.log("🎯 [축 정렬 일치] 사각형 축 완전 동기화 성공 -> state3")
                return

            if self.hitch_is_zero():
                self.state = ParkingState.STATE2_NEUTRAL_FORWARD
                self.neutral_forward_start_time = time.time()
                self.log("hitch zero -> neutral forward before retrying state1")
                return

            # 차량 진행각과 주차장 축이 평행해지도록 반대 방향 가변 조향 스텝 적용
            calculated_steering = int(round(-self.current_heading_error * self.k_angle_state2))
            steering = clamp(calculated_steering, -self.counter_turn_step, self.counter_turn_step)
            
            self.publish_command(int(steering), self.reverse_speed)
            return

        # [STATE 2_FORWARD]: 기존 원본 탈출/보정 전진 기하 알고리즘 계승
        if self.state == ParkingState.STATE2_NEUTRAL_FORWARD:
            self.publish_command(0, self.forward_speed)
            if self.neutral_forward_elapsed() >= self.neutral_forward_duration_sec:
                self.state = ParkingState.STATE1_INITIAL_TURN
                self.neutral_forward_start_time = None
                self.log("neutral forward done -> state1")
            return

        # [STATE 3]: 정렬 완료 상태에서 일직선 후진 진입
        if self.state == ParkingState.STATE3_NEUTRAL_REVERSE:
            self.publish_command(0, self.reverse_speed)
            if self.parking_currently_missing():
                self.state = ParkingState.STATE4_MISSING_CONFIRM
                self.missing_start_time = time.time()
                self.log("parking disappeared -> state4")
            return

        # [STATE 4]: 사각지대 발생 시 안전 추가 후진 완료 확인 (데드 레코닝 유지)
        if self.state == ParkingState.STATE4_MISSING_CONFIRM:
            self.publish_command(0, self.reverse_speed)
            if not self.parking_currently_missing():
                self.state = ParkingState.STATE3_NEUTRAL_REVERSE
                self.missing_start_time = None
                self.log("parking detected again -> state3")
                return

            elapsed = self.parking_missing_elapsed()
            if elapsed >= self.parking_missing_complete_sec:
                self.complete(f"parking disappeared for {elapsed:.1f}s")

    def parking_is_centered(self) -> bool:
        if self.last_parking_center_x is None:
            return False
        return abs(self.last_parking_center_x - IMG_CX) <= self.center_tolerance_px

    # 🌟 [새로 구현]: 사각형 주차 구역과 차량 축의 정렬 상태를 다각도로 평가하는 정밀 완결 검증 함수
    def ready_for_state3_strict(self, heading_error: float) -> bool:
        if self.last_parking_center_x is None or self.last_marker_midpoint_x is None:
            return False
        if self.last_alignment_seen_time is None:
            return False
        if time.time() - self.last_alignment_seen_time > self.parking_detection_timeout_sec:
            return False

        # 1. 위치 동기화 만족 여부 (주차 타깃 x축과 두 차량 마커의 중점이 이미지 중앙부 영역에 존재해야 함)
        parking_centered = abs(self.last_parking_center_x - IMG_CX) <= self.center_tolerance_px
        markers_centered = abs(self.last_marker_midpoint_x - IMG_CX) <= self.center_tolerance_px
        
        # 2. 각도 축 동기화 만족 여부 (틀어진 헤딩 각도 차이가 오차 허용 범위 이내인가?)
        axis_aligned = abs(heading_error) <= self.heading_tolerance_deg
        
        # 3. 종합 평가판정 (위치 일치 + 🌟각도 평행 일치 + 기존 트레일러 수평 조건 및 굴절 제로화 결합)
        return (
            parking_centered
            and markers_centered
            and axis_aligned
            and self.last_parking_is_horizontal
            and self.hitch_is_zero()
        )

    def parking_horizontal_alignment(
        self,
        contour: Optional[List[Point]],
    ) -> Tuple[bool, Optional[float]]:
        if contour is None or len(contour) < 3:
            return False, None

        points = np.array(contour, dtype=np.float32)
        centered = points - np.mean(points, axis=0)
        covariance = centered.T @ centered / max(1, len(points))
        eigvals, eigvecs = np.linalg.eigh(covariance)

        long_index = int(np.argmax(eigvals))
        short_index = 1 - long_index
        long_var = float(eigvals[long_index])
        short_var = float(eigvals[short_index])
        if short_var <= 1e-6:
            return False, None

        aspect_ratio = float(np.sqrt(long_var / short_var))
        if aspect_ratio < self.parking_orientation_min_aspect_ratio:
            return False, None

        long_axis = eigvecs[:, long_index]
        angle = abs(float(np.degrees(np.arctan2(long_axis[1], long_axis[0]))))
        angle = min(angle, 180.0 - angle)
        is_horizontal = angle <= self.parking_horizontal_tolerance_deg
        return is_horizontal, angle

    def hitch_is_zero(self) -> bool:
        return abs(self.current_hitch_angle) <= self.hitch_zero_tolerance_deg

    def hitch_ready_for_state2(self) -> bool:
        return abs(self.current_hitch_angle) <= self.jackknife_recovery_target_deg

    def neutral_forward_elapsed(self) -> float:
        if self.neutral_forward_start_time is None:
            return 0.0
        return time.time() - self.neutral_forward_start_time

    def jackknife_stop_elapsed(self) -> float:
        if self.jackknife_stop_start_time is None:
            return 0.0
        return time.time() - self.jackknife_stop_start_time

    def parking_currently_missing(self) -> bool:
        if self.last_parking_seen_time is None:
            return False
        return self.parking_missing_elapsed() > self.parking_detection_timeout_sec

    def parking_missing_elapsed(self) -> float:
        if self.last_parking_seen_time is None:
            return 0.0
        return time.time() - self.last_parking_seen_time

    def publish_command(self, steering: int, speed: int) -> None:
        target = int(round(clamp(steering * self.steering_sign, -MAX_STEP, MAX_STEP)))
        target = int(
            clamp(
                target,
                self.last_steering - self.max_step_delta,
                self.last_steering + self.max_step_delta,
            )
        )
        self.last_steering = target

        msg = MotionCommand()
        msg.steering = target
        msg.left_speed = int(speed)
        msg.right_speed = int(speed)
        self.publisher.publish(msg)

    def complete(self, reason: str) -> None:
        if self.state == ParkingState.DONE:
            return
        self.state = ParkingState.DONE
        self.publish_command(0, STOP_SPEED)
        self.get_logger().info(f"PARKING COMPLETE: {reason}")

    def log(self, text: str) -> None:
        if self.show_log:
            self.get_logger().info(
                f"[{self.state}] {text}, side={self.parking_side}, "
                f"hitch={self.current_hitch_angle:.1f}, "
                f"err_ang={self.current_heading_error:.1f}°, "
                f"parking_x={self.last_parking_center_x}"
            )

    def side_from_x(self, x: float) -> int:
        return 1 if x >= IMG_CX else -1

    def side_from_path(self, path: List[Point]) -> int:
        start_x = path[0][0]
        end_x = path[-1][0]
        return self.side_from_x(end_x if abs(end_x - IMG_CX) > 1.0 else start_x)

    def extract_perception(self, msg: DetectionArray) -> Dict[str, Dict[str, object]]:
        result: Dict[str, Dict[str, object]] = {}
        for detection in msg.detections:
            name = str(detection.class_name).strip().lower()
            if name not in (CLS_LEFT, CLS_RIGHT, CLS_PARKING):
                continue
            if float(detection.score) < self.minimum_score:
                continue

            center_x = float(detection.bbox.center.position.x)
            center_y = float(detection.bbox.center.position.y)
            contour = self.detection_to_contour(detection)
            area = self.contour_area(contour)
            prev = result.get(name)
            if prev is None or area > float(prev["area"]):
                result[name] = {
                    "center_x": center_x,
                    "center_y": center_y,
                    "contour": contour,
                    "area": area,
                }
        return result

    @staticmethod
    def detection_to_contour(detection) -> Optional[List[Point]]:
        if len(detection.mask.data) < 3:
            return None
        return [
            (float(point.x), float(point.y))
            for point in detection.mask.data
        ]

    @staticmethod
    def contour_area(contour: Optional[List[Point]]) -> float:
        if contour is None or len(contour) < 3:
            return 0.0
        area = 0.0
        for index, point in enumerate(contour):
            next_point = contour[(index + 1) % len(contour)]
            area += point[0] * next_point[1] - next_point[0] * point[1]
        return abs(area) * 0.5

    def markers_inside_parking(self, perception: Dict[str, Dict[str, object]]) -> bool:
        parking = perception.get(CLS_PARKING)
        left = perception.get(CLS_LEFT)
        right = perception.get(CLS_RIGHT)
        if parking is None or left is None or right is None:
            return False

        parking_contour = parking.get("contour")
        if not parking_contour:
            return False

        left_center = (float(left["center_x"]), float(left["center_y"]))
        right_center = (float(right["center_x"]), float(right["center_y"]))
        return (
            self.point_inside_polygon(left_center, parking_contour)
            and self.point_inside_polygon(right_center, parking_contour)
        )

    @staticmethod
    def point_inside_polygon(point: Point, polygon: List[Point]) -> bool:
        x, y = point
        inside = False
        j = len(polygon) - 1
        for i, pi in enumerate(polygon):
            pj = polygon[j]
            yi = pi[1]
            yj = pj[1]
            intersects = (yi > y) != (yj > y)
            if intersects:
                x_intersect = (pj[0] - pi[0]) * (y - yi) / (yj - yi + 1e-9) + pi[0]
                if x < x_intersect:
                    inside = not inside
            j = i
        return inside


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CurvedTrailerParkingMotionPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.publish_command(0, STOP_SPEED)
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
# """
# curved_trailer_parking_motion_planner_node.py

# State-machine motion planner for curved reverse parking with an articulated
# trailer vehicle. (Upgraded with Smart Heading-Distance Joint Variable Trigger Control)

# Inputs:
#   - detections (interfaces_pkg/DetectionArray): YOLO detections for left/right/parking
#   - path_planning_result (interfaces_pkg/PathPlanningResult): parking path points
#   - /articulation/angle (std_msgs/Float32): articulation angle from serial_sender

# Output:
#   - topic_control_signal (interfaces_pkg/MotionCommand): steering and motor command
# """

# import time
# import math
# from typing import Dict, List, Optional, Tuple

# import numpy as np
# import rclpy
# from rclpy.node import Node
# from rclpy.qos import (
#     QoSDurabilityPolicy,
#     QoSHistoryPolicy,
#     QoSProfile,
#     QoSReliabilityPolicy,
# )
# from std_msgs.msg import Float32

# from interfaces_pkg.msg import DetectionArray, MotionCommand, PathPlanningResult


# Point = Tuple[float, float]

# SUB_DETECTION_TOPIC_NAME = "detections"
# SUB_PATH_TOPIC_NAME = "path_planning_result"
# SUB_HITCH_TOPIC_NAME = "/articulation/angle"
# PUB_TOPIC_NAME = "topic_control_signal"

# CLS_LEFT = "left"
# CLS_RIGHT = "right"
# CLS_PARKING = "parking"

# IMG_W = 640.0
# IMG_CX = IMG_W / 2.0

# TIMER = 0.1
# MAX_STEP = 7
# REVERSE_SPEED = -40
# STOP_SPEED = 0


# class ParkingState:
#     WAIT_PATH = "WAIT_PATH"
#     STATE1_INITIAL_TURN = "STATE1_INITIAL_TURN"
#     STATE2_COUNTER_TURN = "STATE2_COUNTER_TURN"
#     STATE2_NEUTRAL_FORWARD = "STATE2_NEUTRAL_FORWARD"
#     STATE3_NEUTRAL_REVERSE = "STATE3_NEUTRAL_REVERSE"
#     STATE4_MISSING_CONFIRM = "STATE4_MISSING_CONFIRM"
#     JACKKNIFE_STOP = "JACKKNIFE_STOP"
#     JACKKNIFE_FORWARD_RECOVERY = "JACKKNIFE_FORWARD_RECOVERY"
#     DONE = "DONE"


# def clamp(value: float, low: float, high: float) -> float:
#     return max(low, min(high, value))


# class CurvedTrailerParkingMotionPlannerNode(Node):
#     def __init__(self) -> None:
#         super().__init__("curved_trailer_parking_motion_planner_node")

#         self.sub_detection_topic = self.declare_parameter("sub_detection_topic", SUB_DETECTION_TOPIC_NAME).value
#         self.sub_path_topic = self.declare_parameter("sub_lane_topic", SUB_PATH_TOPIC_NAME).value
#         self.sub_hitch_topic = self.declare_parameter("sub_hitch_topic", SUB_HITCH_TOPIC_NAME).value
#         self.pub_topic = self.declare_parameter("pub_topic", PUB_TOPIC_NAME).value
#         self.timer_period = float(self.declare_parameter("timer", TIMER).value)

#         # 제어 파라미터 튜닝
#         self.reverse_speed = int(self.declare_parameter("reverse_speed", REVERSE_SPEED).value)
#         self.forward_speed = int(self.declare_parameter("forward_speed", 25).value)

#         self.initial_turn_step = int(self.declare_parameter("initial_turn_step", MAX_STEP).value)
#         self.counter_turn_step = int(self.declare_parameter("counter_turn_step", 7).value)

#         # 가변 조향용 확장 비례 제어 게인값
#         self.k_angle_state1 = float(self.declare_parameter("k_angle_state1", 140.0).value)
#         self.k_angle_state2 = float(self.declare_parameter("k_angle_state2", 180.0).value)

#         # 허용 오차 임계 한계선 설정
#         self.heading_tolerance_deg = float(self.declare_parameter("heading_tolerance_deg", 8.0).value)
#         self.center_tolerance_px = float(self.declare_parameter("center_tolerance_px", 45.0).value)
#         self.hitch_zero_tolerance_deg = float(self.declare_parameter("hitch_zero_tolerance_deg", 4.0).value)

#         # 근거리 사선 진입 차단 임계 거리 및 각도
#         self.emergency_close_dist_px = float(self.declare_parameter("emergency_close_dist_px", 320.0).value)
#         self.emergency_large_angle_deg = float(self.declare_parameter("emergency_large_angle_deg", 35.0).value)

#         # 잭나이프 안전 복구 기준 한계선
#         self.jackknife_recovery_target_deg = float(self.declare_parameter("jackknife_recovery_target_deg", 24.0).value)

#         # 조향 방향 전체 반전 (-1)
#         self.steering_sign = int(self.declare_parameter("steering_sign", -1).value)
#         self.max_step_delta = int(self.declare_parameter("max_step_delta", 2).value)

#         self.parking_horizontal_tolerance_deg = float(self.declare_parameter("parking_horizontal_tolerance_deg", 20.0).value)
#         self.parking_orientation_min_aspect_ratio = float(self.declare_parameter("parking_orientation_min_aspect_ratio", 1.2).value)
#         self.path_straight_tolerance_px = float(self.declare_parameter("path_straight_tolerance_px", 30.0).value)
#         self.path_timeout_sec = float(self.declare_parameter("path_timeout_sec", 0.5).value)
#         self.parking_missing_complete_sec = float(self.declare_parameter("parking_missing_complete_sec", 2.0).value)
#         self.parking_detection_timeout_sec = float(self.declare_parameter("parking_detection_timeout_sec", 0.30).value)
#         self.jackknife_limit_deg = float(self.declare_parameter("jackknife_limit_deg", 40.0).value)
#         self.jackknife_stop_duration_sec = float(self.declare_parameter("jackknife_stop_duration_sec", 1.0).value)
#         self.jackknife_detection_duration_sec = float(self.declare_parameter("jackknife_detection_duration_sec", 0.2).value)
#         self.minimum_score = float(self.declare_parameter("minimum_score", 0.50).value)
#         self.show_log = bool(self.declare_parameter("show_log", True).value)

#         qos = QoSProfile(
#             reliability=QoSReliabilityPolicy.RELIABLE,
#             history=QoSHistoryPolicy.KEEP_LAST,
#             durability=QoSDurabilityPolicy.VOLATILE,
#             depth=1,
#         )

#         self.detection_sub = self.create_subscription(DetectionArray, self.sub_detection_topic, self.detection_callback, qos)
#         self.path_sub = self.create_subscription(PathPlanningResult, self.sub_path_topic, self.path_callback, qos)
#         self.hitch_sub = self.create_subscription(Float32, self.sub_hitch_topic, self.hitch_callback, qos)
#         self.publisher = self.create_publisher(MotionCommand, self.pub_topic, qos)

#         self.path_data: Optional[List[Point]] = None
#         self.last_path_seen_time: Optional[float] = None
#         self.last_path_max_deviation_px: Optional[float] = None
#         self.last_parking_center_x: Optional[float] = None
#         self.last_marker_midpoint_x: Optional[float] = None
#         self.last_alignment_seen_time: Optional[float] = None
#         self.last_parking_is_horizontal = False
#         self.last_parking_angle_deg: Optional[float] = None
#         self.last_parking_seen_time: Optional[float] = None
#         self.current_hitch_angle = 0.0

#         self.parking_side = 1
#         self.state = ParkingState.WAIT_PATH
#         self.last_steering = 0
#         self.missing_start_time: Optional[float] = None
        
#         # 🌟 [새로 추가] 전진 보정 상태 전용 독립 타임스탬프 변수
#         self.forward_start_time: Optional[float] = None
        
#         self.jackknife_stop_start_time: Optional[float] = None
#         self.jackknife_over_limit_start_time: Optional[float] = None

#         self.current_heading_error = 0.0
#         self.timer = self.create_timer(self.timer_period, self.timer_callback)
#         self.get_logger().info("curved_trailer_parking_motion_planner_node initialized.")

#     def detection_callback(self, msg: DetectionArray) -> None:
#         perception = self.extract_perception(msg)
#         now = time.time()
#         parking = perception.get(CLS_PARKING)
#         if parking is not None:
#             self.last_parking_center_x = parking["center_x"]
#             self.last_parking_seen_time = now
#             self.missing_start_time = None
#             self.parking_side = self.side_from_x(parking["center_x"])
#             self.last_parking_is_horizontal, self.last_parking_angle_deg = self.parking_horizontal_alignment(parking.get("contour"))

#         left = perception.get(CLS_LEFT)
#         right = perception.get(CLS_RIGHT)
#         if parking is not None and left is not None and right is not None:
#             self.last_marker_midpoint_x = (float(left["center_x"]) + float(right["center_x"])) / 2.0
#             self.last_alignment_seen_time = now

#         if self.state in (ParkingState.STATE3_NEUTRAL_REVERSE, ParkingState.STATE4_MISSING_CONFIRM):
#             if self.markers_inside_parking(perception):
#                 self.complete("left/right marker centers reached parking boundary")

#     def path_callback(self, msg: PathPlanningResult) -> None:
#         if len(msg.x_points) != len(msg.y_points) or len(msg.x_points) < 2:
#             return
#         self.path_data = list(zip(msg.x_points, msg.y_points))
#         self.last_path_seen_time = time.time()
#         self.last_path_max_deviation_px = self.path_max_deviation_px(self.path_data)
#         if self.last_parking_center_x is None:
#             self.parking_side = self.side_from_path(self.path_data)
#         if self.state == ParkingState.WAIT_PATH:
#             self.state = ParkingState.STATE1_INITIAL_TURN
#             self.log("path received -> state1")

#     def hitch_callback(self, msg: Float32) -> None:
#         self.current_hitch_angle = float(msg.data)

#     @staticmethod
#     def distance(point_a: Point, point_b: Point) -> float:
#         return math.hypot(point_b[0] - point_a[0], point_b[1] - point_a[1])

#     @staticmethod
#     def signed_angle_between(vector_a: Point, vector_b: Point) -> float:
#         cross = vector_a[0] * vector_b[1] - vector_a[1] * vector_b[0]
#         dot = vector_a[0] * vector_b[0] + vector_a[1] * vector_b[1]
#         return math.degrees(math.atan2(cross, dot))

#     def timer_callback(self) -> None:
#         if self.state == ParkingState.DONE:
#             self.publish_command(0, STOP_SPEED)
#             return

#         if self.path_data is None and self.state == ParkingState.WAIT_PATH:
#             self.publish_command(0, STOP_SPEED)
#             return

#         # 잭나이프 예외 안전 제어 루틴
#         if self.state == ParkingState.JACKKNIFE_STOP:
#             self.publish_command(0, STOP_SPEED)
#             if self.jackknife_stop_elapsed() >= self.jackknife_stop_duration_sec:
#                 self.state = ParkingState.JACKKNIFE_FORWARD_RECOVERY
#                 self.jackknife_stop_start_time = None
#             return

#         if self.state == ParkingState.JACKKNIFE_FORWARD_RECOVERY:
#             self.publish_command(0, self.forward_speed)
#             if self.hitch_ready_for_state2():
#                 self.state = ParkingState.STATE2_COUNTER_TURN
#             return

#         if abs(self.current_hitch_angle) >= self.jackknife_limit_deg:
#             now = time.time()
#             if self.jackknife_over_limit_start_time is None:
#                 self.jackknife_over_limit_start_time = now
#             if now - self.jackknife_over_limit_start_time >= self.jackknife_detection_duration_sec:
#                 self.state = ParkingState.JACKKNIFE_STOP
#                 self.jackknife_stop_start_time = now
#                 self.jackknife_over_limit_start_time = None
#                 self.publish_command(0, STOP_SPEED)
#                 return
#             return
#         self.jackknife_over_limit_start_time = None

#         # 기하학 오차 벡터 연산 부
#         heading_error = 0.0
#         remaining_distance = 500.0

#         if self.path_data is not None and len(self.path_data) >= 3:
#             v_veh = (self.path_data[2][0] - self.path_data[0][0], self.path_data[0][1] - self.path_data[2][1])
#             v_park = (self.path_data[-1][0] - self.path_data[-2][0], self.path_data[-2][1] - self.path_data[-1][1])
#             if math.hypot(*v_veh) > 1e-6 and math.hypot(*v_park) > 1e-6:
#                 heading_error = self.signed_angle_between(v_veh, v_park)
#                 self.current_heading_error = heading_error
#             remaining_distance = self.distance(self.path_data[0], self.path_data[-1])

#         distance_denominator = max(40.0, remaining_distance)

#         # 거리와 '축 오차 각도'를 결합하고, 원거리 감도를 최적화(0.015)한 가변 트리거 식
#         error_factor = clamp(abs(heading_error) / 45.0, 0.0, 1.0)
#         dynamic_trigger_deg = clamp(5.0 + (remaining_distance * 0.015 * error_factor), 5.0, 18.0)

#         # 긴급 전진 탈출 가로채기
#         if self.state in (ParkingState.STATE1_INITIAL_TURN, ParkingState.STATE2_COUNTER_TURN):
#             if remaining_distance <= self.emergency_close_dist_px and abs(heading_error) >= self.emergency_large_angle_deg:
#                 self.state = ParkingState.STATE2_NEUTRAL_FORWARD
#                 self.forward_start_time = time.time()  # 🌟 전진 보정 타이머 트리거 선언
#                 self.get_logger().warn(f"🚨 [EMERGENCY ESCAPE] 주차 불가 돌입! Dist: {remaining_distance:.1f}px, Ang: {heading_error:.1f}° -> 전진 보정!")
#                 return

#         # ----------------------------------------------------------------------
#         # 상태 기계 분기 처리 영역
#         # ----------------------------------------------------------------------

#         # [STATE 1]: 초기 진입 단계
#         if self.state == ParkingState.STATE1_INITIAL_TURN:
#             if self.ready_for_state3_strict(heading_error):
#                 self.state = ParkingState.STATE3_NEUTRAL_REVERSE
#                 self.log("🎯 [정렬 순항] 축과 중심선 일치 확인 -> STATE3로 직행 안착")
#                 return

#             calculated_steering = int(round((heading_error / distance_denominator) * self.k_angle_state1))
#             steering = clamp(calculated_steering, -self.initial_turn_step, self.initial_turn_step)
#             self.publish_command(int(steering), self.reverse_speed)

#             if abs(self.current_hitch_angle) >= dynamic_trigger_deg:
#                 self.state = ParkingState.STATE2_COUNTER_TURN
#                 self.log(f"hitch trigger -> state2: dist={remaining_distance:.1f}px, trigger={dynamic_trigger_deg:.1f}°, error={heading_error:.1f}°")
#             return

#         # [STATE 2]: 카운터 조향 및 축 평행 정렬 단계
#         if self.state == ParkingState.STATE2_COUNTER_TURN:
#             if self.ready_for_state3_strict(heading_error):
#                 self.state = ParkingState.STATE3_NEUTRAL_REVERSE
#                 self.log("🎯 [축 정렬 동기화] 직사각형 축 평행 일치 완료 -> state3")
#                 return

#             if self.hitch_is_zero():
#                 self.state = ParkingState.STATE2_NEUTRAL_FORWARD
#                 self.forward_start_time = time.time()  # 🌟 전진 보정 타이머 트리거 선언
#                 self.log("hitch zero -> neutral forward (Timer Started)")
#                 return

#             calculated_steering = int(round((-heading_error / distance_denominator) * self.k_angle_state2))
#             steering = clamp(calculated_steering, -self.counter_turn_step, self.counter_turn_step)
#             self.publish_command(int(steering), self.reverse_speed)
#             return

#         # 🌟 [완벽 수정] [STATE 2_FORWARD]: 타 노드 변수 간섭 차단 전용 타이머 제어
#         if self.state == ParkingState.STATE2_NEUTRAL_FORWARD:
            
#             # 카메라 콜백에서 리셋되지 않는 독립 변수 사용
#             if self.forward_start_time is None:
#                 self.forward_start_time = time.time()
            
#             forward_elapsed = time.time() - self.forward_start_time

#             # 1.5초 타이머가 정확히 누적 측정됩니다.
#             if forward_elapsed >= 2.0:
#                 self.state = ParkingState.STATE1_INITIAL_TURN
#                 self.forward_start_time = None  # 타이머 리셋
#                 self.log(f"🎯 [구출 전진 종료] 1.5초 타임아웃 완료 -> STATE1 후진 재진입")
#                 return
            
#             self.publish_command(0, self.forward_speed)
#             return

#         # [STATE 3]: 직선 후진 안착 단계
#         if self.state == ParkingState.STATE3_NEUTRAL_REVERSE:
#             self.publish_command(0, self.reverse_speed)
#             if self.parking_currently_missing():
#                 self.state = ParkingState.STATE4_MISSING_CONFIRM
#                 self.missing_start_time = time.time()
#                 self.log("parking disappeared -> state4")
#             return

#         # [STATE 4]: 카메라 사각지대 관성 타이머 후진 단계
#         if self.state == ParkingState.STATE4_MISSING_CONFIRM:
#             self.publish_command(0, self.reverse_speed)
#             if not self.parking_currently_missing():
#                 self.state = ParkingState.STATE3_NEUTRAL_REVERSE
#                 self.missing_start_time = None
#                 self.log("parking detected again -> state3")
#                 return
#             elapsed = self.parking_missing_elapsed()
#             if elapsed >= self.parking_missing_complete_sec:
#                 self.complete(f"parking disappeared for {elapsed:.1f}s")

#     def parking_is_centered(self) -> bool:
#         if self.last_parking_center_x is None:
#             return False
#         return abs(self.last_parking_center_x - IMG_CX) <= self.center_tolerance_px

#     def ready_for_state3_strict(self, heading_error: float) -> bool:
#         if self.last_parking_center_x is None or self.last_marker_midpoint_x is None:
#             return False
#         if self.last_alignment_seen_time is None:
#             return False
#         if time.time() - self.last_alignment_seen_time > self.parking_detection_timeout_sec:
#             return False
#         markers_centered = abs(self.last_marker_midpoint_x - IMG_CX) <= self.center_tolerance_px
#         axis_aligned = abs(heading_error) <= self.heading_tolerance_deg
#         return markers_centered and axis_aligned and self.hitch_is_zero()

#     def parking_horizontal_alignment(self, contour: Optional[List[Point]]) -> Tuple[bool, Optional[float]]:
#         if contour is None or len(contour) < 3:
#             return False, None
#         points = np.array(contour, dtype=np.float32)
#         centered = points - np.mean(points, axis=0)
#         covariance = centered.T @ centered / max(1, len(points))
#         eigvals, eigvecs = np.linalg.eigh(covariance)
#         long_index = int(np.argmax(eigvals))
#         short_index = 1 - long_index
#         long_var = float(eigvals[long_index])
#         short_var = float(eigvals[short_index])
#         if short_var <= 1e-6:
#             return False, None
#         aspect_ratio = float(np.sqrt(long_var / short_var))
#         if aspect_ratio < self.parking_orientation_min_aspect_ratio:
#             return False, None
#         long_axis = eigvecs[:, long_index]
#         angle = abs(float(np.degrees(np.arctan2(long_axis[1], long_axis[0]))))
#         angle = min(angle, 180.0 - angle)
#         is_horizontal = angle <= self.parking_horizontal_tolerance_deg
#         return is_horizontal, angle

#     def hitch_is_zero(self) -> bool:
#         return abs(self.current_hitch_angle) <= self.hitch_zero_tolerance_deg

#     def hitch_ready_for_state2(self) -> bool:
#         return abs(self.current_hitch_angle) <= self.jackknife_recovery_target_deg

#     def path_is_straight_enough(self) -> bool:
#         if not self.path_is_recent() or self.last_path_max_deviation_px is None:
#             return False
#         return self.last_path_max_deviation_px <= self.path_straight_tolerance_px

#     def path_is_recent(self) -> bool:
#         if self.last_path_seen_time is None:
#             return False
#         return time.time() - self.last_path_seen_time <= self.path_timeout_sec

#     @staticmethod
#     def path_max_deviation_px(path: List[Point]) -> Optional[float]:
#         if len(path) < 3:
#             return None
#         points = np.array(path, dtype=np.float32)
#         start = points[0]
#         end = points[-1]
#         line = end - start
#         line_len = float(np.linalg.norm(line))
#         if line_len <= 1e-6:
#             return None
#         offsets = points[1:-1] - start
#         deviations = np.abs(line[0] * offsets[:, 1] - line[1] * offsets[:, 0]) / line_len
#         return float(np.max(deviations)) if len(deviations) > 0 else 0.0

#     def jackknife_stop_elapsed(self) -> float:
#         if self.jackknife_stop_start_time is None:
#             return 0.0
#         return time.time() - self.jackknife_stop_start_time

#     def parking_currently_missing(self) -> bool:
#         if self.last_parking_seen_time is None:
#             return False
#         return self.parking_missing_elapsed() > self.parking_detection_timeout_sec

#     def parking_missing_elapsed(self) -> float:
#         if self.last_parking_seen_time is None:
#             return 0.0
#         return time.time() - self.last_parking_seen_time

#     def publish_command(self, steering: int, speed: int) -> None:
#         target = int(round(clamp(steering * self.steering_sign, -MAX_STEP, MAX_STEP)))
#         target = int(clamp(target, self.last_steering - self.max_step_delta, self.last_steering + self.max_step_delta))
#         self.last_steering = target

#         msg = MotionCommand()
#         msg.steering = target
#         msg.left_speed = int(speed)
#         msg.right_speed = int(speed)
#         self.publisher.publish(msg)

#     def complete(self, reason: str) -> None:
#         if self.state == ParkingState.DONE:
#             return
#         self.state = ParkingState.DONE
#         self.publish_command(0, STOP_SPEED)
#         self.get_logger().info(f"PARKING COMPLETE: {reason}")

#     def log(self, text: str) -> None:
#         if self.show_log:
#             self.get_logger().info(
#                 f"[{self.state}] {text}, hitch={self.current_hitch_angle:.1f}, "
#                 f"err_ang={self.current_heading_error:.1f}°, parking_x={self.last_parking_center_x}"
#             )

#     def side_from_x(self, x: float) -> int:
#         return 1 if x >= IMG_CX else -1

#     def side_from_path(self, path: List[Point]) -> int:
#         start_x = path[0][0]
#         end_x = path[-1][0]
#         return self.side_from_x(end_x if abs(end_x - IMG_CX) > 1.0 else start_x)

#     def extract_perception(self, msg: DetectionArray) -> Dict[str, Dict[str, object]]:
#         result: Dict[str, Dict[str, object]] = {}
#         for detection in msg.detections:
#             name = str(detection.class_name).strip().lower()
#             if name not in (CLS_LEFT, CLS_RIGHT, CLS_PARKING):
#                 continue
#             if float(detection.score) < self.minimum_score:
#                 continue
#             center_x = float(detection.bbox.center.position.x)
#             center_y = float(detection.bbox.center.position.y)
#             contour = self.detection_to_contour(detection)
#             area = self.contour_area(contour)
#             prev = result.get(name)
#             if prev is None or area > float(prev["area"]):
#                 result[name] = {
#                     "center_x": center_x,
#                     "center_y": center_y,
#                     "contour": contour,
#                     "area": area,
#                 }
#         return result

#     @staticmethod
#     def detection_to_contour(detection) -> Optional[List[Point]]:
#         if len(detection.mask.data) < 3:
#             return None
#         return [(float(point.x), float(point.y)) for point in detection.mask.data]

#     @staticmethod
#     def contour_area(contour: Optional[List[Point]]) -> float:
#         if contour is None or len(contour) < 3:
#             return 0.0
#         area = 0.0
#         for index, point in enumerate(contour):
#             next_point = contour[(index + 1) % len(contour)]
#             area += point[0] * next_point[1] - next_point[0] * point[1]
#         return abs(area) * 0.5

#     def markers_inside_parking(self, perception: Dict[str, Dict[str, object]]) -> bool:
#         parking = perception.get(CLS_PARKING)
#         left = perception.get(CLS_LEFT)
#         right = perception.get(CLS_RIGHT)
#         if parking is None or left is None or right is None:
#             return False
#         parking_contour = parking.get("contour")
#         if not parking_contour:
#             return False
#         left_center = (float(left["center_x"]), float(left["center_y"]))
#         right_center = (float(right["center_x"]), float(right["center_y"]))
#         return self.point_inside_polygon(left_center, parking_contour) and self.point_inside_polygon(right_center, parking_contour)

#     @staticmethod
#     def point_inside_polygon(point: Point, polygon: List[Point]) -> bool:
#         x, y = point
#         inside = False
#         j = len(polygon) - 1
#         for i, pi in enumerate(polygon):
#             pj = polygon[j]
#             yi = pi[1]
#             yj = pj[1]
#             intersects = (yi > y) != (yj > y)
#             if intersects:
#                 x_intersect = (pj[0] - pi[0]) * (y - yi) / (yj - yi + 1e-9) + pi[0]
#                 if x < x_intersect:
#                     inside = not inside
#             j = i
#         return inside


# def main(args=None) -> None:
#     rclpy.init(args=args)
#     node = CurvedTrailerParkingMotionPlannerNode()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         try:
#             node.publish_command(0, STOP_SPEED)
#         except Exception:
#             pass
#         node.destroy_node()
#         if rclpy.ok():
#             rclpy.shutdown()


# if __name__ == "__main__":
#     main()