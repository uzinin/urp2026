#!/usr/bin/env python3
"""
curved_trailer_parking_motion_planner_node.py

State-machine motion planner for curved reverse parking with an articulated
trailer vehicle. (Upgraded with variable proportional steering & axis alignment checks)
+ Float32 기반의 가벼운 ArUco 평행 검증 로직 적용 (타임아웃 제거)
+ STATE2 직진 시 카메라 중심 정렬을 위한 P 제어 및 데드존 적용
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
    STATE1_STOP_BEFORE_STATE2 = "STATE1_STOP_BEFORE_STATE2"
    STATE2_COUNTER_TURN = "STATE2_COUNTER_TURN"
    STATE2_NEUTRAL_FORWARD = "STATE2_NEUTRAL_FORWARD"
    STATE3_1_HITCH_ZERO_FORWARD = "STATE3_1_HITCH_ZERO_FORWARD"
    STATE3_2_NEUTRAL_REVERSE = "STATE3_2_NEUTRAL_REVERSE"
    STATE4_MISSING_CONFIRM = "STATE4_MISSING_CONFIRM"
    DEBUG_HITCH_STOP = "DEBUG_HITCH_STOP"
    JACKKNIFE_STOP = "JACKKNIFE_STOP"
    JACKKNIFE_FORWARD_RECOVERY = "JACKKNIFE_FORWARD_RECOVERY"
    DONE = "DONE"


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class CurvedTrailerParkingMotionPlannerNode(Node):
    def __init__(self) -> None:
        super().__init__("curved_trailer_parking_motion_planner_node")

        self.sub_detection_topic = self.declare_parameter("sub_detection_topic", SUB_DETECTION_TOPIC_NAME).value
        self.sub_path_topic = self.declare_parameter("sub_lane_topic", SUB_PATH_TOPIC_NAME).value
        self.sub_hitch_topic = self.declare_parameter("sub_hitch_topic", SUB_HITCH_TOPIC_NAME).value
        # ArUco Yaw 토픽 설정
        self.sub_aruco_topic = self.declare_parameter("sub_aruco_topic", "aruco_yaw").value
        self.pub_topic = self.declare_parameter("pub_topic", PUB_TOPIC_NAME).value
        self.timer_period = float(self.declare_parameter("timer", TIMER).value)

        # ========================= TUNING PARAMETERS =========================
        self.reverse_speed = int(self.declare_parameter("reverse_speed", REVERSE_SPEED).value)
        self.forward_speed = int(self.declare_parameter("forward_speed", abs(60)).value)

        self.initial_turn_step = int(self.declare_parameter("initial_turn_step", MAX_STEP).value)
        self.counter_turn_step = int(self.declare_parameter("counter_turn_step", 7).value)

        self.k_parking_x_state1 = float(self.declare_parameter("k_parking_x_state1", 1.0).value)
        self.state2_fixed_steering = int(self.declare_parameter("state2_fixed_steering", 7).value)
        self.state3_k_lateral = float(self.declare_parameter("state3_k_lateral", 0.5).value)
        self.state3_k_hitch = float(self.declare_parameter("state3_k_hitch", 0.6).value)
        self.state3_alpha = float(self.declare_parameter("state3_alpha", 0.3).value)
        self.state3_gamma_limit_deg = float(self.declare_parameter("state3_gamma_limit_deg", 15.0).value)
        self.state3_target_angle_deadzone_deg = float(self.declare_parameter("state3_target_angle_deadzone_deg", 1.5).value)
        self.state3_lookahead_index = int(self.declare_parameter("state3_lookahead_index", 15).value)
        self.state3_max_step_delta = int(self.declare_parameter("state3_max_step_delta", 1).value)

        self.heading_tolerance_deg = float(self.declare_parameter("heading_tolerance_deg", 6.0).value)
        
        # ArUco 마커 평행 오차 허용 각도 (타임아웃 파라미터 삭제됨)
        self.aruco_parallel_tolerance_deg = float(self.declare_parameter("aruco_parallel_tolerance_deg", 4.0).value)

        self.steering_sign = int(self.declare_parameter("steering_sign", 1).value)
        self.max_step_delta = int(self.declare_parameter("max_step_delta", 2).value)

        self.state1_jackknife_margin_deg = float(self.declare_parameter("state1_jackknife_margin_deg", 5.0).value)
        self.state1_min_target_hitch_deg = float(self.declare_parameter("state1_min_target_hitch_deg", 20.0).value)
        self.center_tolerance_px = float(self.declare_parameter("center_tolerance_px", 35.0).value)
        self.parking_horizontal_tolerance_deg = float(self.declare_parameter("parking_horizontal_tolerance_deg", 20.0).value)
        self.parking_orientation_min_aspect_ratio = float(self.declare_parameter("parking_orientation_min_aspect_ratio", 1.2).value)
        self.hitch_zero_tolerance_deg = float(self.declare_parameter("hitch_zero_tolerance_deg", 11.0).value)
        self.neutral_forward_duration_sec = float(self.declare_parameter("neutral_forward_duration_sec", 4.0).value)
        self.state2_neutral_forward_target_parking_y = float(self.declare_parameter("state2_neutral_forward_target_parking_y", 120.0).value)
        self.state3_forward_target_parking_y = float(self.declare_parameter("state3_forward_target_parking_y", 110.0).value)
        self.state1_to_state2_stop_duration_sec = float(self.declare_parameter("state1_to_state2_stop_duration_sec", 1.0).value)
        self.parking_missing_complete_sec = float(self.declare_parameter("parking_missing_complete_sec", 5.0).value)
        self.parking_detection_timeout_sec = float(self.declare_parameter("parking_detection_timeout_sec", 0.30).value)
        self.jackknife_limit_deg = float(self.declare_parameter("jackknife_limit_deg", 40.0).value)
        self.debug_hitch_stop_deg = float(self.declare_parameter("debug_hitch_stop_deg", 60.0).value)
        self.state1_hitch_trigger_deg = clamp(
            self.jackknife_limit_deg - self.state1_jackknife_margin_deg, 0.0, self.jackknife_limit_deg
        )
        self.jackknife_forward_duration_sec = float(self.declare_parameter("jackknife_forward_duration_sec", 5.0).value)
        self.jackknife_detection_duration_sec = float(self.declare_parameter("jackknife_detection_duration_sec", 0.2).value)
        self.jackknife_recovery_target_deg = float(self.declare_parameter("jackknife_recovery_target_deg", self.state1_hitch_trigger_deg).value)
        self.minimum_score = float(self.declare_parameter("minimum_score", 0.50).value)
        self.show_log = bool(self.declare_parameter("show_log", True).value)

        # State2 직진 시 카메라 중앙 정렬을 위한 P 제어 및 데드존 설정
        self.state2_forward_kp = float(self.declare_parameter("state2_forward_kp", 0.05).value)
        self.state2_forward_deadzone_px = float(self.declare_parameter("state2_forward_deadzone_px", 15.0).value)
        # =====================================================================

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )

        self.detection_sub = self.create_subscription(DetectionArray, self.sub_detection_topic, self.detection_callback, qos)
        self.path_sub = self.create_subscription(PathPlanningResult, self.sub_path_topic, self.path_callback, qos)
        self.hitch_sub = self.create_subscription(Float32, self.sub_hitch_topic, self.hitch_callback, qos)
        # Float32 타입으로 ArUco 토픽 구독
        self.aruco_sub = self.create_subscription(Float32, self.sub_aruco_topic, self.aruco_callback, qos)
        
        self.publisher = self.create_publisher(MotionCommand, self.pub_topic, qos)

        self.path_data: Optional[List[Point]] = None
        self.last_parking_center_x: Optional[float] = None
        self.last_parking_center_y: Optional[float] = None
        self.last_marker_midpoint_x: Optional[float] = None
        self.last_alignment_seen_time: Optional[float] = None
        self.last_parking_is_horizontal = False
        self.last_parking_angle_deg: Optional[float] = None
        self.last_parking_seen_time: Optional[float] = None
        self.current_hitch_angle = 0.0
        
        # 최신 ArUco Yaw 각도
        self.last_aruco_yaw: Optional[float] = None

        self.parking_side = 1
        self.state = ParkingState.WAIT_PATH
        self.last_steering = 0
        self.last_command_signature: Optional[Tuple[int, int]] = None
        self.last_log_signature: Optional[Tuple[str, str]] = None
        self.missing_start_time: Optional[float] = None
        self.neutral_forward_start_time: Optional[float] = None
        self.state1_to_state2_stop_start_time: Optional[float] = None
        self.jackknife_stop_start_time: Optional[float] = None
        self.jackknife_forward_start_time: Optional[float] = None
        self.jackknife_over_limit_start_time: Optional[float] = None

        self.current_heading_error = 0.0
        self.state1_target_hitch_angle_deg: Optional[float] = None
        self.state1_target_steering = 0
        self.state1_entry_parking_x_error = 0.0
        self.state3_target_slope_f = 0.0
        self.state3_prev_step = 0

        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        self.get_logger().info("Curved Planner started with Lightweight ArUco Yaw Control")

    def rclpy_spin_node(self) -> None:
        pass

    def aruco_callback(self, msg: Float32) -> None:
        """가볍고 심플하게 Float32 Yaw 각도 수신"""
        self.last_aruco_yaw = float(msg.data)

    def detection_callback(self, msg: DetectionArray) -> None:
        perception = self.extract_perception(msg)
        now = time.time()

        parking = perception.get(CLS_PARKING)
        if parking is not None:
            self.last_parking_center_x = parking["center_x"]
            self.last_parking_center_y = parking["center_y"]
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
            ParkingState.STATE3_2_NEUTRAL_REVERSE,
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
            self.enter_state1("path received -> state1")

    def hitch_callback(self, msg: Float32) -> None:
        self.current_hitch_angle = float(msg.data)

    @staticmethod
    def distance(point_a: Point, point_b: Point) -> float:
        return math.hypot(point_b[0] - point_a[0], point_b[1] - point_a[1])

    @staticmethod
    def signed_angle_between(vector_a: Point, vector_b: Point) -> float:
        cross = vector_a[0] * vector_b[1] - vector_a[1] * vector_b[0]
        dot = vector_a[0] * vector_b[0] + vector_a[1] * vector_b[1]
        return math.degrees(math.atan2(cross, dot))

    def enter_state1(self, reason: str) -> None:
        parking_x_error = 0.0
        if self.last_parking_center_x is not None:
            parking_x_error = self.last_parking_center_x - IMG_CX

        state1_map_left_x = 250.0
        state1_map_right_x = 390.0
        parking_x_for_target = (
            IMG_CX if self.last_parking_center_x is None else self.last_parking_center_x
        )
        normalized_x = (
            (parking_x_for_target - state1_map_left_x)
            / max(state1_map_right_x - state1_map_left_x, 1e-6)
            * 2.0
            - 1.0
        )
        target_hitch_angle = clamp(
            normalized_x * self.state1_hitch_trigger_deg,
            -self.state1_hitch_trigger_deg,
            self.state1_hitch_trigger_deg,
        )
        min_target_hitch_angle = clamp(
            self.state1_min_target_hitch_deg,
            0.0,
            self.state1_hitch_trigger_deg,
        )
        if abs(target_hitch_angle) < min_target_hitch_angle:
            target_sign = 1.0 if normalized_x > 0.0 else -1.0 if normalized_x < 0.0 else float(self.parking_side)
            target_hitch_angle = target_sign * min_target_hitch_angle
        target_steering = int(
            round(
                target_hitch_angle
                / max(self.state1_hitch_trigger_deg, 1e-6)
                * self.initial_turn_step
                * self.k_parking_x_state1
            )
        )
        if 0 < abs(target_steering) < 3:
            target_steering = 3 if target_steering > 0 else -3

        self.state = ParkingState.STATE1_INITIAL_TURN
        self.state1_entry_parking_x_error = parking_x_error
        self.state1_target_hitch_angle_deg = target_hitch_angle
        self.state1_target_steering = int(
            clamp(target_steering, -self.initial_turn_step, self.initial_turn_step)
        )
        self.log(
            f"{reason}: state1 target hitch={self.state1_target_hitch_angle_deg:.1f}°, "
            f"x_err={self.state1_entry_parking_x_error:.1f}px, "
            f"steering={self.state1_target_steering}"
        )

    def state1_target_reached(self) -> bool:
        if self.state1_target_hitch_angle_deg is None:
            return False
        if abs(self.state1_target_hitch_angle_deg) <= self.hitch_zero_tolerance_deg:
            return True
        if self.state1_target_hitch_angle_deg > 0.0:
            return self.current_hitch_angle >= self.state1_target_hitch_angle_deg
        return self.current_hitch_angle <= self.state1_target_hitch_angle_deg

    def enter_state3(self, reason: str) -> None:
        self.state = ParkingState.STATE3_1_HITCH_ZERO_FORWARD
        self.log(reason)

    def enter_state3_reverse(self, reason: str) -> None:
        self.state = ParkingState.STATE3_2_NEUTRAL_REVERSE
        self.state3_target_slope_f = 0.0
        self.state3_prev_step = 0
        self.log(reason)

    def timer_callback(self) -> None:
        if self.state == ParkingState.DONE:
            self.publish_command(0, STOP_SPEED)
            return

        if self.state == ParkingState.DEBUG_HITCH_STOP:
            self.publish_command(0, STOP_SPEED)
            return

        if self.path_data is None and self.state == ParkingState.WAIT_PATH:
            self.publish_command(0, STOP_SPEED)
            self.log("waiting for first path")
            return

        if self.state == ParkingState.JACKKNIFE_STOP:
            self.publish_command(0, STOP_SPEED)
            if self.jackknife_stop_elapsed() >= self.jackknife_forward_duration_sec:
                self.state = ParkingState.JACKKNIFE_FORWARD_RECOVERY
                self.jackknife_stop_start_time = None
                self.log("jackknife stop done -> forward recovery")
            return

        if self.state == ParkingState.JACKKNIFE_FORWARD_RECOVERY:
            self.publish_command(0, self.forward_speed)
            if self.jackknife_forward_elapsed() >= self.jackknife_forward_duration_sec:
                self.jackknife_forward_start_time = None
                self.enter_state1("jackknife forward 5s done -> state1")
            return

        if abs(self.current_hitch_angle) >= self.debug_hitch_stop_deg:
            self.state = ParkingState.DEBUG_HITCH_STOP
            self.publish_command(0, STOP_SPEED)
            self.log(
                f"debug hitch stop: hitch={self.current_hitch_angle:.1f}, "
                f"limit={self.debug_hitch_stop_deg:.1f}"
            )
            return

        if abs(self.current_hitch_angle) >= self.jackknife_limit_deg:
            now = time.time()
            if self.jackknife_over_limit_start_time is None:
                self.jackknife_over_limit_start_time = now
            if now - self.jackknife_over_limit_start_time >= self.jackknife_detection_duration_sec:
                self.state = ParkingState.JACKKNIFE_FORWARD_RECOVERY
                self.jackknife_forward_start_time = now
                self.jackknife_over_limit_start_time = None
                self.publish_command(0, self.forward_speed)
                self.log(
                    f"jackknife limit -> forward 5s: hitch={self.current_hitch_angle:.1f}"
                )
                return
            return
        self.jackknife_over_limit_start_time = None

        if self.state == ParkingState.WAIT_PATH:
            self.enter_state1("wait path fallback -> state1")

        if self.path_data is not None and len(self.path_data) >= 3:
            v_veh = (self.path_data[2][0] - self.path_data[0][0], self.path_data[0][1] - self.path_data[2][1])
            v_park = (self.path_data[-1][0] - self.path_data[-2][0], self.path_data[-2][1] - self.path_data[-1][1])
            
            if math.hypot(*v_veh) > 1e-6 and math.hypot(*v_park) > 1e-6:
                self.current_heading_error = self.signed_angle_between(v_veh, v_park)

        # [STATE 1]
        if self.state == ParkingState.STATE1_INITIAL_TURN:
            if self.state1_target_hitch_angle_deg is None:
                self.enter_state1("state1 target missing -> recalc once")

            if self.ready_for_state3_strict(self.current_heading_error):
                self.enter_state3("🎯 [축 정렬 일치] YOLO 및 ArUco 평행 확인 성공 -> state3")
                return

            self.publish_command(self.state1_target_steering, self.reverse_speed)

            if self.state1_target_reached():
                self.state = ParkingState.STATE1_STOP_BEFORE_STATE2
                self.state1_to_state2_stop_start_time = time.time()
                self.log(
                    f"hitch target reached -> stop before state2: "
                    f"target={self.state1_target_hitch_angle_deg:.1f}°"
                )
            return

        # [STATE 1_STOP]
        if self.state == ParkingState.STATE1_STOP_BEFORE_STATE2:
            self.publish_command(0, STOP_SPEED)
            if self.state1_to_state2_stop_elapsed() >= self.state1_to_state2_stop_duration_sec:
                self.state = ParkingState.STATE2_COUNTER_TURN
                self.state1_to_state2_stop_start_time = None
                self.log("state1 stop done -> state2")
            return

        # [STATE 2]
        if self.state == ParkingState.STATE2_COUNTER_TURN:
            if self.ready_for_state3_strict(self.current_heading_error):
                self.enter_state3("🎯 [축 정렬 일치] YOLO 및 ArUco 평행 확인 성공 -> state3")
                return

            if self.hitch_is_zero():
                self.state = ParkingState.STATE2_NEUTRAL_FORWARD
                self.neutral_forward_start_time = time.time()
                self.log("hitch zero -> neutral forward before retrying state1")
                return

            steering = -self.state1_target_steering
            if 0 < abs(steering) < 3:
                steering = 3 if steering > 0 else -3
            steering = clamp(steering, -self.counter_turn_step, self.counter_turn_step)
            
            self.publish_command(int(steering), self.reverse_speed)
            return

        # [STATE 2_FORWARD] - 카메라 기준 목표점 정렬을 위한 P 제어 및 데드존 반영
        if self.state == ParkingState.STATE2_NEUTRAL_FORWARD:
            steering = 0
            
            if self.last_parking_center_x is not None:
                error_x = self.last_parking_center_x - IMG_CX
                
                # 데드존(허용 오차 범위) 이내이면 조향 0 유지, 벗어나면 비례 제어 개입
                if abs(error_x) <= self.state2_forward_deadzone_px:
                    steering = 0
                else:
                    raw_steering = error_x * self.state2_forward_kp
                    steering = int(clamp(round(raw_steering), -MAX_STEP, MAX_STEP))
                
            self.publish_command(steering, self.forward_speed)
            
            if self.parking_y_reached_state1_retry_line():
                self.neutral_forward_start_time = None
                self.enter_state1("parking_y <= retry line -> state1")
            return

        # [STATE 3-1]
        if self.state == ParkingState.STATE3_1_HITCH_ZERO_FORWARD:
            self.publish_command(0, self.forward_speed)
            if self.parking_y_reached_state3_forward_line():
                self.enter_state3_reverse("parking_y <= state3 forward line -> state3-2 reverse")
            return

        # [STATE 3-2]
        if self.state == ParkingState.STATE3_2_NEUTRAL_REVERSE:
            steering = self.calculate_state3_ver2_steering()
            self.publish_command(steering, self.reverse_speed)
            if self.parking_currently_missing():
                self.state = ParkingState.STATE4_MISSING_CONFIRM
                self.missing_start_time = time.time()
                self.log("parking disappeared -> state4")
            return

        # [STATE 4]
        if self.state == ParkingState.STATE4_MISSING_CONFIRM:
            self.publish_command(0, self.reverse_speed)
            if not self.parking_currently_missing():
                self.missing_start_time = None
                self.enter_state3_reverse("parking detected again -> state3-2")
                return

            elapsed = self.parking_missing_elapsed()
            if elapsed >= self.parking_missing_complete_sec:
                self.complete(f"parking disappeared for {elapsed:.1f}s")

    def calculate_state3_ver2_steering(self) -> int:
        if self.path_data is None or len(self.path_data) < 2:
            return 0

        origin_x, origin_y = self.path_data[0]
        lookahead_index = min(max(1, self.state3_lookahead_index), len(self.path_data) - 1)
        target_x, target_y = self.path_data[lookahead_index]

        dx = target_x - origin_x
        dy = origin_y - target_y
        if abs(dy) > 1e-5:
            target_angle = -math.degrees(math.atan2(dx, dy))
        else:
            target_angle = 0.0

        if abs(target_angle) < self.state3_target_angle_deadzone_deg:
            target_angle = 0.0

        gamma_ref = self.state3_k_lateral * target_angle
        gamma_ref = clamp(gamma_ref, -self.state3_gamma_limit_deg, self.state3_gamma_limit_deg)

        hitch_error = gamma_ref - self.current_hitch_angle
        self.state3_target_slope_f = (
            (1.0 - self.state3_alpha) * self.state3_target_slope_f
            + self.state3_alpha * hitch_error
        )

        step = int(round(self.state3_k_hitch * self.state3_target_slope_f))
        step = int(clamp(step, -MAX_STEP, MAX_STEP))
        step = int(clamp(step, self.state3_prev_step - self.state3_max_step_delta, self.state3_prev_step + self.state3_max_step_delta))
        self.state3_prev_step = step
        return step

    def ready_for_state3_strict(self, heading_error: float) -> bool:
        if self.last_parking_center_x is None:
            return False

        parking_centered = abs(self.last_parking_center_x - IMG_CX) <= self.center_tolerance_px

        aruco_parallel = False
        if self.last_aruco_yaw is not None:
            yaw = abs(self.last_aruco_yaw)
            yaw_error = min(yaw, abs(yaw - 180.0))
            if yaw_error <= self.aruco_parallel_tolerance_deg:
                aruco_parallel = True

        return parking_centered and self.last_parking_is_horizontal and aruco_parallel

    def parking_horizontal_alignment(self, contour: Optional[List[Point]]) -> Tuple[bool, Optional[float]]:
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

    def parking_y_reached_state1_retry_line(self) -> bool:
        if self.last_parking_center_y is None:
            return False
        return self.last_parking_center_y <= self.state2_neutral_forward_target_parking_y

    def parking_y_reached_state3_forward_line(self) -> bool:
        if self.last_parking_center_y is None:
            return False
        return self.last_parking_center_y <= self.state3_forward_target_parking_y

    def neutral_forward_elapsed(self) -> float:
        if self.neutral_forward_start_time is None:
            return 0.0
        return time.time() - self.neutral_forward_start_time

    def state1_to_state2_stop_elapsed(self) -> float:
        if self.state1_to_state2_stop_start_time is None:
            return 0.0
        return time.time() - self.state1_to_state2_stop_start_time

    def jackknife_stop_elapsed(self) -> float:
        if self.jackknife_stop_start_time is None:
            return 0.0
        return time.time() - self.jackknife_stop_start_time

    def jackknife_forward_elapsed(self) -> float:
        if self.jackknife_forward_start_time is None:
            return 0.0
        return time.time() - self.jackknife_forward_start_time

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
        target = int(clamp(target, self.last_steering - self.max_step_delta, self.last_steering + self.max_step_delta))
        self.last_steering = target

        msg = MotionCommand()
        msg.steering = target
        msg.left_speed = int(speed)
        msg.right_speed = int(speed)
        self.publisher.publish(msg)

        command_signature = (target, int(speed))
        if command_signature != self.last_command_signature:
            self.last_command_signature = command_signature
            self.log(f"command changed: steering={target}, speed={int(speed)}")

    def complete(self, reason: str) -> None:
        if self.state == ParkingState.DONE:
            return
        self.state = ParkingState.DONE
        self.publish_command(0, STOP_SPEED)
        self.get_logger().info(f"PARKING COMPLETE: {reason}")

    def log(self, text: str) -> None:
        if not self.show_log:
            return

        log_signature = (self.state, text)
        if log_signature == self.last_log_signature:
            return
        self.last_log_signature = log_signature
        
        aruco_val = f"{self.last_aruco_yaw:.1f}°" if self.last_aruco_yaw is not None else "None"

        self.get_logger().info(
            f"[{self.state}] {text}, side={self.parking_side}, "
            f"hitch={self.current_hitch_angle:.1f}, "
            f"err_ang={self.current_heading_error:.1f}°, "
            f"aruco_yaw={aruco_val}, "
            f"parking_x={self.last_parking_center_x}, "
            f"parking_y={self.last_parking_center_y}"
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
        return [(float(point.x), float(point.y)) for point in detection.mask.data]

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
