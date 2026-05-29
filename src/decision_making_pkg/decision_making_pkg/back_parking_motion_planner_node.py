#!/usr/bin/env python3
"""
parking_motion_planner_node.py

굴절 차량 후진 주차용 motion planner.

입력
----
- path_planning_result (interfaces_pkg/msg/PathPlanningResult)
    parking_path_planner_node가 발행하는 픽셀 좌표 기반 Bézier 경로.
    path[0]은 left/right 마커 중점(현재 차량 기준점), path[-1]은 parking 중심점.
- /articulation/angle (std_msgs/msg/Float32)
    A0 굴절부 센서로 계산한 굴절각 [deg].

출력
----
- topic_control_signal (interfaces_pkg/msg/MotionCommand)
    serial_sender_node를 거쳐 Arduino에 전달될 조향/모터 명령.

중요
----
- 실제 조향 좌우 방향은 카메라 장착 방향, Arduino 조향 부호에 따라 달라질 수 있다.
  처음 바퀴를 띄운 상태에서 steering_sign:=1.0 또는 -1.0을 확인해야 한다.
- stop_distance_px는 아직 영상 픽셀 단위이다. BEV/거리 보정을 적용하면 미터 기준으로 교체한다.
"""

import math
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from std_msgs.msg import Float32

from interfaces_pkg.msg import MotionCommand, PathPlanningResult


Point = Tuple[float, float]


class ParkingMotionPlannerNode(Node):
    """Generate low-speed reverse commands for articulated-vehicle parking."""

    def __init__(self) -> None:
        super().__init__("parking_motion_planner_node")

        # Topics
        self.path_topic = self.declare_parameter(
            "path_topic", "path_planning_result"
        ).value
        self.articulation_topic = self.declare_parameter(
            "articulation_topic", "/articulation/angle"
        ).value
        self.command_topic = self.declare_parameter(
            "command_topic", "topic_control_signal"
        ).value

        # Control frequency / data validity
        self.timer_period = float(self.declare_parameter("timer", 0.10).value)
        self.path_timeout_sec = float(
            self.declare_parameter("path_timeout_sec", 0.50).value
        )
        self.articulation_timeout_sec = float(
            self.declare_parameter("articulation_timeout_sec", 0.50).value
        )
        self.require_articulation = bool(
            self.declare_parameter("require_articulation", True).value
        )

        # Steering control
        self.max_steering_step = int(
            self.declare_parameter("max_steering_step", 7).value
        )
        self.max_step_delta = int(
            self.declare_parameter("max_step_delta", 1).value
        )
        self.lookahead_index = int(
            self.declare_parameter("lookahead_index", 8).value
        )
        self.turn_deadband_deg = float(
            self.declare_parameter("turn_deadband_deg", 2.0).value
        )
        self.turn_angle_for_max_step_deg = float(
            self.declare_parameter("turn_angle_for_max_step_deg", 35.0).value
        )
        self.steering_sign = float(
            self.declare_parameter("steering_sign", 1.0).value
        )
        self.steering_lpf_alpha = float(
            self.declare_parameter("steering_lpf_alpha", 0.35).value
        )

        # Optional articulation correction:
        # 0.0 keeps the first test safe until the real corrective sign is verified.
        self.articulation_gain = float(
            self.declare_parameter("articulation_gain", 0.0).value
        )

        # Reverse speed / stopping conditions
        self.reverse_speed = int(
            self.declare_parameter("reverse_speed", -30).value
        )
        self.slow_reverse_speed = int(
            self.declare_parameter("slow_reverse_speed", -18).value
        )
        self.slow_turn_angle_deg = float(
            self.declare_parameter("slow_turn_angle_deg", 18.0).value
        )
        self.articulation_warning_deg = float(
            self.declare_parameter("articulation_warning_deg", 22.0).value
        )
        self.articulation_stop_deg = float(
            self.declare_parameter("articulation_stop_deg", 32.0).value
        )
        self.stop_distance_px = float(
            self.declare_parameter("stop_distance_px", 25.0).value
        )

        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )

        # Latest input state
        self.path_data: Optional[List[Point]] = None
        self.path_received_time = None
        self.articulation_angle: Optional[float] = None
        self.articulation_received_time = None

        # Command state for LPF / slew limit
        self.filtered_turn_angle = 0.0
        self.previous_steering_step = 0
        self.last_state = ""

        self.path_sub = self.create_subscription(
            PathPlanningResult,
            self.path_topic,
            self.path_callback,
            self.qos_profile,
        )
        self.articulation_sub = self.create_subscription(
            Float32,
            self.articulation_topic,
            self.articulation_callback,
            self.qos_profile,
        )
        self.command_pub = self.create_publisher(
            MotionCommand,
            self.command_topic,
            self.qos_profile,
        )
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        self.get_logger().info(
            "Parking motion planner started: "
            f"path={self.path_topic}, articulation={self.articulation_topic}, "
            f"command={self.command_topic}, reverse_speed={self.reverse_speed}"
        )
        self.get_logger().warn(
            "First test must be performed with wheels lifted or motor speed limited; "
            "verify steering_sign before ground driving."
        )

    def path_callback(self, msg: PathPlanningResult) -> None:
        if len(msg.x_points) != len(msg.y_points):
            self.get_logger().warn("Ignored invalid path: x_points/y_points length mismatch.")
            return

        self.path_data = [
            (float(x), float(y)) for x, y in zip(msg.x_points, msg.y_points)
        ]
        self.path_received_time = self.get_clock().now()

    def articulation_callback(self, msg: Float32) -> None:
        self.articulation_angle = float(msg.data)
        self.articulation_received_time = self.get_clock().now()

    def age_seconds(self, stamp) -> float:
        if stamp is None:
            return float("inf")
        return (self.get_clock().now() - stamp).nanoseconds / 1.0e9

    @staticmethod
    def distance(point_a: Point, point_b: Point) -> float:
        return math.hypot(point_b[0] - point_a[0], point_b[1] - point_a[1])

    @staticmethod
    def signed_angle_between(vector_a: Point, vector_b: Point) -> float:
        """Return signed smallest angle from a to b in degrees."""
        cross = vector_a[0] * vector_b[1] - vector_a[1] * vector_b[0]
        dot = vector_a[0] * vector_b[0] + vector_a[1] * vector_b[1]
        return math.degrees(math.atan2(cross, dot))

    def calculate_path_turn_angle(self, path: List[Point]) -> Optional[float]:
        """
        Measure how the new Bézier path bends immediately ahead of the marker center.

        Local tangent: path[1] - path[0]
        Look-ahead direction: path[index] - path[0]
        Positive/negative value indicates curve direction in image coordinates.
        """
        if len(path) < 3:
            return None

        index = max(2, min(self.lookahead_index, len(path) - 1))
        local_vector = (path[1][0] - path[0][0], path[1][1] - path[0][1])
        ahead_vector = (path[index][0] - path[0][0], path[index][1] - path[0][1])

        if math.hypot(*local_vector) < 1.0e-6 or math.hypot(*ahead_vector) < 1.0e-6:
            return None

        return self.signed_angle_between(local_vector, ahead_vector)

    def reset_steering(self) -> None:
        self.filtered_turn_angle = 0.0
        self.previous_steering_step = 0

    def publish_command(self, steering: int, speed: int, state: str) -> None:
        command = MotionCommand()
        command.steering = int(steering)
        command.left_speed = int(speed)
        command.right_speed = int(speed)
        self.command_pub.publish(command)

        if state != self.last_state:
            self.get_logger().info(
                f"parking_state={state}, steering={steering}, speed={speed}, "
                f"articulation={self.articulation_angle}"
            )
            self.last_state = state

    def publish_stop(self, state: str) -> None:
        self.reset_steering()
        self.publish_command(0, 0, state)

    def compute_steering_step(self, path_turn_angle: float) -> int:
        if abs(path_turn_angle) < self.turn_deadband_deg:
            path_turn_angle = 0.0

        self.filtered_turn_angle = (
            (1.0 - self.steering_lpf_alpha) * self.filtered_turn_angle
            + self.steering_lpf_alpha * path_turn_angle
        )

        # Path bend contribution. steering_sign is intentionally configurable.
        steering_float = (
            self.steering_sign
            * self.filtered_turn_angle
            / self.turn_angle_for_max_step_deg
            * self.max_steering_step
        )

        # Disabled by default until the correction direction is calibrated on the vehicle.
        if self.articulation_angle is not None:
            steering_float += self.articulation_gain * self.articulation_angle

        desired_step = int(round(steering_float))
        desired_step = max(
            -self.max_steering_step, min(self.max_steering_step, desired_step)
        )

        # Do not command abrupt full-lock steering between timer cycles.
        limited_step = max(
            self.previous_steering_step - self.max_step_delta,
            min(self.previous_steering_step + self.max_step_delta, desired_step),
        )
        self.previous_steering_step = limited_step
        return limited_step

    def timer_callback(self) -> None:
        # 1. Stop if path detections disappear or stale path remains in memory.
        if self.path_data is None or self.age_seconds(self.path_received_time) > self.path_timeout_sec:
            self.publish_stop("STOP_PATH_TIMEOUT")
            return

        if len(self.path_data) < 3:
            self.publish_stop("STOP_INVALID_PATH")
            return

        # 2. For articulated parking, do not reverse without a current articulation reading.
        if self.require_articulation:
            if (
                self.articulation_angle is None
                or self.age_seconds(self.articulation_received_time)
                > self.articulation_timeout_sec
            ):
                self.publish_stop("STOP_ARTICULATION_TIMEOUT")
                return

        # 3. Hard jackknife prevention.
        if (
            self.articulation_angle is not None
            and abs(self.articulation_angle) >= self.articulation_stop_deg
        ):
            self.publish_stop("STOP_ARTICULATION_LIMIT")
            return

        # 4. Target reached in provisional pixel distance.
        remaining_distance = self.distance(self.path_data[0], self.path_data[-1])
        if remaining_distance <= self.stop_distance_px:
            self.publish_stop("PARKING_COMPLETE")
            return

        # 5. Calculate steering demand from path curvature.
        path_turn_angle = self.calculate_path_turn_angle(self.path_data)
        if path_turn_angle is None:
            self.publish_stop("STOP_PATH_GEOMETRY")
            return

        steering_step = self.compute_steering_step(path_turn_angle)

        # 6. Slow reverse if a sharp curve or increasing articulation is seen.
        should_slow = abs(path_turn_angle) >= self.slow_turn_angle_deg
        if self.articulation_angle is not None:
            should_slow = should_slow or (
                abs(self.articulation_angle) >= self.articulation_warning_deg
            )

        speed = self.slow_reverse_speed if should_slow else self.reverse_speed
        state = "REVERSE_SLOW" if should_slow else "REVERSE_TRACKING"
        self.publish_command(steering_step, speed, state)

    def stop_vehicle(self) -> None:
        self.publish_stop("SHUTDOWN_STOP")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ParkingMotionPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.stop_vehicle()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
