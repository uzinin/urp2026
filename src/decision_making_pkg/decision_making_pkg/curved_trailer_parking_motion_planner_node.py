#!/usr/bin/env python3
"""
curved_trailer_parking_motion_planner_node.py

State-machine motion planner for curved reverse parking with an articulated
trailer vehicle.

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
from typing import Dict, List, Optional, Tuple

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
REVERSE_SPEED = -30
STOP_SPEED = 0


class ParkingState:
    WAIT_PATH = "WAIT_PATH"
    STATE1_INITIAL_TURN = "STATE1_INITIAL_TURN"
    STATE2_COUNTER_TURN = "STATE2_COUNTER_TURN"
    STATE3_NEUTRAL_REVERSE = "STATE3_NEUTRAL_REVERSE"
    STATE4_MISSING_CONFIRM = "STATE4_MISSING_CONFIRM"
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
        # reverse_speed:
        #   후진 속도입니다.
        #   serial_sender/펌웨어 기준으로 음수이면 후진입니다.
        #   너무 빠르면 parking 중심을 지나치기 쉬우니,
        #   처음에는 -20~-30 정도로 낮게 시작하세요.
        self.reverse_speed = int(
            self.declare_parameter("reverse_speed", REVERSE_SPEED).value
        )

        # initial_turn_step:
        #   State1에서 parking 방향으로 휘게 만들 때 쓰는 조향 크기입니다.
        #   parking이 화면 오른쪽이면 +값, 왼쪽이면 -값으로 자동 변환됩니다.
        #   곡선 진입이 부족하면 키우고, 너무 빨리 접히면 줄이세요.
        #   범위는 0~7입니다.
        self.initial_turn_step = int(
            self.declare_parameter("initial_turn_step", MAX_STEP).value
        )

        # counter_turn_step:
        #   State2에서 트레일러를 펴기 위해 반대 방향으로 치는
        #   조향 크기입니다. articulation angle이 계속 커지면 키우고,
        #   너무 빨리 펴져서 경로를 놓치면 줄이세요.
        #   범위는 0~7입니다.
        self.counter_turn_step = int(
            self.declare_parameter("counter_turn_step", 5).value
        )

        # steering_sign:
        #   실제 차량에서 조향 부호가 반대로 먹을 때 -1로 바꾸는
        #   최종 부호 보정값입니다.
        #   기존 펌웨어 규약은 +가 오른쪽, -가 왼쪽입니다.
        self.steering_sign = int(self.declare_parameter("steering_sign", 1).value)

        # max_step_delta:
        #   타이머 한 주기마다 조향 명령이 바뀔 수 있는 최대량입니다.
        #   작을수록 부드럽고, 클수록 반응이 빠릅니다.
        self.max_step_delta = int(self.declare_parameter("max_step_delta", 2).value)

        # state1_hitch_trigger_deg:
        #   State1 -> State2 전환 기준 articulation angle입니다.
        #   이 각도에 도달하면 parking 방향 조향을 멈추고
        #   반대 조향을 시작합니다.
        #   너무 늦게 반대 조향하면 낮추고, 너무 빨리 펴지면 올리세요.
        self.state1_hitch_trigger_deg = float(
            self.declare_parameter("state1_hitch_trigger_deg", 20.0).value
        )

        # center_tolerance_px:
        #   State2 -> State3 전환 기준입니다.
        #   parking 중심 x좌표가 이미지 중앙(기본 320px)에서
        #   이 값 이내로 들어오면
        #   조향 중립 후진으로 넘어갑니다.
        self.center_tolerance_px = float(
            self.declare_parameter("center_tolerance_px", 35.0).value
        )

        # parking_missing_complete_sec:
        #   State3 이후 parking detection이 이 시간 이상 사라지면
        #   주차 완료로 봅니다.
        #   요청하신 기본값은 2초입니다.
        self.parking_missing_complete_sec = float(
            self.declare_parameter("parking_missing_complete_sec", 2.0).value
        )

        # parking_detection_timeout_sec:
        #   parking이 "잠깐 안 보인다"고 판단하는 최소 시간입니다.
        #   이 시간이 지나면 State4로 들어가고,
        #   총 missing 시간이 위 complete_sec를 넘으면 완료됩니다.
        #   camera/yolo 프레임 드랍이 많으면 0.3보다 조금 키우세요.
        self.parking_detection_timeout_sec = float(
            self.declare_parameter("parking_detection_timeout_sec", 0.30).value
        )

        # jackknife_limit_deg:
        #   articulation angle 절댓값이 이 값을 넘으면 안전을 위해 정지합니다.
        #   실제 차량 한계보다 보수적으로 잡으세요.
        self.jackknife_limit_deg = float(
            self.declare_parameter("jackknife_limit_deg", 30.0).value
        )

        # minimum_score:
        #   detections에서 left/right/parking으로 인정할 최소 confidence입니다.
        #   오검출이 많으면 올리고, detection이 자주 끊기면 낮추세요.
        self.minimum_score = float(self.declare_parameter("minimum_score", 0.50).value)

        # show_log:
        #   True이면 state 전환과 주요 값을 ROS log로 출력합니다.
        # =====================================================================
        self.show_log = bool(self.declare_parameter("show_log", True).value)

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
        self.last_parking_seen_time: Optional[float] = None
        self.current_hitch_angle = 0.0

        self.parking_side = 1
        self.state = ParkingState.WAIT_PATH
        self.last_steering = 0
        self.missing_start_time: Optional[float] = None

        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        self.get_logger().info(
            "curved_trailer_parking_motion_planner_node started: "
            f"detections={self.sub_detection_topic}, path={self.sub_path_topic}, "
            f"hitch={self.sub_hitch_topic}, pub={self.pub_topic}"
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

    def timer_callback(self) -> None:
        if self.state == ParkingState.DONE:
            self.publish_command(0, STOP_SPEED)
            return

        if self.path_data is None and self.state == ParkingState.WAIT_PATH:
            self.publish_command(0, STOP_SPEED)
            self.log("waiting for first path")
            return

        if abs(self.current_hitch_angle) >= self.jackknife_limit_deg:
            self.publish_command(0, STOP_SPEED)
            self.log(f"jackknife limit stop: hitch={self.current_hitch_angle:.1f}")
            return

        if self.state == ParkingState.WAIT_PATH:
            self.state = ParkingState.STATE1_INITIAL_TURN

        if self.state == ParkingState.STATE1_INITIAL_TURN:
            steering = self.parking_side * self.initial_turn_step
            self.publish_command(steering, self.reverse_speed)
            if abs(self.current_hitch_angle) >= self.state1_hitch_trigger_deg:
                self.state = ParkingState.STATE2_COUNTER_TURN
                self.log(f"hitch trigger -> state2: hitch={self.current_hitch_angle:.1f}")
            return

        if self.state == ParkingState.STATE2_COUNTER_TURN:
            steering = -self.parking_side * self.counter_turn_step
            self.publish_command(steering, self.reverse_speed)
            if self.parking_is_centered():
                self.state = ParkingState.STATE3_NEUTRAL_REVERSE
                self.log("parking center reached image center -> state3")
            return

        if self.state == ParkingState.STATE3_NEUTRAL_REVERSE:
            self.publish_command(0, self.reverse_speed)
            if self.parking_currently_missing():
                self.state = ParkingState.STATE4_MISSING_CONFIRM
                self.missing_start_time = time.time()
                self.log("parking disappeared -> state4")
            return

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
