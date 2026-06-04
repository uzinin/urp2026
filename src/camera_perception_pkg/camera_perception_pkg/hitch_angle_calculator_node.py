#!/usr/bin/env python3

import math
from typing import Dict, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from std_msgs.msg import Float32

from interfaces_pkg.msg import DetectionArray


SUB_DETECTION_TOPIC_NAME = "detections"
PUB_HITCH_TOPIC_NAME = "/articulation/angle"

CLS_LEFT = "left"
CLS_RIGHT = "right"

ORIGIN_X = 320.0
ORIGIN_Y = 480.0
MINIMUM_SCORE = 0.50
ANGLE_DIVISOR = 2.0
ANGLE_SMOOTHING_ALPHA = 0.35


class HitchAngleCalculatorNode(Node):
    def __init__(self) -> None:
        super().__init__("hitch_angle_calculator_node")

        self.sub_detection_topic = self.declare_parameter(
            "sub_detection_topic", SUB_DETECTION_TOPIC_NAME
        ).value
        self.pub_hitch_topic = self.declare_parameter(
            "pub_hitch_topic", PUB_HITCH_TOPIC_NAME
        ).value
        self.origin_x = float(self.declare_parameter("origin_x", ORIGIN_X).value)
        self.origin_y = float(self.declare_parameter("origin_y", ORIGIN_Y).value)
        self.minimum_score = float(
            self.declare_parameter("minimum_score", MINIMUM_SCORE).value
        )
        self.angle_divisor = float(
            self.declare_parameter("angle_divisor", ANGLE_DIVISOR).value
        )
        self.angle_smoothing_alpha = float(
            self.declare_parameter(
                "angle_smoothing_alpha",
                ANGLE_SMOOTHING_ALPHA,
            ).value
        )
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
        self.hitch_pub = self.create_publisher(Float32, self.pub_hitch_topic, qos)
        self.last_logged_angle: Optional[float] = None
        self.filtered_angle: Optional[float] = None

        self.get_logger().info(
            "hitch_angle_calculator_node started: "
            f"origin=({self.origin_x:.1f}, {self.origin_y:.1f}), "
            f"angle_divisor={self.angle_divisor:.1f}, "
            f"smoothing_alpha={self.angle_smoothing_alpha:.2f}"
        )

    def detection_callback(self, msg: DetectionArray) -> None:
        centers = self.extract_marker_centers(msg)
        left = centers.get(CLS_LEFT)
        right = centers.get(CLS_RIGHT)
        if left is None or right is None:
            return

        midpoint_x = (left[0] + right[0]) / 2.0
        midpoint_y = (left[1] + right[1]) / 2.0

        dx = midpoint_x - self.origin_x
        dy = self.origin_y - midpoint_y
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            angle_deg = 0.0
        else:
            angle_deg = math.degrees(math.atan2(dx, dy))

        scaled_angle_deg = angle_deg / max(self.angle_divisor, 1e-6)
        if self.filtered_angle is None:
            self.filtered_angle = scaled_angle_deg
        else:
            alpha = max(0.0, min(1.0, self.angle_smoothing_alpha))
            self.filtered_angle = (
                (1.0 - alpha) * self.filtered_angle
                + alpha * scaled_angle_deg
            )
        published_angle_deg = self.filtered_angle

        hitch_msg = Float32()
        hitch_msg.data = float(published_angle_deg)
        self.hitch_pub.publish(hitch_msg)

        if self.should_log(published_angle_deg):
            self.get_logger().info(
                f"hitch_angle={published_angle_deg:.1f} deg, "
                f"raw_angle={angle_deg:.1f} deg, "
                f"scaled_angle={scaled_angle_deg:.1f} deg, "
                f"midpoint=({midpoint_x:.1f}, {midpoint_y:.1f}), "
                f"left=({left[0]:.1f}, {left[1]:.1f}), "
                f"right=({right[0]:.1f}, {right[1]:.1f})"
            )

    def extract_marker_centers(
        self, msg: DetectionArray
    ) -> Dict[str, Tuple[float, float]]:
        result: Dict[str, Tuple[float, float]] = {}
        best_score: Dict[str, float] = {}

        for detection in msg.detections:
            name = str(detection.class_name).strip().lower()
            if name not in (CLS_LEFT, CLS_RIGHT):
                continue
            score = float(detection.score)
            if score < self.minimum_score:
                continue
            if name in best_score and score <= best_score[name]:
                continue

            best_score[name] = score
            result[name] = (
                float(detection.bbox.center.position.x),
                float(detection.bbox.center.position.y),
            )

        return result

    def should_log(self, angle_deg: float) -> bool:
        if not self.show_log:
            return False
        if self.last_logged_angle is None:
            self.last_logged_angle = angle_deg
            return True
        if abs(angle_deg - self.last_logged_angle) < 0.5:
            return False
        self.last_logged_angle = angle_deg
        return True


def main(args=None) -> None:
    rclpy.init(args=args)
    node = HitchAngleCalculatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
