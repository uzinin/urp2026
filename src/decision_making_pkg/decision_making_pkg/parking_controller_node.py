#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
parking_controller_node.py  (articulation 센서 연동 버전)

트레일러 견인 차량 후방 카메라 + 힌지 전위차계 기반 자율 주차 - 주차 알고리즘 (1).

[입력]
  detections   (interfaces_pkg/DetectionArray) <- yolov8_node   : 카메라 인지 (left/right/parking)
  articulation (std_msgs/Int32)                <- serial node    : 힌지 관절 ADC (실측 굴절각)
[출력]
  topic_control_signal (interfaces_pkg/MotionCommand) -> serial node

제어 구조 (캐스케이드):
  - 바깥 루프(카메라): 주차판-트레일러 측방오차 e_lat, 트레일러 정렬각 theta -> 목표 굴절각 art_des
  - 안쪽 루프(힌지센서): 실측 굴절각 art -> art_des 가 되도록 견인차 조향각 결정
  - 굴절각이 너무 커지면 정지 후 펴기(잭나이프 복구)
  - 트레일러가 주차판과 위치/각도 허용범위에 들어오면 완료

펌웨어 규약(driving.ino): steering -7~7 (+=우회전, 내부 폐루프), speed -255~255 (음수=후진).
"""

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSHistoryPolicy,
                       QoSDurabilityPolicy, QoSReliabilityPolicy)

from interfaces_pkg.msg import DetectionArray, MotionCommand
from std_msgs.msg import Int32


# =================== 설정값 (ros2 param set 으로 변경 가능) ===================
SUB_DETECTION_TOPIC = 'detections'
SUB_ARTIC_TOPIC     = 'articulation'
PUB_TOPIC           = 'topic_control_signal'

CLS_LEFT, CLS_RIGHT, CLS_PARKING = 'left', 'right', 'parking'
IMG_W, IMG_H = 640, 480
IMG_CX = IMG_W / 2.0

CONTROL_PERIOD = 0.1
DET_TIMEOUT    = 0.5   # detection 끊기면 정지 [s]
ARTIC_TIMEOUT  = 0.3   # articulation 끊기면 정지 [s] (센서모드)

# --- 구동/조향 한계 (펌웨어 규약) ---
STEER_MAX     = 7      # 펌웨어 MAX_STEERING_STEP
REVERSE_SPEED = -80    # 후진 속도 (음수=후진, 펌웨어가 처리)
FORWARD_SPEED = 80     # 잭나이프 복구 전진 속도
STOP_SPEED    = 0

# --- 굴절각(art) 신호 선택 ---
USE_ARTIC_SENSOR = True   # True: 힌지센서(A0, ADC) 사용 / False: 카메라 픽셀 proxy 사용
ARTIC_CENTER_ADC = 512    # 트레일러 정렬(직선)일 때 ADC 값  ★캘리브레이션 필수
ARTIC_SIGN       = 1      # (adc-center)*sign 이 '트레일러가 화면 우측으로 꺾임'=+ 가 되도록  ★

# --- 제어 게인 (전부 튜닝 대상; 기본값은 센서모드 ADC 단위) ---
KP_LAT  = 1.0    # 측방오차(px)        -> 목표 굴절각(art 단위)
KP_HEAD = 0.0    # 트레일러 정렬각(deg) -> 목표 굴절각(art 단위)  (위치 먼저 맞춘 뒤 키울 것)
KP_ART  = 0.03   # 굴절각 오차(art)     -> 조향
ART_MAX    = 250.0   # 목표 굴절각 한계 [art 단위]
STEER_SIGN = 1       # 후진 역학+카메라 반전 보정. 음피드백이 안 되면 -1
#  ※ 카메라 proxy 모드(USE_ARTIC_SENSOR=False)일 때 권장값:
#     ART_MAX≈130, KP_LAT≈0.5, KP_ART≈0.06, JACK_ENTER≈170, JACK_EXIT≈60 (단위=px)

# --- 잭나이프(과도 굴절) 보호 (art 단위) ---
JACK_ENTER = 320.0
JACK_EXIT  = 120.0
RECOVERY_FORWARD = True

# --- 주차 완료 판정 (카메라, px/deg) ---
POS_TOL     = 30.0
ANG_TOL     = 8.0
PARK_NEAR_H = 120.0

SHOW_LOG = True
# =============================================================================


class S:
    SEARCH, REVERSE, RECOVERY, DONE = 'SEARCH', 'REVERSE', 'RECOVERY', 'DONE'


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class ParkingControllerNode(Node):
    def __init__(self):
        super().__init__('parking_controller_node')

        gp = self.declare_parameter
        self.sub_topic   = gp('sub_detection_topic', SUB_DETECTION_TOPIC).value
        self.artic_topic = gp('sub_artic_topic', SUB_ARTIC_TOPIC).value
        self.pub_topic   = gp('pub_topic', PUB_TOPIC).value
        self.use_artic   = gp('use_artic_sensor', USE_ARTIC_SENSOR).value
        self.artic_center= gp('artic_center_adc', ARTIC_CENTER_ADC).value
        self.artic_sign  = gp('artic_sign', ARTIC_SIGN).value
        self.kp_lat      = gp('kp_lat', KP_LAT).value
        self.kp_head     = gp('kp_head', KP_HEAD).value
        self.kp_art      = gp('kp_art', KP_ART).value
        self.art_max     = gp('art_max', ART_MAX).value
        self.steer_sign  = gp('steer_sign', STEER_SIGN).value
        self.reverse_spd = gp('reverse_speed', REVERSE_SPEED).value
        self.forward_spd = gp('forward_speed', FORWARD_SPEED).value
        self.jack_enter  = gp('jack_enter', JACK_ENTER).value
        self.jack_exit   = gp('jack_exit', JACK_EXIT).value
        self.recov_fwd   = gp('recovery_forward', RECOVERY_FORWARD).value
        self.pos_tol     = gp('pos_tol', POS_TOL).value
        self.ang_tol     = gp('ang_tol', ANG_TOL).value
        self.park_near_h = gp('park_near_h', PARK_NEAR_H).value
        self.show_log    = gp('show_log', SHOW_LOG).value

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )

        self.sub = self.create_subscription(
            DetectionArray, self.sub_topic, self.detection_cb, qos)
        self.artic_sub = self.create_subscription(
            Int32, self.artic_topic, self.artic_cb, qos)
        self.pub = self.create_publisher(MotionCommand, self.pub_topic, qos)

        self.latest = None
        self.last_det_t = 0.0
        self.artic_adc = None
        self.last_artic_t = 0.0
        self.state = S.SEARCH

        self.timer = self.create_timer(CONTROL_PERIOD, self.control_loop)
        mode = 'ARTICULATION SENSOR' if self.use_artic else 'CAMERA proxy'
        self.get_logger().info(f'parking_controller_node started (inner loop: {mode})')

    # ---------------- 콜백 ----------------
    def detection_cb(self, msg: DetectionArray):
        self.latest = msg
        self.last_det_t = time.time()

    def artic_cb(self, msg: Int32):
        self.artic_adc = msg.data
        self.last_artic_t = time.time()

    # ---------------- 인지 ----------------
    def _best(self, dets, name):
        cand = [d for d in dets if d.class_name == name]
        return max(cand, key=lambda d: d.score) if cand else None

    def perceive(self):
        if self.latest is None:
            return None
        dets = self.latest.detections
        L = self._best(dets, CLS_LEFT)
        R = self._best(dets, CLS_RIGHT)
        P = self._best(dets, CLS_PARKING)
        if L is None or R is None or P is None:
            return None

        lx, ly = L.bbox.center.position.x, L.bbox.center.position.y
        rx, ry = R.bbox.center.position.x, R.bbox.center.position.y
        px     = P.bbox.center.position.x
        p_h    = P.bbox.size.y

        trailer_cx = (lx + rx) / 2.0
        theta = math.degrees(math.atan2(ry - ly, rx - lx))   # 트레일러 기울기[deg], 정렬시 0
        e_lat = px - trailer_cx                               # 측방오차[px]
        cam_gamma = trailer_cx - IMG_CX                       # 카메라 굴절 proxy[px] (fallback용)

        return dict(trailer_cx=trailer_cx, theta=theta, e_lat=e_lat,
                    p_h=p_h, cam_gamma=cam_gamma)

    def get_articulation(self, f):
        """내부 루프용 굴절각(art) 반환. 센서모드면 ADC offset, 아니면 카메라 proxy(px)."""
        if self.use_artic:
            if self.artic_adc is None:
                return None
            return (self.artic_adc - self.artic_center) * self.artic_sign
        return f['cam_gamma']

    # ---------------- 제어 루프 ----------------
    def control_loop(self):
        if self.state == S.DONE:
            self._send(0, STOP_SPEED, STOP_SPEED)
            return

        now = time.time()
        if (now - self.last_det_t) > DET_TIMEOUT:
            self.state = S.SEARCH
            self._send(0, STOP_SPEED, STOP_SPEED)
            self._log('no detection -> STOP')
            return
        if self.use_artic and (now - self.last_artic_t) > ARTIC_TIMEOUT:
            self.state = S.SEARCH
            self._send(0, STOP_SPEED, STOP_SPEED)
            self._log('no articulation feedback -> STOP (시리얼/보드레이트 확인)')
            return

        f = self.perceive()
        if f is None:
            self.state = S.SEARCH
            self._send(0, STOP_SPEED, STOP_SPEED)
            self._log('missing left/right/parking -> STOP')
            return

        art = self.get_articulation(f)
        if art is None:
            self.state = S.SEARCH
            self._send(0, STOP_SPEED, STOP_SPEED)
            self._log('no articulation value -> STOP')
            return

        # ---- 잭나이프 복구 ----
        if self.state == S.RECOVERY:
            if abs(art) <= self.jack_exit:
                self.state = S.REVERSE
            else:
                self._recover(art)
                return
        elif abs(art) >= self.jack_enter:
            self.state = S.RECOVERY
            self._recover(art)
            return

        # ---- 완료 판정 ----
        if (abs(f['e_lat']) <= self.pos_tol and
                abs(f['theta']) <= self.ang_tol and
                f['p_h'] >= self.park_near_h):
            self.state = S.DONE
            self._send(0, STOP_SPEED, STOP_SPEED)
            self.get_logger().info('=== PARKING DONE ===')
            return

        # ---- 후진 주차 제어 (캐스케이드) ----
        self.state = S.REVERSE
        art_des = clamp(self.kp_lat * f['e_lat'] + self.kp_head * (-f['theta']),
                        -self.art_max, self.art_max)
        e_art = art_des - art
        steer = self.steer_sign * self.kp_art * e_art
        steer = int(round(clamp(steer, -STEER_MAX, STEER_MAX)))

        self._send(steer, self.reverse_spd, self.reverse_spd)
        adc_str = f' adc={self.artic_adc}' if self.use_artic else ''
        self._log(f'e_lat={f["e_lat"]:.0f} th={f["theta"]:.1f} '
                  f'art={art:.0f}{adc_str} ades={art_des:.0f} steer={steer}')

    def _recover(self, art):
        steer = self.steer_sign * (-STEER_MAX if art > 0 else STEER_MAX)
        spd = self.forward_spd if self.recov_fwd else self.reverse_spd
        self._send(int(steer), spd, spd)
        self._log(f'art={art:.0f} steer={int(steer)} '
                  f'{"FWD" if self.recov_fwd else "REV"}')

    # ---------------- 유틸 ----------------
    def _send(self, steering, left, right):
        msg = MotionCommand()
        msg.steering = int(steering)
        msg.left_speed = int(left)
        msg.right_speed = int(right)
        self.pub.publish(msg)

    def _log(self, txt):
        if self.show_log:
            self.get_logger().info(f'[{self.state}] {txt}')


def main(args=None):
    rclpy.init(args=args)
    node = ParkingControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('\n\nshutdown\n\n')
    finally:
        try:
            node._send(0, 0, 0)
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
