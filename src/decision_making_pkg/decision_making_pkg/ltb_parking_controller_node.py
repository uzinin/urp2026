#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
parking_controller_node.py  (BEV 없음 / 중점 x 기반 트레일러각 / 위치 필터 + 실시간 시각화 장착)
"""

import math
import time
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSHistoryPolicy,
                       QoSDurabilityPolicy, QoSReliabilityPolicy)
from rcl_interfaces.msg import SetParametersResult

from sensor_msgs.msg import Image       # 👈 카메라 이미지 구독을 위해 추가
from cv_bridge import CvBridge          # 👈 이미지 변환용 추가
from interfaces_pkg.msg import DetectionArray, MotionCommand


# =================== 설정값 ===================
SUB_DETECTION_TOPIC = 'detections'
SUB_IMAGE_TOPIC     = 'image_raw'       # 👈 카메라 원본 영상 토픽 명
PUB_TOPIC           = 'topic_control_signal'
CLS_LEFT, CLS_RIGHT, CLS_PARKING = 'left', 'right', 'parking'

IMG_W, IMG_H = 640, 480
CONTROL_PERIOD = 0.1
DET_TIMEOUT    = 0.5

STEER_MAX     = 7
REVERSE_SPEED = -80
FORWARD_SPEED = 80
STOP_SPEED    = 0

MASK_STEP = 8          # left/right 무게중심 계산용 폴리곤 솎기

# 트레일러 각(중점 x)
HALF_SEP = 130.0       # 한쪽만 보일 때 중점 추정용: 두 배기구 간격의 절반[px]  ★튜닝

# 위치 노이즈 필터 (left/right/parking 공통)
POS_ALPHA   = 0.4      # EMA (작을수록 부드럽고 느림)
JUMP_MAX_PX = 80.0     # 한 프레임 위치 점프가 이보다 크면 이상치로 무시[px]
MAX_SKIPS   = 5

# 완료 (폴리곤 포함)
INSIDE_MARGIN = 0.0    # left/right 가 폴리곤 경계 안쪽 이만큼이어야 인정[px]
DONE_FRAMES   = 5

# 제어 게인 (전부 px 단위 기준)
KP_POS  = 1.0          # 주차판 중심오차(px) -> 목표 트레일러각(px)
TA_MAX  = 150.0        # 목표 트레일러각 한계[px]
KP_ART  = 0.05         # 트레일러각 오차(px) -> 조향
STEER_SIGN = 1         # 후진 역학 보정. 발산하면 -1

# 잭나이프 / 소실 대비 (px)
JACK_ENTER = 140.0
JACK_EXIT  = 60.0
EDGE_MARGIN = 45.0     # 배기구 x가 좌우 가장자리 이 안쪽이면 '곧 소실'
LOST_GRACE  = 2.0
RECOVERY_MODE = 'forward'
RECOV_STEER_SIGN = 0

SHOW_LOG = True
# =============================================


class S:
    SEARCH, REVERSE, RECOVERY, DONE = 'SEARCH', 'REVERSE', 'RECOVERY', 'DONE'


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class ParkingControllerNode(Node):
    def __init__(self):
        super().__init__('parking_controller_node')

        gp = self.declare_parameter
        self.sub_topic   = gp('sub_detection_topic', SUB_DETECTION_TOPIC).value
        self.sub_img_topic = gp('sub_image_topic', SUB_IMAGE_TOPIC).value # 👈 파라미터 추가
        self.pub_topic   = gp('pub_topic', PUB_TOPIC).value
        self.mask_step   = gp('mask_step', MASK_STEP).value
        self.img_w       = gp('img_w', IMG_W).value
        self.img_h       = gp('img_h', IMG_H).value
        self.half_sep    = gp('half_sep', HALF_SEP).value
        self.pos_alpha   = gp('pos_alpha', POS_ALPHA).value
        self.jump_max_px = gp('jump_max_px', JUMP_MAX_PX).value
        self.max_skips   = gp('max_skips', MAX_SKIPS).value
        self.inside_margin = gp('inside_margin', INSIDE_MARGIN).value
        self.done_frames = gp('done_frames', DONE_FRAMES).value
        self.kp_pos      = gp('kp_pos', KP_POS).value
        self.ta_max      = gp('ta_max', TA_MAX).value
        self.kp_art      = gp('kp_art', KP_ART).value
        self.steer_sign  = gp('steer_sign', STEER_SIGN).value
        self.reverse_spd = gp('reverse_speed', REVERSE_SPEED).value
        self.forward_spd = gp('forward_speed', FORWARD_SPEED).value
        self.jack_enter  = gp('jack_enter', JACK_ENTER).value
        self.jack_exit   = gp('jack_exit', JACK_EXIT).value
        self.edge_margin = gp('edge_margin', EDGE_MARGIN).value
        self.lost_grace  = gp('lost_grace', LOST_GRACE).value
        self.recov_mode  = gp('recovery_mode', RECOVERY_MODE).value
        self.recov_steer_sign = gp('recov_steer_sign', RECOV_STEER_SIGN).value
        self.show_log    = gp('show_log', SHOW_LOG).value

        self.center_x = self.img_w / 2.0
        self.add_on_set_parameters_callback(self._on_params)

        qos = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                         history=QoSHistoryPolicy.KEEP_LAST,
                         durability=QoSDurabilityPolicy.VOLATILE, depth=1)
        
        # 구독기 그룹
        self.sub = self.create_subscription(DetectionArray, self.sub_topic, self.detection_cb, qos)
        self.sub_img = self.create_subscription(Image, self.sub_img_topic, self.image_cb, qos) # 👈 이미지 토픽 구독
        self.pub = self.create_publisher(MotionCommand, self.pub_topic, qos)

        # 이미지 핸들링 및 가시화 버퍼
        self.cv_bridge = CvBridge()
        self.current_frame = None
        self.vis_data = {}

        self.latest = None
        self.last_det_t = 0.0
        self.state = S.SEARCH
        self.done_cnt = 0
        self.last_ta = None
        self.last_ta_sign = 1
        self.last_seen_t = 0.0
        self._f = {}            # 위치 EMA 필터 상태
        self._mask_warned = False

        self.timer = self.create_timer(CONTROL_PERIOD, self.control_loop)
        self.get_logger().info(f'parking_controller_node started (image x-center angle, recovery={self.recov_mode})')

    def _on_params(self, params):
        m = {'mask_step': 'mask_step', 'img_w': 'img_w', 'img_h': 'img_h',
             'half_sep': 'half_sep', 'pos_alpha': 'pos_alpha',
             'jump_max_px': 'jump_max_px', 'max_skips': 'max_skips',
             'inside_margin': 'inside_margin', 'done_frames': 'done_frames',
             'kp_pos': 'kp_pos', 'ta_max': 'ta_max', 'kp_art': 'kp_art',
             'steer_sign': 'steer_sign', 'reverse_speed': 'reverse_spd',
             'forward_speed': 'forward_spd', 'jack_enter': 'jack_enter',
             'jack_exit': 'jack_exit', 'edge_margin': 'edge_margin',
             'lost_grace': 'lost_grace', 'recovery_mode': 'recov_mode',
             'recov_steer_sign': 'recov_steer_sign'}
        for p in params:
            if p.name in m:
                setattr(self, m[p.name], p.value)
                if p.name == 'img_w':
                    self.center_x = self.img_w / 2.0
        return SetParametersResult(successful=True)

    # ---------------- 콜백 ----------------
    def detection_cb(self, msg: DetectionArray):
        self.latest = msg
        self.last_det_t = time.time()

    def image_cb(self, msg: Image):
        """실시간 모니터링 드로잉용 원본 이미지 버퍼 백업 콜백"""
        try:
            self.current_frame = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f'이미지 전환 예외: {e}')

    # ---------------- 헬퍼 ----------------
    def _best(self, dets, name):
        c = [d for d in dets if d.class_name == name]
        return max(c, key=lambda d: d.score) if c else None

    def _mask_xy(self, det, step=1):
        m = getattr(det, 'mask', None)
        if m is None or not getattr(m, 'data', None):
            return None
        pts = m.data[::max(1, int(step))]
        if len(pts) < 3:
            pts = m.data
        if len(pts) < 3:
            return None
        return np.array([[p.x, p.y] for p in pts], dtype=np.float32)

    def _raw_centroid(self, det):
        xy = self._mask_xy(det, self.mask_step)
        if xy is None:
            c = det.bbox.center.position
            return float(c.x), float(c.y)
        m = xy.mean(axis=0)
        return float(m[0]), float(m[1])

    def _filter_xy(self, key, x, y):
        st = self._f.get(key)
        if st is None:
            self._f[key] = {'x': x, 'y': y, 'skip': 0}
            return x, y
        if math.hypot(x - st['x'], y - st['y']) > self.jump_max_px:
            st['skip'] += 1
            if st['skip'] <= self.max_skips:
                return st['x'], st['y']
        st['skip'] = 0
        a = self.pos_alpha
        st['x'] += a * (x - st['x'])
        st['y'] += a * (y - st['y'])
        return st['x'], st['y']

    def _drop(self, key):
        self._f.pop(key, None)

    def _inside(self, poly, x, y):
        d = cv2.pointPolygonTest(poly.reshape(-1, 1, 2).astype(np.float32),
                                 (float(x), float(y)), True)
        return d >= self.inside_margin

    # ---------------- 인지 ----------------
    def perceive(self):
        if self.latest is None:
            return None
        dets = self.latest.detections
        L = self._best(dets, CLS_LEFT)
        R = self._best(dets, CLS_RIGHT)
        P = self._best(dets, CLS_PARKING)

        out = dict(has_trailer=False, has_p=False, edge=False,
                   Lc=None, Rc=None, tcx=None, ta=None,
                   pcx=None, pcy=None, poly=None,
                   raw_masks=dict(L=None, R=None, P=None)) # 시각화 오버레이 백업용

        # left / right 필터 및 마스크 백업
        if L is not None:
            out['Lc'] = self._filter_xy('Lc', *self._raw_centroid(L))
            out['raw_masks']['L'] = self._mask_xy(L, 1)
        else:
            self._drop('Lc')
        if R is not None:
            out['Rc'] = self._filter_xy('Rc', *self._raw_centroid(R))
            out['raw_masks']['R'] = self._mask_xy(R, 1)
        else:
            self._drop('Rc')

        Lc, Rc = out['Lc'], out['Rc']
        if Lc is not None and Rc is not None:
            out['tcx'] = (Lc[0] + Rc[0]) / 2.0
        elif Lc is not None:
            out['tcx'] = Lc[0] + self.half_sep
        elif Rc is not None:
            out['tcx'] = Rc[0] - self.half_sep

        if out['tcx'] is not None:
            out['has_trailer'] = True
            out['ta'] = out['tcx'] - self.center_x
            for c in (Lc, Rc):
                if c is not None and (c[0] < self.edge_margin or
                                      c[0] > (self.img_w - self.edge_margin)):
                    out['edge'] = True

        # parking 필터 + 폴리곤
        if P is not None:
            out['has_p'] = True
            poly = self._mask_xy(P, 1)
            out['raw_masks']['P'] = poly
            if poly is not None:
                out['poly'] = poly
                cx, cy = float(poly[:, 0].mean()), float(poly[:, 1].mean())
            else:
                cx, cy = P.bbox.center.position.x, P.bbox.center.position.y
                if not self._mask_warned:
                    self._mask_warned = True
                    self.get_logger().warn('parking mask 없음 -> 폴리곤 완료판정 불가. yolov8 seg 확인.')
            out['pcx'], out['pcy'] = self._filter_xy('Pc', cx, cy)
        else:
            self._drop('Pc')
        return out

    # ---------------- 제어 루프 ----------------
    def control_loop(self):
        self.vis_data = {} # 프레임 루프 진입 시 그래픽스 버퍼 리셋
        
        if self.state == S.DONE:
            self._send(0, STOP_SPEED, STOP_SPEED)
            self._draw_and_show()
            return

        now = time.time()
        if (now - self.last_det_t) > DET_TIMEOUT:
            self.state = S.SEARCH; self.done_cnt = 0
            self._send(0, STOP_SPEED, STOP_SPEED)
            self._log('no detection -> STOP')
            self._draw_and_show()
            return

        f = self.perceive()
        if f is None:
            self._send(0, STOP_SPEED, STOP_SPEED)
            self._draw_and_show()
            return

        # 그래픽 오버레이용 감지 데이터 복사 연동
        self.vis_data['raw_masks'] = f['raw_masks']
        self.vis_data['Lc'] = f['Lc']
        self.vis_data['Rc'] = f['Rc']
        self.vis_data['tcx'] = f['tcx']
        self.vis_data['pcx'] = f['pcx']
        self.vis_data['pcy'] = f['pcy']

        if f['has_trailer']:
            ta = f['ta']
            self.last_ta = ta
            if abs(ta) > 1e-3:
                self.last_ta_sign = 1 if ta > 0 else -1
            self.last_seen_t = now
        else:
            ta = None

        # 배기구 둘 다 소실
        if not f['has_trailer']:
            self.done_cnt = 0
            if self.last_ta is not None and (now - self.last_seen_t) <= self.lost_grace:
                self.state = S.RECOVERY
                self._recover(self.last_ta_sign)
                self._log(f'LOST L/R -> recover (sign={self.last_ta_sign})')
            else:
                self.state = S.SEARCH
                self._send(0, STOP_SPEED, STOP_SPEED)
                self._log('LOST L/R (grace 초과) -> STOP')
            self._draw_and_show()
            return

        # 복구(가장자리 근접 or 굴절 과대)
        if self.state == S.RECOVERY:
            if abs(ta) <= self.jack_exit and not f['edge']:
                self.state = S.REVERSE
            else:
                self.done_cnt = 0
                self._recover(1 if ta > 0 else -1)
                self._log(f'RECOVER ta={ta:.0f} edge={f["edge"]}')
                self._draw_and_show()
                return
        elif abs(ta) >= self.jack_enter or f['edge']:
            self.state = S.RECOVERY
            self.done_cnt = 0
            self._recover(1 if ta > 0 else -1)
            self._log(f'enter RECOVER ta={ta:.0f} edge={f["edge"]}')
            self._draw_and_show()
            return

        # 주차판 필요
        if not f['has_p']:
            self.state = S.SEARCH; self.done_cnt = 0
            self._send(0, STOP_SPEED, STOP_SPEED)
            self._log('no parking -> STOP')
            self._draw_and_show()
            return

        # ----- 완료 판정: 폴리곤 인체크 -----
        l_in = r_in = False
        if f['poly'] is not None and f['Lc'] is not None and f['Rc'] is not None:
            l_in = self._inside(f['poly'], *f['Lc'])
            r_in = self._inside(f['poly'], *f['Rc'])
            if l_in and r_in:
                self.done_cnt += 1
                if self.done_cnt >= self.done_frames:
                    self.state = S.DONE
                    self._send(0, STOP_SPEED, STOP_SPEED)
                    self.get_logger().info('=== PARKING DONE (L,R in parking) ===')
                    self._draw_and_show()
                    return
            else:
                self.done_cnt = 0
        else:
            self.done_cnt = 0

        # 판정 상태 전달 복사
        self.vis_data.update(dict(l_in=l_in, r_in=r_in))

        # ----- 후진 제어 -----
        self.state = S.REVERSE
        e_park = self.center_x - f['pcx']
        ta_des = clamp(self.kp_pos * e_park, -self.ta_max, self.ta_max)
        e_ta = ta_des - ta
        steer = int(round(clamp(self.steer_sign * self.kp_art * e_ta, -STEER_MAX, STEER_MAX)))
        
        self._send(steer, self.reverse_spd, self.reverse_spd)
        self._log(f'ta={ta:.0f} px={f["pcx"]:.0f} e_park={e_park:.0f} '
                  f'ta_des={ta_des:.0f} steer={steer} Lin={l_in} Rin={r_in} done{self.done_cnt}')

        # 계산된 연산 인자 가시화 버퍼에 바인딩
        self.vis_data.update(dict(ta=ta, e_park=e_park, ta_des=ta_des, steer=steer))
        self._draw_and_show()

    def _recover(self, sign_a):
        if self.recov_mode == 'reverse':
            steer = -self.steer_sign * sign_a * STEER_MAX
            spd = self.reverse_spd
        else:
            steer = self.recov_steer_sign * sign_a * STEER_MAX
            spd = self.forward_spd
        self.vis_data['steer'] = int(steer)
        self._send(int(steer), int(spd), int(spd))

    # ---------------- 🖥️ 실시간 HUD 이미지 시각화 구현부 ----------------
    def _draw_and_show(self):
        if self.current_frame is None:
            return

        canvas = self.current_frame.copy()
        h, w = canvas.shape[:2]
        cx = int(self.center_x)

        # 1. 고정 기준선 가이드라인 드로잉 (이미지 세로 센터선 = 원본 차량 중앙 정렬 타겟)
        cv2.line(canvas, (cx, 0), (cx, h), (0, 140, 255), 2) # 주황색 기준 센터선
        
        masks = self.vis_data.get('raw_masks', {})

        # [A. 주차판 마스크 다각형 영역 및 중심 스폿 플로팅]
        if masks.get('P') is not None:
            p_poly = masks['P'].astype(np.int32)
            cv2.polylines(canvas, [p_poly], isClosed=True, color=(0, 255, 0), thickness=2) # 완벽 폴리곤 영역선 (초록)
            
            if 'pcx' in self.vis_data and self.vis_data['pcx'] is not None:
                px, py = int(self.vis_data['pcx']), int(self.vis_data['pcy'])
                cv2.circle(canvas, (px, py), 6, (0, 215, 255), -1) # 주차판의 연산 필터 무게중심점 (노란색)
                cv2.line(canvas, (px, py), (cx, py), (255, 255, 0), 1) # 주차판 중심오차(e_park) 가로축 가이드

        # [B. 좌우 배기구 마스크 폴리곤 및 무게중심 필터점 스폿팅]
        if self.vis_data.get('Lc') is not None:
            lx, ly = int(self.vis_data['Lc'][0]), int(self.vis_data['Lc'][1])
            if masks.get('L') is not None:
                cv2.polylines(canvas, [masks['L'].astype(np.int32)], isClosed=True, color=(255, 0, 0), thickness=1)
            # 폴리곤 내부 충족 여부에 따라 색상 피드백 변경 (포함 시 초록 원, 미포함 시 파란 원)
            l_color = (0, 255, 0) if self.vis_data.get('l_in') else (255, 0, 0)
            cv2.circle(canvas, (lx, ly), 5, l_color, -1)
            
        if self.vis_data.get('Rc') is not None:
            rx, ry = int(self.vis_data['Rc'][0]), int(self.vis_data['Rc'][1])
            if masks.get('R') is not None:
                cv2.polylines(canvas, [masks['R'].astype(np.int32)], isClosed=True, color=(0, 0, 255), thickness=1)
            r_color = (0, 255, 0) if self.vis_data.get('r_in') else (0, 0, 255)
            cv2.circle(canvas, (rx, ry), 5, r_color, -1)

        # [C. 트레일러 중점 데이터 가이드라인 투영]
        if self.vis_data.get('tcx') is not None:
            tcx_val = int(self.vis_data['tcx'])
            # 배기구 좌우 측정 높이의 평균 부근에 가로축 중점 가이드 바 표시
            mean_y = int((ly + ry)/2) if (self.vis_data.get('Lc') and self.vis_data.get('Rc')) else (ly if self.vis_data.get('Lc') else ry)
            cv2.line(canvas, (tcx_val, mean_y - 15), (tcx_val, mean_y + 15), (255, 0, 255), 2) # 핑크색 실시간 가로축 트레일러 중점 스틱
            
            # 듀얼 검출 상태인 경우 연결축 라인 연결
            if self.vis_data.get('Lc') and self.vis_data.get('Rc'):
                cv2.line(canvas, (lx, ly), (rx, ry), (255, 0, 255), 1)

        # [D. 제어 목표 타겟라인 지표 표기 (ta_des)]
        if 'ta_des' in self.vis_data:
            # ta_des는 center_x 기준으로 픽셀 오차만큼 이동한 가상의 정렬 목표선 위치임
            des_x = int(cx + self.vis_data['ta_des'])
            cv2.line(canvas, (des_x, 0), (des_x, h), (0, 255, 255), 1) # 노란색 얇은 세로 점선 대용 (목표 트레일러 중점선)

        # 2. 실시간 제어 상태창 HUD 대시보드 텍스트 출력
        y_idx = 30
        def put_txt(text, color=(255, 255, 255)):
            nonlocal y_idx
            cv2.putText(canvas, text, (15, y_idx), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, text, (15, y_idx), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)
            y_idx += 22

        state_colors = {S.SEARCH: (255, 128, 0), S.REVERSE: (0, 255, 0), S.RECOVERY: (0, 0, 255), S.DONE: (255, 255, 255)}
        curr_color = state_colors.get(self.state, (255, 255, 255))

        put_txt(f"STATE: {self.state}  (Done Count: {self.done_cnt}/{self.done_frames})", curr_color)
        
        if 'ta' in self.vis_data and self.vis_data['ta'] is not None:
            put_txt(f"Trailer Articulation (ta): {self.vis_data['ta']:.1f} px")
        if 'e_park' in self.vis_data and self.vis_data['e_park'] is not None:
            put_txt(f"Parking Target Error (e_park): {self.vis_data['e_park']:.1f} px")
        if 'ta_des' in self.vis_data:
            put_txt(f"Desired Trailer Target (ta_des): {self.vis_data['ta_des']:.1f} px")
        if 'steer' in self.vis_data:
            put_txt(f"Command Steering Signal: {self.vis_data['steer']}", (0, 255, 255))
            
        # 내부 포함 체크 판정 가시화 출력
        lin_flag = self.vis_data.get('l_in', False)
        rin_flag = self.vis_data.get('r_in', False)
        put_txt(f"Polygon In-Check -> Left: {lin_flag} | Right: {rin_flag}", (200, 255, 200) if (lin_flag and rin_flag) else (200, 200, 255))

        # 좌우 선제적 소실 감지 데드존 마진 가이드 선
        cv2.line(canvas, (int(self.edge_margin), 0), (int(self.edge_margin), h), (0, 0, 150), 1)
        cv2.line(canvas, (w - int(self.edge_margin), 0), (w - int(self.edge_margin), h), (0, 0, 150), 1)

        # 모니터 윈도우 생성 가동
        cv2.imshow("Pixel-space Autonomous Parking Monitor", canvas)
        cv2.waitKey(1)

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
        cv2.destroyAllWindows() # 👈 가시화 창 리소스 반환 안전 차단 추가
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()