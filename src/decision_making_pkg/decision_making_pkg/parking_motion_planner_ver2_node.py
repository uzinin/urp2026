import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy

import math
import time
from std_msgs.msg import String, Bool, Float32  # 포텐쇼미터 데이터용 Float32
from interfaces_pkg.msg import PathPlanningResult, DetectionArray, MotionCommand
from .lib import decision_making_func_lib as DMFL

#---------------Variable Setting---------------
SUB_DETECTION_TOPIC_NAME = "detections"
SUB_PATH_TOPIC_NAME = "path_planning_result"
SUB_HITCH_TOPIC_NAME = "/articulation/angle"    # serial_sender_node 포텐쇼미터 각도 토픽
SUB_PARKING_COMPLETE_TOPIC_NAME = "/parking/complete"
SUB_TRAFFIC_LIGHT_TOPIC_NAME = "yolov8_traffic_light_info"
SUB_LIDAR_OBSTACLE_TOPIC_NAME = "lidar_obstacle_info"
PUB_TOPIC_NAME = "topic_control_signal"

#----------------------------------------------

# 모션 플랜 발행 주기 (초)
TIMER = 0.1

# 기본 CONSTANTS (기존 제어 틀 유지)
MAX_STEP = 7
THETA_MAX_DEG = 75.0  
ALPHA = 0.3
MAX_STEP_DELTA = 1

# 🌟 트레일러 픽셀 뷰 후진 전용 제어 게인 및 안전 한계선
K_LATERAL = 0.5        # 픽셀 오차 각도를 목표 꺾임각으로 변환하는 게인 (실차 튜닝 필요)
K_HITCH = 0.6          # 꺾임각 오차를 조향 스텝으로 변환하는 게인
JACKKNIFE_LIMIT = 35.0  # 잭나이프 현상 방지 한계 각도 (도 단위)
REVERSE_SPEED = -30   # 후진 구동 속도 (하드웨어 모터 스펙에 맞게 부호/크기 설정)

# 🌟 데드 레코닝 설정을 위한 파라미터
PARKING_NEAR_THRESHOLD_Y = 380   # 주차 영역이 코앞에 도달했다고 판단하는 픽셀 기준 (사각지대 진입 직전)
BLIND_ZONE_DRIVE_DURATION = 1.5  # 사각지대에 가려진 후, 주차 칸에 딱 맞게 들어가기 위해 추가로 후진할 시간 (초 단위)

class MotionPlanningNode(Node):
    def __init__(self):
        super().__init__('motion_planner_node')

        # 토픽 이름 설정
        self.sub_detection_topic = self.declare_parameter('sub_detection_topic', SUB_DETECTION_TOPIC_NAME).value
        self.sub_path_topic = self.declare_parameter('sub_lane_topic', SUB_PATH_TOPIC_NAME).value
        self.sub_hitch_topic = self.declare_parameter('sub_hitch_topic', SUB_HITCH_TOPIC_NAME).value  
        self.sub_parking_complete_topic = self.declare_parameter(
            'sub_parking_complete_topic',
            SUB_PARKING_COMPLETE_TOPIC_NAME
        ).value
        # self.sub_traffic_light_topic = self.declare_parameter('sub_traffic_light_topic', SUB_TRAFFIC_LIGHT_TOPIC_NAME).value
        # self.sub_lidar_obstacle_topic = self.declare_parameter('sub_lidar_obstacle_topic', SUB_LIDAR_OBSTACLE_TOPIC_NAME).value
        self.pub_topic = self.declare_parameter('pub_topic', PUB_TOPIC_NAME).value
        
        self.timer_period = self.declare_parameter('timer', TIMER).value

        # QoS 설정
        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        # 변수 초기화
        self.detection_data = None
        self.path_data = None
        # self.traffic_light_data = None
        # self.lidar_data = None

        self.steering_command = 0
        self.left_speed_command = 0
        self.right_speed_command = 0

        # 추가 변수 설정
        self.current_hitch_angle = 0.0  # 실시간 포텐쇼미터 데이터 (도 단위)
        self.target_slope_f = 0.0
        self.prev_step = 0
        self.cnt_dead = 0

        # 🌟 데드 레코닝 제어용 상태 변수들
        self.parking_zone_near = False       # 사각지대 진입 직전(매우 가까움) 상태 플래그
        self.dead_reckoning_started = False  # 사각지대 진입 후 타이머 카운트 시작 플래그
        self.blind_start_time = None         # 사각지대에 들어간 시점의 타임스탬프
        self.is_parking_completed = False    # 최종 주차 완료(영구 정지) 플래그

        # 서브스크라이버 설정
        self.detection_sub = self.create_subscription(DetectionArray, self.sub_detection_topic, self.detection_callback, self.qos_profile)
        self.path_sub = self.create_subscription(PathPlanningResult, self.sub_path_topic, self.path_callback, self.qos_profile)
        self.hitch_sub = self.create_subscription(Float32, self.sub_hitch_topic, self.hitch_callback, self.qos_profile) 
        self.parking_complete_sub = self.create_subscription(Bool, self.sub_parking_complete_topic, self.parking_complete_callback, self.qos_profile)
        # self.traffic_light_sub = self.create_subscription(String, self.sub_traffic_light_topic, self.traffic_light_callback, self.qos_profile)
        # self.lidar_sub = self.create_subscription(Bool, self.sub_lidar_obstacle_topic, self.lidar_callback, self.qos_profile)

        # 퍼블리셔 설정
        self.publisher = self.create_publisher(MotionCommand, self.pub_topic, self.qos_profile)

        # 타이머 설정
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    def detection_callback(self, msg: DetectionArray):
        self.detection_data = msg
        # 이번 루프에서 주차 영역이 인식되었는지 확인할 변수
        detected_now = False
        
        for det in msg.detections:
            if det.class_name == "red_parking_zone":
                detected_now = True
                # 사각지대에 들어가기 전, 차량 범퍼 바로 위에 주차선이 도달했는지 검사
                y_max = det.bbox.center.position.y + det.bbox.size.y / 2
                if y_max >= PARKING_NEAR_THRESHOLD_Y:
                    self.parking_zone_near = True
                break
        
        # 🌟 [데드 레코닝 핵심 로직] 
        # 직전까지는 코앞에 주차 구역이 있었는데(parking_zone_near), 지금 순간에 인식이 끊겼다면(detected_now == False)
        # 사각지대에 들어간 것으로 판단하고 타이머를 작동시킵니다.
        if self.parking_zone_near and not detected_now and not self.dead_reckoning_started:
            self.dead_reckoning_started = True
            self.blind_start_time = time.time()
            self.get_logger().warn("⚠️ [BLIND ZONE DETECTED] 주차 구역이 사각지대에 가려짐. 데드 레코닝(추정 항법) 후진 시작!")

    def path_callback(self, msg: PathPlanningResult):
        self.path_data = list(zip(msg.x_points, msg.y_points))

    def hitch_callback(self, msg: Float32):
        self.current_hitch_angle = msg.data

    def parking_complete_callback(self, msg: Bool):
        if msg.data and not self.is_parking_completed:
            self.is_parking_completed = True
            self.steering_command = 0
            self.left_speed_command = 0
            self.right_speed_command = 0
            self.get_logger().info("[PARKING COMPLETE] parking_path_planner stop signal received.")
            self.publish_motion_command()
                
    def timer_callback(self):
        # 🌟 1단계: 최종 주차 완료 상태라면 즉시 정지유지
        if self.is_parking_completed:
            self.steering_command = 0
            self.left_speed_command = 0
            self.right_speed_command = 0
            self.publish_motion_command()
            return

        # 🌟 2단계: 사각지대 진입 후 데드 레코닝 주행 상태인지 체크
        if self.dead_reckoning_started:
            elapsed_time = time.time() - self.blind_start_time
            
            # 설정한 시간(예: 1.5초)이 지나기 전까지는 핸들을 똑바로 펴고 막바지 후진을 유지
            if elapsed_time < BLIND_ZONE_DRIVE_DURATION:
                self.get_logger().info(f"⏳ [DEAD RECKONING] 목표 도달까지 추가 후진 중... ({elapsed_time:.1f}s / {BLIND_ZONE_DRIVE_DURATION}s)")
                self.steering_command = 0 # 마지막 주차 정렬을 위해 핸들을 직선(0)으로 고정하거나 상황에 맞게 제어 가능
                self.left_speed_command = REVERSE_SPEED
                self.right_speed_command = REVERSE_SPEED
                self.publish_motion_command()
                return
            else:
                # 지정된 시간을 모두 채웠다면 완전히 주차가 완료된 것으로 간주
                self.is_parking_completed = True
                self.get_logger().info("🎯 [PARKING COMPLETE] 데드 레코닝 완료. 차량을 정지합니다.")
                self.steering_command = 0
                self.left_speed_command = 0
                self.right_speed_command = 0
                self.publish_motion_command()
                return
        # 1. 경로 데이터가 아예 없는 경우 (안전을 위해 즉시 정지)
        if self.path_data is None:
            self.steering_command = 0
            self.left_speed_command = 0
            self.right_speed_command = 0
            self.get_logger().warn("---------Path data none!!!---------")
            self.publish_motion_command()
            return
    
        # 경로 점이 너무 부족한 경우 (Lookahead 타겟팅 불가하므로 안전 속도로 직선 후진)
        elif len(self.path_data) < 10:
            self.cnt_dead += 1
            if self.cnt_dead > 30:
                self.get_logger().info("Dead reckoning mode - Safe Straight Reversing")
                self.steering_command = 0
                self.left_speed_command = REVERSE_SPEED
                self.right_speed_command = REVERSE_SPEED
                self.publish_motion_command()
            return
    
        # 2. 정상 트레일러 동적 패스 추종 및 이중 루프 제어 가동
        else:
            self.cnt_dead = 0

            # 🚨 [비상 안전장치] 잭나이프 한계 도달 시 즉시 정지하여 물리적 파손 방지
            if abs(self.current_hitch_angle) > JACKKNIFE_LIMIT:
                self.get_logger().error(f"🚨 JACKKNIFE WARNING! Angle: {self.current_hitch_angle:.1f}°")
                self.steering_command = 0
                self.left_speed_command = 0
                self.right_speed_command = 0
                self.publish_motion_command()
                return

            # ----------------------------------------------------------------------
            # [외곽 루프 (Outer Loop)] 🌟 이미지 픽셀 매칭형 목표 각도 추출
            # ----------------------------------------------------------------------
            # 패스의 시작점 (트레일러 본네트 노란 점 픽셀)
            origin_x = self.path_data[0][0]
            origin_y = self.path_data[0][1]

            # 가이드라인 상에서 쫓아갈 앞쪽 목표점 선정 (15번째 픽셀 점 타겟팅)
            LOOKAHEAD_INDEX = min(15, len(self.path_data) - 1)
            target_point = self.path_data[LOOKAHEAD_INDEX]
            
            # 픽셀 좌표 오차 계산
            dx = target_point[0] - origin_x
            # 픽셀 좌표계는 상단이 0이므로, 전방 진행 방향을 양수(+)로 만들기 위해 부호 반전
            dy = origin_y - target_point[1]  

            if abs(dy) > 1e-5:
                # 🔄 트레일러 후진 역조향 기하학 매핑:
                # 목표점이 우측(dx > 0)에 있으면 트레일러 뒷무릎을 우측으로 밀기 위해
                # 목표 꺾임각(gamma_ref)이 음수(-)가 되어 견인차가 왼쪽으로 꺾도록 마이너스(-) 부착
                target_angle = -math.degrees(math.atan2(dx, dy))
            else:
                target_angle = 0.0

            # 미세 픽셀 흔들림 필터링 데드존 (1.5도 미만 컷)
            if abs(target_angle) < 1.5:
                target_angle = 0.0

            # 목표 꺾임각(gamma_ref) 산출 및 과도한 접힘 방지 제한 (최대 15도)
            gamma_ref = K_LATERAL * target_angle
            gamma_ref = max(-15.0, min(15.0, gamma_ref))

            # ----------------------------------------------------------------------
            # [내부 루프 (Inner Loop)] 🌟 목표 꺾임각 추종을 위한 포텐쇼미터 피드백 제어
            # ----------------------------------------------------------------------
            hitch_error = gamma_ref - self.current_hitch_angle
        
            # 기존 8조 알고리즘의 조향 필터 메커니즘 원본 그대로 계승 (LPF 적용)
            self.target_slope_f = (1 - ALPHA) * self.target_slope_f + ALPHA * hitch_error
            
            # P 제어 기반 최종 조향 스텝 명령 산출
            step_f = K_HITCH * self.target_slope_f
            step = int(round(step_f))
            step = max(-MAX_STEP, min(MAX_STEP, step))
            
            # Slew Rate Limiter (갑작스러운 핸들 털림 및 잭나이프 가속 방지)
            step = max(self.prev_step - MAX_STEP_DELTA, min(self.prev_step + MAX_STEP_DELTA, step))
            self.prev_step = step
            self.steering_command = step

            # 속도 결정 (안전 후진 상수 고정)
            self.left_speed_command = REVERSE_SPEED
            self.right_speed_command = REVERSE_SPEED

        # 실시간 제어 데이터 로깅
        self.get_logger().info(f"[PIXEL TRAILER] TargetAng: {target_angle:.1f}° | "
                               f"GammaRef: {gamma_ref:.1f}° | Hitch: {self.current_hitch_angle:.1f}° | "
                               f"SteerStep: {self.steering_command}")

        self.publish_motion_command()

    def publish_motion_command(self):
        # 모션 명령 메시지 생성 및 퍼블리시
        motion_command_msg = MotionCommand()
        motion_command_msg.steering = self.steering_command
        motion_command_msg.left_speed = self.left_speed_command
        motion_command_msg.right_speed = self.right_speed_command
        self.publisher.publish(motion_command_msg)


def main(args=None):
    rclpy.init(args=args)
    node = MotionPlanningNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n\nshutdown\n\n")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
