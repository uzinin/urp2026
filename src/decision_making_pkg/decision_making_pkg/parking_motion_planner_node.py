import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy

import math
from std_msgs.msg import String, Bool, Float32  # 포텐쇼미터 데이터용 Float32
from interfaces_pkg.msg import PathPlanningResult, DetectionArray, MotionCommand
from .lib import decision_making_func_lib as DMFL

#---------------Variable Setting---------------
SUB_DETECTION_TOPIC_NAME = "detections"
SUB_PATH_TOPIC_NAME = "path_planning_result"
SUB_HITCH_TOPIC_NAME = "/articulation/angle"    # serial_sender_node 포텐쇼미터 각도 토픽
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
K_LATERAL = 1.5        # 픽셀 오차 각도를 목표 꺾임각으로 변환하는 게인 (실차 튜닝 필요)
K_HITCH = 0.6          # 꺾임각 오차를 조향 스텝으로 변환하는 게인
JACKKNIFE_LIMIT = 30.0  # 잭나이프 현상 방지 한계 각도 (도 단위)
REVERSE_SPEED =  -30 # 후진 구동 속도 (하드웨어 모터 스펙에 맞게 부호/크기 설정)


class MotionPlanningNode(Node):
    def __init__(self):
        super().__init__('motion_planner_node')

        # 토픽 이름 설정
        self.sub_detection_topic = self.declare_parameter('sub_detection_topic', SUB_DETECTION_TOPIC_NAME).value
        self.sub_path_topic = self.declare_parameter('sub_lane_topic', SUB_PATH_TOPIC_NAME).value
        self.sub_hitch_topic = self.declare_parameter('sub_hitch_topic', SUB_HITCH_TOPIC_NAME).value  
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

        # 서브스크라이버 설정
        self.detection_sub = self.create_subscription(DetectionArray, self.sub_detection_topic, self.detection_callback, self.qos_profile)
        self.path_sub = self.create_subscription(PathPlanningResult, self.sub_path_topic, self.path_callback, self.qos_profile)
        self.hitch_sub = self.create_subscription(Float32, self.sub_hitch_topic, self.hitch_callback, self.qos_profile) 
        # self.traffic_light_sub = self.create_subscription(String, self.sub_traffic_light_topic, self.traffic_light_callback, self.qos_profile)
        # self.lidar_sub = self.create_subscription(Bool, self.sub_lidar_obstacle_topic, self.lidar_callback, self.qos_profile)

        # 퍼블리셔 설정
        self.publisher = self.create_publisher(MotionCommand, self.pub_topic, self.qos_profile)

        # 타이머 설정
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    def detection_callback(self, msg: DetectionArray):
        self.detection_data = msg

    def path_callback(self, msg: PathPlanningResult):
        self.path_data = list(zip(msg.x_points, msg.y_points))

    def hitch_callback(self, msg: Float32):
        self.current_hitch_angle = msg.data
                
    def timer_callback(self):
        if self.path_data is None:
            return
           
        # 🌟 [긴급 진단 출력] 패스의 처음, 중간, 끝점의 생동작 픽셀 값을 찍어봅니다.
        self.get_logger().info(f"===[DATA CHECK]=== Len: {len(self.path_data)} | "
                               f"Index 0: {self.path_data[0]} | "
                               f"Index 15: {self.path_data[min(15, len(self.path_data)-1)]} | "
                               f"Index Last: {self.path_data[-1]}")
        
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
    
        # 2. 정상 트레일러 동적 패스 추종 가동
        else:
            self.cnt_dead = 0

            # [안전장치] 잭나이프 한계 도달 시 즉시 정지
            if abs(self.current_hitch_angle) > JACKKNIFE_LIMIT:
                self.get_logger().error(f"🚨 JACKKNIFE WARNING! Angle: {self.current_hitch_angle:.1f}°")
                self.steering_command = 0
                self.left_speed_command = 0
                self.right_speed_command = 0
                self.publish_motion_command()
                return

            # ----------------------------------------------------------------------
            # [외곽 루프 (Outer Loop)] 🌟 유동적 Lookahead로 역조향 타이밍 확보
            # ----------------------------------------------------------------------
            origin_x = self.path_data[0][0]
            origin_y = self.path_data[0][1]

            # 🔥 교정 1: 남은 패스 길이에 비례하여 Lookahead를 유동적으로 결정합니다.
            # 멀리 있을 땐 끝점(남은 길이의 90%), 가까워지면 코앞을 보게 하여 역조향을 유도합니다.
            path_len = len(self.path_data)
            LOOKAHEAD_INDEX = int(path_len * 0.85)
            LOOKAHEAD_INDEX = max(5, min(path_len - 1, LOOKAHEAD_INDEX)) # 최소 5개 앞은 보되 배열 범위 안으로 제한
           
            target_point = self.path_data[LOOKAHEAD_INDEX]
           
            dx = target_point[0] - origin_x
            dy = origin_y - target_point[1]  

            if abs(dy) > 1e-5:
                pure_angle = math.degrees(math.atan2(dx, dy))
                # 편차 보상 게인을 0.05에서 0.03으로 살짝 낮춰 오버슈트(과도하게 꺾임)를 방지합니다.
                target_angle = (pure_angle + (dx * 0.05)) + (self.current_hitch_angle * 0.4)
            else:
                target_angle = 0.0

            # 🔥 교정 2: 대각선 진입 시 너무 깊게 접히는 것을 막기 위해
            # K_LATERAL은 유지하되, 최대 목표 꺾임각(gamma_ref) 한계를 20도 -> 14도로 줄입니다.
            # 이렇게 해야 트레일러가 적당히 접히고, 내부 루프가 쉽게 역조향으로 풀 수 있습니다.
            K_LATERAL_TUNED = 1.5
            gamma_ref = K_LATERAL_TUNED * target_angle
            gamma_ref = max(-14.0, min(14.0, gamma_ref))

            # ----------------------------------------------------------------------
            # [내부 루프 (Inner Loop)] 🌟 포텐쇼미터 피드백 강화 (명령에 칼같이 반응)
            # ----------------------------------------------------------------------
            hitch_error = gamma_ref - self.current_hitch_angle
       
            # LPF (기존 유지)
            self.target_slope_f = (1 - ALPHA) * self.target_slope_f + ALPHA * hitch_error
           
            # 🔥 교정 3: K_HITCH 게인을 0.6 -> 0.85로 상향합니다.
            # 외곽 루프의 꺾임 명령보다 포텐쇼미터의 '현재 오차를 줄이려는 힘'을 키워서
            # 제때 핸들을 반대로 쳐서 트레일러를 펴주도록 만듭니다.
            K_HITCH_TUNED = 0.85
            step_f = K_HITCH_TUNED * self.target_slope_f
            step = int(round(step_f))
            step = max(-MAX_STEP, min(MAX_STEP, step))
           
            # Slew Rate Limiter (기존 유지)
            step = max(self.prev_step - MAX_STEP_DELTA, min(self.prev_step + MAX_STEP_DELTA, step))
            self.prev_step = step
            self.steering_command = step

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
