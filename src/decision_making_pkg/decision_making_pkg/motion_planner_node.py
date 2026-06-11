import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy

from std_msgs.msg import String, Bool
from interfaces_pkg.msg import PathPlanningResult, DetectionArray, MotionCommand
from .lib import decision_making_func_lib as DMFL

#---------------Variable Setting---------------
SUB_DETECTION_TOPIC_NAME = "detections"
SUB_PATH_TOPIC_NAME = "path_planning_result"
SUB_TRAFFIC_LIGHT_TOPIC_NAME = "yolov8_traffic_light_info"
SUB_LIDAR_OBSTACLE_TOPIC_NAME = "lidar_obstacle_info"
PUB_TOPIC_NAME = "topic_control_signal"

#----------------------------------------------

# 모션 플랜 발행 주기 (초) - 소수점 필요 (int형은 반영되지 않음)
TIMER = 0.1

# 추가 CONSTANTS 선언
MAX_STEP = 7
THETA_MAX_DEG = 75.0
ALPHA = 0.3
MAX_STEP_DELTA = 2
MAX_SPEED = 100
MIN_SPEED = 100


class MotionPlanningNode(Node):
    def __init__(self):
        super().__init__('motion_planner_node')

        # 토픽 이름 설정
        self.sub_detection_topic = self.declare_parameter('sub_detection_topic', SUB_DETECTION_TOPIC_NAME).value
        self.sub_path_topic = self.declare_parameter('sub_lane_topic', SUB_PATH_TOPIC_NAME).value
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

        # 추가 변수 설정 (8조)
        self.target_slope_f = 0
        self.prev_step = 0
        self.cnt_dead = 0

        # 서브스크라이버 설정
        self.detection_sub = self.create_subscription(DetectionArray, self.sub_detection_topic, self.detection_callback, self.qos_profile)
        self.path_sub = self.create_subscription(PathPlanningResult, self.sub_path_topic, self.path_callback, self.qos_profile)
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
                
    # def traffic_light_callback(self, msg: String):
    #     self.traffic_light_data = msg

    # def lidar_callback(self, msg: Bool):
        # self.lidar_data = msg
        
    def timer_callback(self):
        # 1. 경로 데이터가 아예 없는 경우 (안전을 위해 정지)
        if self.path_data is None:
            self.steering_command = 0
            self.left_speed_command = 0
            self.right_speed_command = 0
            self.get_logger().warn("---------Path data none!!!---------")
    
        # 2. 경로 데이터가 부족한 경우 (데드 레코닝 또는 정지)
        elif len(self.path_data) < 3:
            self.cnt_dead += 1
            if self.cnt_dead > 30:
                self.get_logger().info("Dead reckoning mode")
                self.steering_command = 0
                self.left_speed_command = 100
                self.right_speed_command = 100
    
    # 3. 정상 경로 추종 (8조 알고리즘 작동)
        else:
            self.cnt_dead = 0
            target_slope = DMFL.calculate_slope_between_points(self.path_data[-10], self.path_data[-1])
        
            # [8조 제어 로직 적용]
            # Dead Zone 설정
            if abs(target_slope) < 1.0:
                    target_slope = 0.0

                # LPF (Low Pass Filter)
            self.target_slope_f = (1 - ALPHA) * self.target_slope_f + ALPHA * target_slope
            
            # 단위 변환 및 스텝 제한
            step_f = (self.target_slope_f / THETA_MAX_DEG) * MAX_STEP
            step = int(round(step_f))
            step = max(-MAX_STEP, min(MAX_STEP, step))
            
            # 변화율 제한 (Slew Rate Limiter)
            step = max(self.prev_step - MAX_STEP_DELTA, min(self.prev_step + MAX_STEP_DELTA, step))
            self.prev_step = step
            self.steering_command = step

                # 속도 결정 (기본 주행)
            tmp_speed = MAX_SPEED - abs(self.target_slope_f) / THETA_MAX_DEG * (MAX_SPEED - MIN_SPEED)
            self.left_speed_command = int(max(MIN_SPEED, tmp_speed))
            self.right_speed_command = int(max(MIN_SPEED, tmp_speed))





        # if self.lidar_data is not None and self.lidar_data.data is True:
            # # 라이다가 장애물을 감지한 경우
            # self.steering_command = 0 
            # self.left_speed_command = 0 
            # self.right_speed_command = 0 

        # elif self.traffic_light_data is not None and self.traffic_light_data.data == 'Red':
        #     # 빨간색 신호등을 감지한 경우
        #     for detection in self.detection_data.detections:
        #         if detection.class_name=='traffic_light':
        #             x_min = int(detection.bbox.center.position.x - detection.bbox.size.x / 2) # bbox의 좌측상단 꼭짓점 x좌표
        #             x_max = int(detection.bbox.center.position.x + detection.bbox.size.x / 2) # bbox의 우측하단 꼭짓점 x좌표
        #             y_min = int(detection.bbox.center.position.y - detection.bbox.size.y / 2) # bbox의 좌측상단 꼭짓점 y좌표
        #             y_max = int(detection.bbox.center.position.y + detection.bbox.size.y / 2) # bbox의 우측하단 꼭짓점 y좌표

        #             # if y_max < 150:
        #                 # 신호등 위치에 따른 정지명령 결정
        #             self.steering_command = 0 
        #             self.left_speed_command = 0 
        #             self.right_speed_command = 0
        # else:
        #     if self.path_data is None:
        #         self.steering_command = 0
        #         self.get_logger().warn("---------Path data none!!!---------")
        #     elif len(self.path_data) < 3:
        #         self.cnt_dead += 1
        #         if self.cnt_dead > 30:
        #             self.get_logger().info("dead recogning")
        #             self.steering_command = 0
        #             self.left_speed_command = 100
        #             self.right_speed_command = 100
        #     else:
        #         self.cnt_dead = 0
        #         self.get_logger().info("Path ok!!!!!")
        #         target_slope = DMFL.calculate_slope_between_points(self.path_data[-10], self.path_data[-1])
        #         self.get_logger().info(f"Calculated slope: {target_slope}")
                
        #         # if target_slope > 0:
        #         #     self.steering_command =  7 # 예시 조향 값 (7이 최대 조향) 
        #         # elif target_slope < 0:
        #         #     self.steering_command =  -7
        #         # else:
        #         #     self.steering_command = 0

        #         # 8조 경로 추종 알고리즘
        #         # 1) Dead Zone 설정, -1 ~ 1도 사이 값은 각도 0으로 고정
        #         if abs(target_slope) < 1.0:
        #             target_slope = 0.0

        #         # 2) 기울기 값 Low pass 필터
        #         self.target_slope_f = (1-ALPHA) * self.target_slope_f + ALPHA * target_slope
                
        #         # 3) 1차원 단위 변환 (slope to steer_command)
        #         step_f = (self.target_slope_f / THETA_MAX_DEG) * MAX_STEP
        #         step = int(round(step_f))
        #         step = max(-MAX_STEP, min(MAX_STEP, step))

        #         # 4) step 변화율 제한
        #         step = max(self.prev_step - MAX_STEP_DELTA, min(self.prev_step + MAX_STEP_DELTA, step))
        #         self.prev_step = step

        #         # 5) steer_command 할당
        #         # self.steering_command = step + MAX_STEP + 1
        #         self.steering_command = step

        #     # 8조 노란불의 경우 속도
        #     if self.traffic_light_data is not None and self.traffic_light_data.data == 'Yellow':
        #         self.left_speed_command = 50  # 예시 속도 값 (255가 최대 속도)
        #         self.right_speed_command = 50  # 예시 속도 값 (255가 최대 속도)
        #     else:
        #         tmp_speed = MAX_SPEED - abs(self.target_slope_f) / THETA_MAX_DEG * (MAX_SPEED - MIN_SPEED)
        #         if tmp_speed < MIN_SPEED:
        #             tmp_speed = MIN_SPEED
        #         self.left_speed_command = int(tmp_speed)   # 예시 속도 값 (255가 최대 속도)
        #         self.right_speed_command = int(tmp_speed)  # 예시 속도 값 (255가 최대 속도)


        self.get_logger().info(f"steering: {self.steering_command}, " 
                               f"left_speed: {self.left_speed_command}, " 
                               f"right_speed: {self.right_speed_command}")

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
