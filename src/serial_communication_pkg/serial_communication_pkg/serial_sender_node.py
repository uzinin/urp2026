import time
import serial
import rclpy

from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy

from std_msgs.msg import Int32
from std_msgs.msg import Float32

from interfaces_pkg.msg import MotionCommand
from .lib import protocol_convert_func_lib as PCFL


#---------------Variable Setting---------------
# Subscribe할 토픽 이름
SUB_TOPIC_NAME = "topic_control_signal"

# 아두이노 장치 이름
PORT = '/dev/ttyACM0'
BAUD_RATE = 115200

# 굴절부 토픽 이름
ARTICULATION_RAW_TOPIC = "/articulation/potentiometer_raw"
ARTICULATION_ANGLE_TOPIC = "/articulation/angle"
#----------------------------------------------


class SerialSenderNode(Node):
    def __init__(self, sub_topic=SUB_TOPIC_NAME):
        super().__init__('serial_sender_node')

        self.declare_parameter('sub_topic', sub_topic)
        self.declare_parameter('port', PORT)
        self.declare_parameter('baud_rate', BAUD_RATE)

        # 굴절각 보정 파라미터
        # 실제 차량에서 측정 후 수정하세요.
        self.declare_parameter('adc_min', 60)
        self.declare_parameter('adc_max', 240)
        self.declare_parameter('angle_min_deg', -40.0)
        self.declare_parameter('angle_max_deg', 40.0)

        self.sub_topic = self.get_parameter('sub_topic').get_parameter_value().string_value
        self.port = self.get_parameter('port').get_parameter_value().string_value
        self.baud_rate = self.get_parameter('baud_rate').get_parameter_value().integer_value

        self.adc_min = self.get_parameter('adc_min').get_parameter_value().integer_value
        self.adc_max = self.get_parameter('adc_max').get_parameter_value().integer_value
        self.angle_min_deg = self.get_parameter('angle_min_deg').get_parameter_value().double_value
        self.angle_max_deg = self.get_parameter('angle_max_deg').get_parameter_value().double_value

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        # Serial 연결
        self.ser = serial.Serial(self.port, self.baud_rate, timeout=0.01)
        time.sleep(1)

        self.get_logger().info(f"Connected to Arduino: {self.port}")

        # 주행 명령 Subscribe
        self.subscription = self.create_subscription(
            MotionCommand,
            self.sub_topic,
            self.data_callback,
            qos_profile
        )

        # 굴절부 가변저항 Publish
        self.articulation_raw_pub = self.create_publisher(
            Int32,
            ARTICULATION_RAW_TOPIC,
            qos_profile
        )

        self.articulation_angle_pub = self.create_publisher(
            Float32,
            ARTICULATION_ANGLE_TOPIC,
            qos_profile
        )

        # Arduino에서 들어오는 Serial 데이터 읽기용 타이머
        self.serial_read_timer = self.create_timer(0.01, self.read_serial_callback)

    def data_callback(self, msg):
        steering = msg.steering
        left_speed = msg.left_speed
        right_speed = msg.right_speed

        serial_msg = PCFL.convert_serial_message(
            steering,
            left_speed,
            right_speed
        )

        self.ser.write(serial_msg.encode())

    def adc_to_angle(self, adc_value):
        """
        굴절부 ADC 값을 각도(degree)로 변환.
        현재는 선형 변환 방식.
        """

        if adc_value < self.adc_min:
            adc_value = self.adc_min
        elif adc_value > self.adc_max:
            adc_value = self.adc_max

        ratio = (adc_value - self.adc_min) / (self.adc_max - self.adc_min)

        angle_deg = self.angle_min_deg + ratio * (
            self.angle_max_deg - self.angle_min_deg
        )

        return angle_deg

    def read_serial_callback(self):
        """
        Arduino에서 들어오는 데이터를 읽음.
        Arduino가 a512 형태로 보내면 굴절부 센서값으로 처리.
        """

        try:
            while self.ser.in_waiting > 0:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()

                if line == "":
                    return

                # 굴절부 가변저항 데이터 예: a512
                if line.startswith("a"):
                    adc_value = int(line[1:])

                    raw_msg = Int32()
                    raw_msg.data = adc_value
                    self.articulation_raw_pub.publish(raw_msg)

                    angle_deg = self.adc_to_angle(adc_value)

                    angle_msg = Float32()
                    angle_msg.data = float(angle_deg)
                    self.articulation_angle_pub.publish(angle_msg)

                    self.get_logger().info(
                        f"Articulation ADC: {adc_value}, angle: {angle_deg:.2f} deg"
                    )

                else:
                    self.get_logger().debug(f"Arduino message: {line}")

        except ValueError:
            self.get_logger().warn(f"Invalid articulation data: {line}")

        except Exception as e:
            self.get_logger().error(f"Serial read error: {e}")

    def stop_vehicle(self):
        steering = 0
        left_speed = 0
        right_speed = 0

        message = PCFL.convert_serial_message(
            steering,
            left_speed,
            right_speed
        )

        self.ser.write(message.encode())

    def close_serial(self):
        if self.ser.is_open:
            self.ser.close()
            self.get_logger().info("Serial port closed")


def main(args=None):
    rclpy.init(args=args)

    node = SerialSenderNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        print("\n\nshutdown\n\n")
        node.stop_vehicle()

    finally:
        node.close_serial()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()




# import time
# import serial
# import rclpy
# from rclpy.node import Node
# from rclpy.qos import QoSProfile
# from rclpy.qos import QoSHistoryPolicy
# from rclpy.qos import QoSDurabilityPolicy
# from rclpy.qos import QoSReliabilityPolicy
# from interfaces_pkg.msg import MotionCommand
# from .lib import protocol_convert_func_lib as PCFL

# #---------------Variable Setting---------------
# # Subscribe할 토픽 이름
# SUB_TOPIC_NAME = "topic_control_signal"

# # 아두이노 장치 이름 (ls /dev/ttyA* 명령을 터미널 창에 입력하여 확인)
# PORT='/dev/ttyACM0'
# #----------------------------------------------

# ser = serial.Serial(PORT, 115200, timeout=1)
# time.sleep(1)

# class SerialSenderNode(Node):
#   def __init__(self, sub_topic=SUB_TOPIC_NAME):
#     super().__init__('serial_sender_node')
    
#     self.declare_parameter('sub_topic', sub_topic)
    
#     self.sub_topic = self.get_parameter('sub_topic').get_parameter_value().string_value
    
#     qos_profile = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, 
#                              history=QoSHistoryPolicy.KEEP_LAST, 
#                              durability=QoSDurabilityPolicy.VOLATILE, 
#                              depth=1)
    
#     self.subscription = self.create_subscription(MotionCommand, self.sub_topic, self.data_callback, qos_profile)

#   def data_callback(self, msg):
#     steering = msg.steering
#     left_speed = msg.left_speed
#     right_speed = msg.right_speed

#     serial_msg =  PCFL.convert_serial_message(steering, left_speed, right_speed)
#     ser.write(serial_msg.encode())

# def main(args=None):
#   rclpy.init(args=args)
#   node = SerialSenderNode()
#   try:
#       rclpy.spin(node)
      
#   except KeyboardInterrupt:
#       print("\n\nshutdown\n\n")
#       steering = 0
#       left_speed = 0
#       right_speed = 0
#       message = PCFL.convert_serial_message(steering, left_speed, right_speed)
#       ser.write(message.encode())
#       pass
    
#   finally:
#     ser.close()
#     print('closed')
    
#   node.destroy_node()
#   rclpy.shutdown()
  
# if __name__ == '__main__':
#   main()
