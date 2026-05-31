import time
from typing import Optional

import rclpy
import serial
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, QoSReliabilityPolicy
from std_msgs.msg import Float32, Int32

from interfaces_pkg.msg import MotionCommand
from .lib import protocol_convert_func_lib as PCFL


SUB_TOPIC_NAME = "topic_control_signal"
PORT = "/dev/ttyACM0"
BAUD_RATE = 115200

ARTICULATION_RAW_TOPIC = "/articulation/potentiometer_raw"
ARTICULATION_ANGLE_TOPIC = "/articulation/angle"


class SerialSenderNode(Node):
    """Send motion commands to Arduino and publish A0 articulation measurements."""

    def __init__(self, sub_topic: str = SUB_TOPIC_NAME) -> None:
        super().__init__("serial_sender_node")

        self.declare_parameter("sub_topic", sub_topic)
        self.declare_parameter("port", PORT)
        self.declare_parameter("baud_rate", BAUD_RATE)

        # Set these four calibration values after physically measuring A0.
        self.declare_parameter("articulation_adc_min", 200)
        self.declare_parameter("articulation_adc_max", 440)
        self.declare_parameter("articulation_angle_min_deg", -40.0)
        self.declare_parameter("articulation_angle_max_deg", 40.0)

        self.sub_topic = self.get_parameter("sub_topic").value
        self.port = self.get_parameter("port").value
        self.baud_rate = int(self.get_parameter("baud_rate").value)
        self.adc_min = int(self.get_parameter("articulation_adc_min").value)
        self.adc_max = int(self.get_parameter("articulation_adc_max").value)
        self.angle_min_deg = float(self.get_parameter("articulation_angle_min_deg").value)
        self.angle_max_deg = float(self.get_parameter("articulation_angle_max_deg").value)

        if self.adc_max <= self.adc_min:
            raise ValueError("articulation_adc_max must be greater than articulation_adc_min")

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=10,
        )

        self.ser = serial.Serial(self.port, self.baud_rate, timeout=0.0)
        time.sleep(1.0)  # Arduino may reset when the serial port is opened.
        self.ser.reset_input_buffer()
        self._serial_buffer = ""

        self.subscription = self.create_subscription(
            MotionCommand, self.sub_topic, self.data_callback, qos_profile
        )
        self.articulation_raw_pub = self.create_publisher(
            Int32, ARTICULATION_RAW_TOPIC, qos_profile
        )
        self.articulation_angle_pub = self.create_publisher(
            Float32, ARTICULATION_ANGLE_TOPIC, qos_profile
        )
        self.serial_read_timer = self.create_timer(0.01, self.read_serial_callback)

        self.get_logger().info(
            f"Connected to Arduino on {self.port} at {self.baud_rate} baud; "
            "receiving A0 articulation telemetry."
        )

    def data_callback(self, msg: MotionCommand) -> None:
        """Transmit steering and driving commands using the existing command protocol."""
        serial_msg = PCFL.convert_serial_message(
            msg.steering, msg.left_speed, msg.right_speed
        )
        if not serial_msg.endswith("\n"):
            serial_msg += "\n"
        try:
            self.ser.write(serial_msg.encode("ascii"))
        except serial.SerialException as exc:
            self.get_logger().error(f"Serial write error: {exc}")

    def adc_to_angle(self, adc_value: int) -> float:
        adc_limited = max(self.adc_min, min(self.adc_max, adc_value))
        ratio = (adc_limited - self.adc_min) / (self.adc_max - self.adc_min)
        return self.angle_min_deg + ratio * (self.angle_max_deg - self.angle_min_deg)+8

    def publish_articulation(self, line: str) -> None:
        """Parse Arduino telemetry formatted as a<ADC>, for example a512."""
        if not line.startswith("a"):
            self.get_logger().debug(f"Ignored Arduino message: {line}")
            return

        payload = line[1:].strip()
        try:
            adc_value = int(payload)
        except ValueError:
            self.get_logger().warning(f"Invalid articulation packet: {line!r}")
            return

        if not 0 <= adc_value <= 1023:
            self.get_logger().warning(f"Out-of-range articulation ADC value: {adc_value}")
            return

        raw_msg = Int32()
        raw_msg.data = adc_value
        self.articulation_raw_pub.publish(raw_msg)

        angle_msg = Float32()
        angle_msg.data = float(self.adc_to_angle(adc_value))
        self.articulation_angle_pub.publish(angle_msg)

    def read_serial_callback(self) -> None:
        """Read complete newline-terminated packets without parsing partial lines."""
        try:
            waiting = self.ser.in_waiting
            if waiting <= 0:
                return

            self._serial_buffer += self.ser.read(waiting).decode("ascii", errors="ignore")
            while "\n" in self._serial_buffer:
                line, self._serial_buffer = self._serial_buffer.split("\n", 1)
                line = line.strip("\r ")
                if line:
                    self.publish_articulation(line)
        except serial.SerialException as exc:
            self.get_logger().error(f"Serial read error: {exc}")

    def stop_vehicle(self) -> None:
        if not self.ser.is_open:
            return
        message = PCFL.convert_serial_message(0, 0, 0)
        if not message.endswith("\n"):
            message += "\n"
        self.ser.write(message.encode("ascii"))

    def close_serial(self) -> None:
        if self.ser.is_open:
            self.ser.close()
            self.get_logger().info("Serial port closed")


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node: Optional[SerialSenderNode] = None
    try:
        node = SerialSenderNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node is not None:
            node.stop_vehicle()
    finally:
        if node is not None:
            node.close_serial()
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
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
