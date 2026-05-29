#include <Arduino.h>
#include <stdlib.h>

// ---------------- Serial protocol ----------------
// PC -> Arduino : s<steering>l<left_speed>r<right_speed>\n
// Arduino -> PC : a<articulation_adc>\n
// Example      : a512\n
const unsigned int MAX_INPUT = 32;

// ---------------- Pin setting (URP / Arduino Mega) ----------------
const int STEERING_1 = 7;
const int STEERING_2 = 6;
const int FORWARD_RIGHT_1 = 3;
const int FORWARD_RIGHT_2 = 2;
const int FORWARD_LEFT_1 = 5;
const int FORWARD_LEFT_2 = 4;

// A3 is used only inside Arduino to control the towing-vehicle steering motor.
const int STEERING_POT = A3;
// A0 is the articulation joint potentiometer published to ROS 2.
const int ARTICULATION_POT = A0;

// ---------------- Steering calibration ----------------
const int STEERING_SPEED = 128;
const int STEERING_RESISTANCE_MOST_LEFT = 685;
const int STEERING_RESISTANCE_MOST_RIGHT = 545;
const int MAX_STEERING_STEP = 7;

// ---------------- State ----------------
int target_angle = 0;
int steering_resistance = 0;
int measured_steering_step = 0;
int left_speed = 0;
int right_speed = 0;

unsigned long last_command_time = 0;
const unsigned long COMMAND_INTERVAL_MS = 50;

unsigned long last_articulation_publish_time = 0;
const unsigned long ARTICULATION_INTERVAL_MS = 20;

// ---------------- Function declarations ----------------
void steerRight();
void steerLeft();
void maintainSteering();
void setLeftMotorSpeed(int speed);
void setRightMotorSpeed(int speed);
void processIncomingByte(byte in_byte);
void processData(const char *data);
void publishArticulationValue();

void setup() {
  Serial.begin(115200);

  pinMode(STEERING_POT, INPUT);
  pinMode(ARTICULATION_POT, INPUT);
  pinMode(STEERING_1, OUTPUT);
  pinMode(STEERING_2, OUTPUT);
  pinMode(FORWARD_RIGHT_1, OUTPUT);
  pinMode(FORWARD_RIGHT_2, OUTPUT);
  pinMode(FORWARD_LEFT_1, OUTPUT);
  pinMode(FORWARD_LEFT_2, OUTPUT);

  maintainSteering();
  setLeftMotorSpeed(0);
  setRightMotorSpeed(0);
}

void loop() {
  const unsigned long now = millis();

  while (Serial.available() > 0) {
    processIncomingByte((byte)Serial.read());
  }

  // Only the articulation sensor on A0 is transmitted to the PC.
  if (now - last_articulation_publish_time >= ARTICULATION_INTERVAL_MS) {
    publishArticulationValue();
    last_articulation_publish_time = now;
  }

  if (now - last_command_time >= COMMAND_INTERVAL_MS) {
    steering_resistance = analogRead(STEERING_POT);
    measured_steering_step = map(
      steering_resistance,
      STEERING_RESISTANCE_MOST_LEFT,
      STEERING_RESISTANCE_MOST_RIGHT,
      -MAX_STEERING_STEP,
      MAX_STEERING_STEP
    );
    measured_steering_step = constrain(
      measured_steering_step,
      -MAX_STEERING_STEP,
      MAX_STEERING_STEP
    );

    if (measured_steering_step == target_angle) {
      maintainSteering();
    } else if (measured_steering_step > target_angle) {
      steerLeft();
    } else {
      steerRight();
    }

    setLeftMotorSpeed(left_speed);
    setRightMotorSpeed(right_speed);
    last_command_time = now;
  }
}

void publishArticulationValue() {
  const int articulation_value = analogRead(ARTICULATION_POT);
  Serial.print('a');
  Serial.println(articulation_value);
}

void steerRight() {
  analogWrite(STEERING_1, STEERING_SPEED);
  analogWrite(STEERING_2, 0);
}

void steerLeft() {
  analogWrite(STEERING_1, 0);
  analogWrite(STEERING_2, STEERING_SPEED);
}

void maintainSteering() {
  analogWrite(STEERING_1, 0);
  analogWrite(STEERING_2, 0);
}

void setLeftMotorSpeed(int speed) {
  speed = constrain(speed, -255, 255);
  if (speed > 0) {
    analogWrite(FORWARD_LEFT_1, speed);
    analogWrite(FORWARD_LEFT_2, 0);
  } else {
    analogWrite(FORWARD_LEFT_1, 0);
    analogWrite(FORWARD_LEFT_2, -speed);
  }
}

void setRightMotorSpeed(int speed) {
  speed = constrain(speed, -255, 255);
  if (speed > 0) {
    analogWrite(FORWARD_RIGHT_1, speed);
    analogWrite(FORWARD_RIGHT_2, 0);
  } else {
    analogWrite(FORWARD_RIGHT_1, 0);
    analogWrite(FORWARD_RIGHT_2, -speed);
  }
}

void processIncomingByte(byte in_byte) {
  static char input_line[MAX_INPUT];
  static unsigned int input_pos = 0;

  if (in_byte == '\n') {
    input_line[input_pos] = '\0';
    processData(input_line);
    input_pos = 0;
  } else if (in_byte != '\r') {
    if (input_pos < MAX_INPUT - 1) {
      input_line[input_pos++] = (char)in_byte;
    } else {
      // Discard an overlong packet safely.
      input_pos = 0;
    }
  }
}

void processData(const char *data) {
  const char *s_ptr = strchr(data, 's');
  const char *l_ptr = strchr(data, 'l');
  const char *r_ptr = strchr(data, 'r');

  if (s_ptr == NULL || l_ptr == NULL || r_ptr == NULL) {
    return;
  }

  const int new_angle = atoi(s_ptr + 1);
  const int new_left_speed = atoi(l_ptr + 1);
  const int new_right_speed = atoi(r_ptr + 1);

  target_angle = constrain(new_angle, -MAX_STEERING_STEP, MAX_STEERING_STEP);
  left_speed = constrain(new_left_speed, -255, 255);
  right_speed = constrain(new_right_speed, -255, 255);
}
