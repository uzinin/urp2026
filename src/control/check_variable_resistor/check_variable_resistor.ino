//int sensorPin = A3;
//
//void setup() {
//
//  Serial.begin(9600);
//
//  pinMode(sensorPin, INPUT);
//
//}
//
// 
//
//void loop() {
//
//  int value = analogRead(sensorPin);
//
//  Serial.println(value);
//
//  delay(100);
//
//}




int sensorPin1 = A3;
int sensorPin2 = A0; // A0 핀 추가

void setup() {
  Serial.begin(9600);
  
  pinMode(sensorPin1, INPUT);
  pinMode(sensorPin2, INPUT); // A0 핀 입력 설정 추가
}

void loop() {
  // 두 핀의 아날로그 값을 각각 읽어옵니다.
  int value1 = analogRead(sensorPin1);
  int value2 = analogRead(sensorPin2);

  // 시리얼 모니터에서 확인하기 쉽게 출력합니다.
  Serial.print("A3: ");
  Serial.print(value1);
  Serial.print("\t A0: "); // \t는 탭(일정 간격 띄어쓰기)을 의미합니다.
  Serial.println(value2);

  delay(100);
}
