// E-BTS two-phase driver (Arduino / Teensy).
//
// Drives two output pins as a complementary two-phase square wave:
//   D2 (pinA) and D4 (pinB) each toggle every 20 ms, so each is HIGH for
//   20 ms out of every 40 ms -> 25 Hz, 50% duty, 180 degrees out of phase.
// Serial is opened at 115200 baud (no data is sent). See the physical-setup
// README for what D2/D4 drive in the rig.

const int pinA = 2;
const int pinB = 4;

elapsedMillis timer;
int state = 0;

void setup() {
  pinMode(pinA, OUTPUT);
  pinMode(pinB, OUTPUT);
  Serial.begin(115200);

  digitalWrite(pinA, HIGH);
  digitalWrite(pinB, LOW);
  timer = 0;
  state = 1;
}

void loop() {
  if (state == 1 && timer >= 20) {
    digitalWrite(pinA, LOW);
    digitalWrite(pinB, HIGH);
    timer = 0;
    state = 2;
  }

  if (state == 2 && timer >= 20) {
    digitalWrite(pinB, LOW);
    digitalWrite(pinA, HIGH);
    timer = 0;
    state = 1;
  }
}
