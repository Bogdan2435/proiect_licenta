#define enA 9
#define in1 6
#define in2 7

#define enB 10
#define in3 4
#define in4 5

int rotDirection = 0;
int pressed = false;

void setup() {
  pinMode(enA, OUTPUT);
  pinMode(in1, OUTPUT);
  pinMode(in2, OUTPUT);

  pinMode(enB, OUTPUT);
  pinMode(in3, OUTPUT);
  pinMode(in4, OUTPUT);
 
  // Directia de rotire
  digitalWrite(in1, LOW);
  digitalWrite(in2, HIGH);

  digitalWrite(in3, LOW);
  digitalWrite(in4, HIGH);
}

void loop() {
  
  // valori de la 0 la 100 pentru putere
  analogWrite(enA, 100); 
  analogWrite(enB, 100);
}