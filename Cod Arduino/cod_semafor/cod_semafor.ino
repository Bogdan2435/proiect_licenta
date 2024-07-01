int red = 9;
int green = 7;

void setup(){
  
  pinMode(red, OUTPUT);
  pinMode(green,  OUTPUT);
  
}
void loop(){
digitalWrite(red, HIGH);
delay(15000);
digitalWrite(red,  LOW);
  
digitalWrite(green, HIGH);
delay(15000);
digitalWrite(green,  LOW);

}