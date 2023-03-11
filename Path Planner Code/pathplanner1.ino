#include <WiFi.h>

//CONNECTION SETUP 
const char* ssid = "WIFI_NAME"; //enter wifi name
const char* password =  "psw"; //enter password

const uint16_t port = 1234; //CHANGE IF CONNECTION DOESNT WORK
const char * host = "192.168.1.83";//change as per ip address

String inp="";

//MOTOR STUFF
#define MOTRF 27 //right forward
#define MOTRB 26 //right back
#define MOTLF 16 //left forward
#define MOTLB 17 //left back
const int enA=14;
const int enB=25;

//PLEASE CHANGE THESE AS PER EXPERIMENTAL DATA
#define pixel 15 //PIXEL SIDE LENGTH change this as required in CENTIMETERS
#define RPM 200 //RPM OF MOTOR change this as required
#define Radius 5 //RADIUS OF WHEEL change this as required in CENTIMETERS
float time_pixel=0; // time taken for motor to make vehicle cross 1 pixel
float time_turn=2000; // MILISECOND time taken for spot turn of vehicle by 90 deg CHANGE AS REQUIRED





void setup()
{

 //CONNECTION STUFF
  Serial.begin(115200);
 
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.println("...");
  }
 
  Serial.print("WiFi connected with IP: ");
  Serial.println(WiFi.localIP());

//MOTOR STUFF
Serial.begin(115200);
pinMode(MOTRF, OUTPUT);
pinMode(MOTLB, OUTPUT);
pinMode(MOTRB, OUTPUT);
pinMode(MOTLF, OUTPUT);

//SPECIFICS CALCULATION
float dist_rev=3.14*2*Radius; //basically circumference
float time_rev=RPM/60;
float rev_pixel= pixel/dist_rev;
time_pixel= time_rev*rev_pixel*1000; //x1000 because miliseconds
Serial.println(time_pixel);

int len=inp.length();

}


void receive()
{
    WiFiClient client;
 
    while(client.connect(host, port)){
 
    Serial.println("Connected to server successful!");
 
     inp = client.read(); //read the input string from pc

    Serial.print(inp);

    }
    Serial.println("Disconnecting...");
    client.stop();
 
    delay(10000);
}


void FORWARD()
{
  digitalWrite(MOTRF, HIGH);
  digitalWrite(MOTLF, HIGH);
  delay(time_pixel);
  digitalWrite(MOTRF, LOW);
  digitalWrite(MOTLF, LOW);

}

void BACKWARD()
{
  digitalWrite(MOTRB, HIGH);
  digitalWrite(MOTLB, HIGH);
  delay(time_pixel);
  digitalWrite(MOTRB, LOW);
  digitalWrite(MOTLB, LOW);
}

void LEFT()
{
  digitalWrite(MOTRF, HIGH);
  digitalWrite(MOTLB, HIGH);
  delay(time_turn);
  digitalWrite(MOTRF, LOW);
  digitalWrite(MOTLB, LOW);
}

void RIGHT()
{
  digitalWrite(MOTRB, HIGH);
  digitalWrite(MOTLF, HIGH);
  delay(time_turn);
  digitalWrite(MOTRB, LOW);
  digitalWrite(MOTLF, LOW);
}



int i=0;

void loop() {

  digitalWrite(enA, HIGH);
  digitalWrite(enB, HIGH);
while(i<inp.length())
{
if(i>0)
{
  switch(inp.charAt(i)){

    
    case 'N': if(inp.charAt(i-1)=='E')
                {
                  LEFT();
                  FORWARD();
                }
                else if(inp.charAt(i-1)=='W')
                {
                  RIGHT();
                  FORWARD();
                }
                else
                {
                  FORWARD();
                }
                break;
                    
    case 'E': if(inp.charAt(i-1)=='S')
                {
                  LEFT();
                  FORWARD();
                }
                else if(inp.charAt(i-1)=='N')
                {
                  RIGHT();
                  FORWARD();
                }
                else
                {
                  FORWARD();
                }
                break;

          
    case 'W': if(inp.charAt(i-1)=='N')
                {
                  LEFT();
                  FORWARD();
                }
                else if(inp.charAt(i-1)=='S')
                {
                  RIGHT();
                  FORWARD();
                }
                else
                {
                  FORWARD();
                }
                break;
       
    case 'S': if(inp.charAt(i-1)=='W')
                {
                  LEFT();
                  FORWARD();
                }
                else if(inp.charAt(i-1)=='E')
                {
                  RIGHT();
                  FORWARD();
                }
                else
                {
                  FORWARD();
                }
                break;

                
                }
    
  }

 else{
  if(inp.charAt(0)=='N'){
    FORWARD();
  }
  else if(inp.charAt(0)=='E'){
    RIGHT();
    FORWARD();
    
  }
    else if(inp.charAt(0)=='W'){
    LEFT();
    FORWARD();
    
  }
    else if(inp.charAt(0)=='S'){
   
    BACKWARD();
    
  }
 }
 
  i++;
}
}
