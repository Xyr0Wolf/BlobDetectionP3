from gpiozero import LED, DistanceSensor, AngularServo, Servo
from time import sleep
from math import sin
#sensor = DistanceSensor(17, 27)
servoInner = AngularServo(12, min_angle=-90, max_angle=90)
servoOuter = Servo(13)

# turnTheeseOn = (LED(26), LED(16))
# for turnOn in turnTheeseOn:
#     turnOn.on()
#     sleep(1)
#     print(turnOn)

servoInner.angle = 0
servoOuter.value = -0.1

ree = 0.1
while True:
    ree += 0.01
    servoInner.angle = sin(ree)*45
    sleep(0.01)


sleep(1)

# while True:
#     print('Distance thingy: ', sensor.distance, 'm')
#     sleep(1)