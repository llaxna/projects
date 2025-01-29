import RPi.GPIO as GPIO
from time import sleep

ledpin = 18

GPIO.setmode(GPIO.BCM)
GPIO.setup(ledpin, GPIO.OUT)

GPIO.setwarnings(False)

pwm = GPIO.PWM(ledpin, 1000)

pwm.start(0) 
while True:
    for duty in range(0, 101):
        print(duty)
        pwm.ChangeDutyCycle(duty)
        sleep(0.05)
    sleep(0.5)
    for duty in range(100, -1, -1):
        print(duty)
        pwm.ChangeDutyCycle(duty)
        sleep(0.05)
    pwm.ChangeDutyCycle(0)
    sleep(0.5)