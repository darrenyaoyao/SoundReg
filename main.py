#!/usr/bin/env python3
import signal
import sys
import threading
import RPi.GPIO as GPIO

BUTTON_GPIO = 23 
RECORD_LED_GPIO = 17
record_led_high_sec = [0]

def signal_handler(sig, frame):
    GPIO.cleanup()
    sys.exit(0)

def button_pressed_callback(channel):
    print("Button pressed!")
    GPIO.output(RECORD_LED_GPIO, True)
    t = threading.Timer(3.0, check_record_led)
    t.start()

def check_record_led():
    GPIO.output(RECORD_LED_GPIO, False)

if __name__ == '__main__':
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(BUTTON_GPIO, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    GPIO.setup(RECORD_LED_GPIO, GPIO.OUT)
    GPIO.add_event_detect(BUTTON_GPIO, GPIO.FALLING, 
            callback=button_pressed_callback, bouncetime=100)

    
    signal.signal(signal.SIGINT, signal_handler)
    
    while True:
        a = 0
