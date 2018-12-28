import RPI.GPIO as GPIO
import sys
import time

pin_camera_1 = 17
pin_camera_2 = 18
pin_camera_3 = 27
pin_camera_4 = 22


class camera():
	def __init__(self):
		self.pin = pin_camera_1
		GPIO.setmode(GPIO.BCM)
		GPIO.setwarnings(False)
		GPIO.setup(self.pin,GPIO.OUT)
		GPIO.output(self.pin,GPIO.LOW)

	def vibrate(self):
		GPIO.output(self.pin, GPIO.HIGH)
		print('vibrating')

	def off(self):
		GPIO.output(self.pin, GPIO.LOW)
		print("Stop")

# specific channel
def blink_every_two_secs():
	GPIO.setmode(GPIO.BOARD)
	GPIO.setup(12, GPIO.OUT)

	p = GPIO.PWM(12, 0.5)
	p.start(1)
	input('Press return to stop:')   # use raw_input for Python 2
	p.stop()
	GPIO.cleanup()


def dim_led():
	p = GPIO.PWM(12, 50)  # channel=12 frequency=50Hz
	p.start(0)
	try:
		while 1:
			for dc in range(0, 101, 5):
				p.ChangeDutyCycle(dc)
				time.sleep(0.1)
			for dc in range(100, -1, -5):
				p.ChangeDutyCycle(dc)
				time.sleep(0.1)
	except KeyboardInterrupt:
		pass
	p.stop()
	GPIO.cleanup()