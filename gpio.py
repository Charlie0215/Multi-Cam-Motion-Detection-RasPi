import RPI.GPIO as GPIO
import sys
import time

pin_camera_1 = 17
pin_camera_2 = 18
pin_camera_3 = 27
pin_camera_4 = 22


class camera_1():
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


class camera_2():
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


class camera_3():
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


class camera_4():
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

