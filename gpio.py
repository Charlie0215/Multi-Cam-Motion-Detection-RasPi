import RPi.GPIO as GPIO
import sys
import time
import threading

pin_for_camera_1 = 17
pin_for_camera_2 = 18
#pin_camera_3 = 27
#pin_camera_4 = 22

lock = threading.Lock()

class  vibrate_thread(threading.Thread):  
    def __init__(self, threadID, name, counter, pin):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter
        if pin == 1:
            self.pin = pin_for_camera_1
        else:
            self.pin = pin_for_camera_2
    def run(self):
        global n, lock
        lock.acquire()
        print("Starting " + self.name)
        vibrate(self.pin)
        print("Exiting " + self.name)
        exit(self.pin)
        lock.release()
def vibrate(pin):
	GPIO.setmode(GPIO.BCM)
	GPIO.setwarnings(False)
	GPIO.setup(pin,GPIO.OUT)
	GPIO.output(pin, GPIO.HIGH)
	time.sleep(0.25)
	#GPIO.output(pin, GPIO.LOW)
def exit(pin):
	GPIO.output(pin, GPIO.LOW)
	print("Stop")

'''
class camera(threading.Thread):
	def __init__(self, pin):
		threading.Thread.__init__(self)
		if pin == 1:
			self.pin = pin_for_camera_1
		else:
			self.pin = pin_for_camera_2
		GPIO.setmode(GPIO.BCM)
		GPIO.setwarnings(False)
		GPIO.setup(self.pin,GPIO.OUT)
		GPIO.output(self.pin,GPIO.LOW)
		#self.threads = []

	def vibrate(self):
		GPIO.output(self.pin, GPIO.HIGH)
		time.sleep(0.6)
		GPIO.output(self.pin, GPIO.LOW)
		#time.sleep(0.1)
		#GPIO.cleanup()

	def off(self):
		GPIO.output(self.pin, GPIO.LOW)
		#GPIO.cleanup()
		print("Stop")
	
	def thread(self):
		threads = []
		t = threading.Thread(target=self.vibrate)
		threads.append(t)
	
'''
'''
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
'''
	
