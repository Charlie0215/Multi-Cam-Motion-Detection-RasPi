import socket, time
import cv2
import numpy as np
import Image, StringIO

cap = cv2.VideoCapture(0)
ret = cap.set(CV_CAP_PROP_FRAME_WIDTH, 320) 
ret = cap.set(CV_CAP_PROP_FRAME_HEIGHT, 240)

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind(("0.0.0.0", 9996))
sock.listen(2)

dst, dst_addr = sock.accept()

print("Destination connected by", dst_addr)

while True:

	flag, frame = cap.read()
	if flag == True:
		#image = cv2.fromarray(frame)
		pi = Image.fromstring("RGB", frame.shape, img.tostring())
		buf = StringIO.StringIO()
		pi.save(buf, format="JPEG")
		jepg = buf.getvalue()
		buf.close()
		transfer = jpeg.replace("\n", "\\-n")

		try:
			dst.sendall(transfer + "\n")
			time.sleep(0.04)
		except Exception as ex:
			dst, dst_addr = sock.accept()
			print "Destination Connected Again By", dst_addr
		except KeyboardInterupt:
			print("Interrupted")
			break
sock.close()
dst.close()

