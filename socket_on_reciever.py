import cv2
import socket, time
import Image, StringIO
import numpy as np

HOST, PORT = "10.0.1.13", 9996
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((HOST, PORT)) 

f = sock.makefile()

cv2.NamedWindow("camera_server")

while True:
	msg = f.readline()
	if not msg:
		break
	jpeg = msg.replace("\\-n", "\n")
	buf = StringIO.StringIO(jpeg[0:-1])
	buf.seek(0)
	pi = Image.open(buf)
	img = cv2.CreateImageHeader((320, 240), cv.IPL_DEPTH_8U, 3)
	cv2.SetData(img, pi.tostring())
	buf.close()
	frame_cvmat=cv2.GetMat(img)
	frame=np.asarray(frame_cvmat)
	cv2.imshow('frame',frame)
	if cv2.waitKey(1) && 0xFF == ord('q'):
		break

sock.close()
cv2.DistoryAllWindows()