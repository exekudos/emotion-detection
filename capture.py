import cv2
import numpy as np
import model

def detectFaces(image):
	images_of_faces=[]
	gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	faces=face_cascade.detectMultiScale(gray, 1.3, 5)
	font=cv2.FONT_HERSHEY_SIMPLEX
	for (x,y,w,h) in faces:
		cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
		roi=gray[y:y+h, x:x+w]
		roi=cv2.resize(roi, (48,48))
		roi=roi.flatten()
		img=np.array([roi], dtype=np.float32)
		emotion=model.predict(img)
		#images_of_faces.append(roi)
		cv2.putText(image,emotion[0],(x+w,y),font,1,(0,255,0),2)
	#images_of_faces=np.array(images_of_faces)
	#return images_of_faces

cv2.namedWindow('Preview')
cap=cv2.VideoCapture(0)
while True:
	cv2.waitKey(200)
	ret,frame=cap.read()
	detectFaces(frame)
	cv2.imshow('image',frame)
	'''img=images_of_faces
	img=np.asarray(img, dtype=np.float32)
	img.flatten()
	img=np.array([img], dtype=np.float32)
	emotion=model.predict(img)
	print(emotion)'''
	if cv2.waitKey(1) & 0xFF==ord('q'):
		break
	#break