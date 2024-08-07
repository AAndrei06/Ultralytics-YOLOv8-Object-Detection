from ultralytics import YOLO
import numpy as np
import cv2
import cvzone
from sort import *
import math

model = YOLO('yolov8n.pt')
capture = cv2.VideoCapture('video.mp4')

tracker = Sort(max_age = 20,min_hits = 3,iou_threshold = 0.3)
going = []
coming = []

while True:
	
	isTrue, img = capture.read() 
	results = model(img, stream = True)
	detections = np.empty((0,5))
	
	for r in results:
		for box in r.boxes:
			conf = int(math.ceil(box.conf[0]*100))
			cls = model.names[int(box.cls[0])]
			if cls == 'car' or cls == 'truck' or cls == 'bus' or cls == 'motorbike' and conf > 30:
				x1,y1,x2,y2 = box.xyxy[0]
				x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
				cvzone.cornerRect(img,(x1,y1,x2-x1,y2-y1),l = 12,rt = 1,colorR = (255,0,0))
				cvzone.putTextRect(img,f'{cls} {conf}%',(max(0,x1+7),max(20,y1-8)),scale = 1,thickness = 2,offset = 7)
				detections = np.vstack((detections,np.array([x1,y1,x2,y2,conf])))
				
	cv2.line(img,(0,500),(1920,500),(0,0,255),thickness = 7)
	resultsTracker = tracker.update(detections)
	
	for result in resultsTracker:
		x1,y1,x2,y2,id = result
		x1,y1,x2,y2,id = int(x1),int(y1),int(x2),int(y2),int(id)
		cx,cy = (x1+x2)//2, (y1+y2)//2
		
		if 480 < cy < 520:
			if cx < 920 and coming.count(id) == 0:	
				coming.append(id)
				cv2.line(img,(0,500),(1920,500),(0,255,0),thickness = 7)

			elif cx > 920 and  going.count(id) == 0:
				going.append(id)
				cv2.line(img,(0,500),(1920,500),(0,255,9),thickness = 7)

				
	
	img = cv2.putText(img, f'Coming: {len(coming)}', (60,60), cv2.FONT_HERSHEY_SIMPLEX , 2, (255,0,0), 5, cv2.LINE_AA)
	img = cv2.putText(img, f'Going: {len(going)}', (1600,60), cv2.FONT_HERSHEY_SIMPLEX , 2, (255,0,0), 5, cv2.LINE_AA)
	
	cv2.imshow('Highway',img)
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break
	

capture.release()
cv2.destroyAllWindows()
