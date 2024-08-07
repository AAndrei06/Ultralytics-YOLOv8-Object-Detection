from ultralytics import YOLO
import cv2
import math
import cvzone

cap = cv2.VideoCapture('video.mp4')

model = YOLO('best.pt')


while True:
	isTrue, img = cap.read()
	
	results = model.track(img, conf=0.3, iou=0.5)
	
	for r in results:
		print(r)
		for box in r.boxes:
			conf = int(math.ceil(box.conf[0]*100))
			x1,y1,x2,y2 = box.xyxy[0]
			x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
			cvzone.cornerRect(img,(x1,y1,x2-x1,y2-y1),l = 12,rt = 1,colorR = (255,0,0))
			cvzone.putTextRect(img,f'Cat {conf}%',(max(0,x1+7),max(20,y1-8)),scale = 1,thickness = 2,offset = 7)
		
	cv2.imshow('Image',img)
	
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break
	

cap.release()
cv2.destroyAllWindows()
