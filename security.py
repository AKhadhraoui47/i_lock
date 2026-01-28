#! /usr/bin/python
 
from picamera2 import  Picamera2
import face_recognition
import imutils
import pickle
import time
import cv2

from gpiozero import AngularServo
servo = AngularServo(18, initial_angle=90, min_pulse_width=0.0006, max_pulse_width=0.0023)

currentname = "Unknown"
encodingsP = "encodings.pickle"

print("[INFO] loading encodings + face detector...")
data = pickle.loads(open(encodingsP, "rb").read())

print("[INFO] starting camera...")
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(config)
picam2.start()
time.sleep(2.0)

frame_count = 0
start_time = time.time()

while True:
	 
	frame = picam2.capture_array()
	frame = imutils.resize(frame, width=500)
	boxes = face_recognition.face_locations(frame)
	encodings = face_recognition.face_encodings(frame, boxes)
	names = []
	for encoding in encodings:
		matches = face_recognition.compare_faces(data["encodings"],
			encoding)
		name = "Unknown"  
		if True in matches:
			matchedIdxs = [i for (i, b) in enumerate(matches) if b]
			counts = {}
			for i in matchedIdxs:
				name = data["names"][i]
				counts[name] = counts.get(name, 0) + 1
			name = max(counts, key=counts.get)
			if currentname != name:
				currentname = name
				print(f"[RECOGNISED] {currentname}")
				
				print("[SERVO] Unlocking door")
				servo.angle = 90
				time.sleep = 2 
		else:
			if currentname != "Unknown":
				currentname = "Unknown"
				print("[WARNING] Unknown detected")
				servo.angle = 0
				print("[SERVO] Door locked") 
		names.append(name)

	for ((top, right, bottom, left), name) in zip(boxes, names):
		 
		cv2.rectangle(frame, (left, top), (right, bottom),
			(0, 255, 225), 2)
		y = top - 15 if top - 15 > 15 else top + 15
		cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
			.8, (0, 255, 255), 2)
	 
	cv2.imshow("Facial Recognition is Running", frame)
	key = cv2.waitKey(1) & 0xFF
	 
	if key == 27:
		break
	frame_count += 1

elapsed_time = time.time() - start_time
fps = frame_count / elapsed_time
print("[INFO] elasped time: {:.2f}".format(elapsed_time))
print("[INFO] approx. FPS: {:.2f}".format(fps))

cv2.destroyAllWindows()
picam2.stop()
