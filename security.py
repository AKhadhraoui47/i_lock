#! /usr/bin/python

# import the necessary packages
from picamera2 import  Picamera2
import face_recognition
import imutils
import pickle
import time
import cv2

#Add the important details for Servo Control
from gpiozero import AngularServo
servo = AngularServo(18, initial_angle=90, min_pulse_width=0.0006, max_pulse_width=0.0023)

#Initialize 'currentname' to trigger only when a new person is identified.
currentname = "Unknown"
#Determine faces from encodings.pickle file model created from train_model.py
encodingsP = "encodings.pickle"

# load the known faces and embeddings along with OpenCV's Haar
# cascade for face detection
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

# loop over frames from the video file stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to 500px (to speedup processing)
	frame = picam2.capture_array()
	#frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
	frame = imutils.resize(frame, width=500)
	# Detect the fce boxes
	boxes = face_recognition.face_locations(frame)
	# compute the facial embeddings for each face bounding box
	encodings = face_recognition.face_encodings(frame, boxes)
	names = []

	# loop over the facial embeddings
	for encoding in encodings:
		# attempt to match each face in the input image to our known
		# encodings
		matches = face_recognition.compare_faces(data["encodings"],
			encoding)
		name = "Unknown" #if face is not recognized, then print Unknown

		# check to see if we have found a match
		if True in matches:
			# find the indexes of all matched faces then initialize a
			# dictionary to count the total number of times each face
			# was matched
			matchedIdxs = [i for (i, b) in enumerate(matches) if b]
			counts = {}

			# loop over the matched indexes and maintain a count for
			# each recognized face face
			for i in matchedIdxs:
				name = data["names"][i]
				counts[name] = counts.get(name, 0) + 1

			# determine the recognized face with the largest number
			# of votes (note: in the event of an unlikely tie Python
			# will select first entry in the dictionary)
			name = max(counts, key=counts.get)

			#If someone in your dataset is identified, print their name on the screen
			if currentname != name:
				currentname = name
				print(f"[RECOGNISED] {currentname}")
				
				#Unlocks door lock
				print("[SERVO] Unlocking door")
				servo.angle = 90
				time.sleep = 2
				#servo.angle = 0
		else:
			if currentname != "Unknown":
				currentname = "Unknown"
				print("[WARNING] Unknown detected")
				servo.angle = 0
				print("[SERVO] Door locked") 
		# update the list of names
		names.append(name)

	# loop over the recognized faces
	for ((top, right, bottom, left), name) in zip(boxes, names):
		# draw the predicted face name on the image - color is in BGR
		cv2.rectangle(frame, (left, top), (right, bottom),
			(0, 255, 225), 2)
		y = top - 15 if top - 15 > 15 else top + 15
		cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
			.8, (0, 255, 255), 2)

	# display the image to our screen
	cv2.imshow("Facial Recognition is Running", frame)
	key = cv2.waitKey(1) & 0xFF

	# quit when 'q' key is pressed
	if key == 27:
		break

	frame_count += 1

# stop the timer and display FPS information
elapsed_time = time.time() - start_time
fps = frame_count / elapsed_time
print("[INFO] elasped time: {:.2f}".format(elapsed_time))
print("[INFO] approx. FPS: {:.2f}".format(fps))

# do a bit of cleanup
cv2.destroyAllWindows()
picam2.stop()
