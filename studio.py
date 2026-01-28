import cv2
from picamera2 import Picamera2
import os

if len(sys.argv) > 1:
    name = sys.argv[1]
else:
    name = input("Enter the person's name: ")

os.makedirs(f"dataset/{name}", exist_ok=True)

picam2 = Picamera2()

config = picam2.create_preview_configuration(main={"size": (512, 304), "format": "RGB888"}) 
picam2.configure(config)

picam2.start()
    
img_counter = 0

while True:
    image = picam2.capture_array()
    cv2.imshow("Press Space to take a photo", image )
    k = cv2.waitKey(1)
       
    if k%256 == 32:
        img_name = "dataset/{}/image_{}.jpg".format(name,img_counter)
        cv2.imwrite(img_name, image)
        print("{} written!".format(img_name))
        img_counter += 1
    elif k%256 == 27:
        print("Escape hit, closing...")
        break

picam2.stop()
cv2.destroyAllWindows()
