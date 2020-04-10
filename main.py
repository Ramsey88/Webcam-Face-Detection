import cv2
import Transformation
from facedetector import FaceDetector

video = cv2.VideoCapture(0)
p = FaceDetector(faceCascadePath="haarcascade_frontalface_default.xml")
while True:
 (grabbed, frame) = video.read()
 print(grabbed,frame)
 if not grabbed:
     break
 frame = Transformation.resize(frame, width=300)
 gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 faces = p.detect(gray,1.3)
 for (x, y, w, h) in faces:
     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
 cv2.imshow("camera", frame)

 k = cv2.waitKey(30) & 0xff
 if k == 27:
    break
video.release()
cv2.destroyAllWindows()