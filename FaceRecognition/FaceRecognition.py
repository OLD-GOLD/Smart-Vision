import cv2
import os
import math
import time
import pyttsx3
start_time = time.time()

# os.system('python FaceTrainer.py')

recognizer = cv2.face.LBPHFaceRecognizer_create()
# In line below, you need to specify the location of the files
recognizer.read('D:\Software app development\Python projects\FaceRecognition\ trainer.yml')
cascadePath = "D:\Software app development\Python projects\projectAssets\haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX
# indicate id counter
id = 0
# names that are related to the id such as Naji =0
names = ["Naji","Amer","another dude i don't know "]
# Actual width of face (Assumption)
actual_width = 19.5
# Initialize and start realtime video capture
frame = cv2.VideoCapture(0)
# Define min window size to be recognized as a face
minW = 0.1 * frame.get(3)
minH = 0.1 * frame.get(4)

voice = pyttsx3.init()


print("--- %s seconds ---" % (time.time() - start_time))


while True:
    ret, img = frame.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )
    for (x, y, width, height) in faces:
        cv2.rectangle(img, (x, y), (x + width, y + height), (0, 0, 0), 2)
        id, confidence = recognizer.predict(gray[y:y + height, x:x + width])
        pixel_distance = math.sqrt(((x + width)) ** 2 + ((y + height)) ** 2)
        actual_distance = (pixel_distance / width) * actual_width
        distance = '%.2f' % actual_distance
        # if confidence is 100 then its a perfect match (Hard to reach)
        if (confidence <= 60):
            id = names[id]
            # voice.say(id)
            # voice.runAndWait()
            confidence = "{0}%".format(round(100 - confidence))
        else: # if confidence is 0 then classify as unknown
            id = "unknown"
            confidence = "{0}%".format(round(100 - confidence))
            


        cv2.putText(img, str("Name: "+id), (x, y-60), font, 1, (0, 0, 0), 2)
        cv2.putText(img, str("Confidence: "+confidence), (x, y-35), font, 1, (0, 0, 0), 2)
        cv2.putText(img, str("Distance: " + distance), (x, y-10), font, 1, (0, 0, 0), 2)
        cv2.imshow('camera', img)

    end = cv2.waitKey(1) & 0xff  # press Esc to break the loop
    if end == 27:
        break

frame.release()
cv2.destroyAllWindows()