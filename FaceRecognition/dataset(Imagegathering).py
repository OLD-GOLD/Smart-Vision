import cv2

path = 'Dataset(images)'
# Initialize the camera usage
capture = cv2.VideoCapture(0)

# Load the classifier(model) that is pre-programmed to identify faces
# In line below, you need to specify the location of the file
face_detector = cv2.CascadeClassifier('D:\Software app development\Python projects\projectAssets\haarcascade_frontalface_default.xml')

face_id = input("Enter an id for a user \n")

# Initialize the count variable so that it keep records of how many images taken
count = 0

while(True):
    ret,frame = capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        # Save the captured image into the datasets folder
        if ret:
            # if video is rolling take images and save them
            # In line below, you need to specify the location of the file
            name = "Dataset(images)\img." + str(count)+'.'+str(face_id)+".jpg"
            print(name+' created successfully')
            # writing the extracted images
            cv2.imwrite(name, frame)
            count +=1
    end = cv2.waitKey(1) & 0xff  # press Esc to break the loop
    if end == 27:
        break
    elif count >= 50:# Take 50 face sample and stop video
        break


capture.release()
cv2.destroyAllWindows()