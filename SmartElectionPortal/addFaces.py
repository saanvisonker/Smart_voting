import cv2
import pickle
# covert face data into text data
import numpy as np 
import os

# if data file does not exist then create new
if not os.path.exists('data/'):
    os.makedirs('data/')
# used to select primary web cam
video = cv2.VideoCapture(0)
# used for face detection
facedetect= cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_data = []

i =0
name=input("Enter your aadhar number:")
framesTotal =51
captureAfterFrame=2
 
#  from video take frames and when frame value== 51 then stop
while True:
    ret, frame =video.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces =facedetect.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        crop_img = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img,(50,50))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
        if len(face_data)<=framesTotal and i%captureAfterFrame==0:
            face_data.append(resized_img)
            cv2.putText(frame,str(len(face_data)),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(50,50,255),1)
        i=i+1
             
           

    cv2.imshow('frame',frame)
    k=cv2.waitKey(1)
    if k ==ord('q') or len(face_data)>=framesTotal:
        break

video.release()
cv2.destroyAllWindows()

face_data= np.asarray(face_data)
face_data=face_data.reshape(framesTotal,-1) 
print(len(face_data))
print(face_data)

if'name.pk1' not in os.listdir('data/'):
    name=[name]*framesTotal
    with open('data/names.pk1','wb') as f:
        pickle.dump(name,f)
else:
    with open('data/names.pk1','rb') as f:
        name =pickle.load(f)
        # append data
        name=name+[name]*framesTotal
        with open('data/names.pk1','wb') as f:
            pickle.dump(name,f)

if 'face_data.pk1' not in os.listdir('data/'):
    with open('data/face_data.pk1','wb') as f:
            pickle.dump(face_data,f)
else: 
    with open('data/face_data.pk1','rb') as f:
        faces =pickle.load(f)
        faces=np.append(faces,face_data,axis=0)
        with open('data/names.pk1','wb') as f:
            pickle.dump(name,f)
