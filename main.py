'''
The Simple python program the uses OpenCV to detect no of faces and eyes in the video
@author: Mohit Singh
@github: https://github.com/devmohit-live
@LinkedIN : https://www.linkedin.com/in/devmohitsingh/

'''



import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0) # or video_feed_of_ip_of_your_camera
eye_cascade=cv.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
detector=cv.CascadeClassifier("haarcascade_frontalface_default.xml")


while True: 
    ret, frame = cap.read()
    faces=detector.detectMultiScale(frame,1.3,5)
    
    if(len(faces) > 0):
        
        for face in faces:
    
            x,y,w,h=face

            cv.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),5)
            eyes = eye_cascade.detectMultiScale(frame)

            if len(eyes) > 0:
                for (x_eye,y_eye,w_eye,h_eye) in eyes: 
                    center = (int(x_eye + 0.5*w_eye), int(y_eye + 0.5*h_eye)) 
                    radius = int(0.3 * (w_eye + h_eye)) 
                    color = (0, 255, 0) 
                    thickness = 3 
                    cv.circle(frame, center, radius, color, thickness)

                    
    frame=cv.putText(frame, str(len(faces))+' Faces '+str(len(eyes))+' eye dtected',(50,50),cv.FONT_HERSHEY_SIMPLEX,0.75,[0,255,0],3)
    cv.imshow('Input', frame) 
    
    if cv.waitKey(1) == 13:  # press enter to exit
        break 
        
cv.destroyAllWindows()
cap.release() 
