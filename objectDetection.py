import cv2
import numpy as np
from ultralytics import YOLO
import torch

#The following line Brings in the Video in Question
cap = cv2.VideoCapture(0)

#yolo8m.pt is just a model in the YOLO Database, its a medium strength model
model =  YOLO("yolov8m.pt")

#The While loop is used to keep playing the video frame by frame
while True:
    ret, frame = cap.read()
    #The if Loop is used to stop the window incase there are no more frames left
    if not ret:
        break

    results = model(frame)
    result = results[0]
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
    classes = np.array(result.boxes.cls.cpu(), dtype="int")
    for cls, bbox in zip(classes, bboxes):
        (x, y, x2, y2) = bbox

        cv2.rectangle(frame, (x,y), (x2,y2), (0,0,225), 2)
        cv2.putText(frame, str(cls), (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,225), 2)

    #The Following Line is used to open a window showing the frame of the video,
    #but the while true loop makes it so that it keep showing the frames till the end
    cv2.imshow('cam', frame)
    #Unless the waitkey command is used, the window opens and closes abruptly
    key=cv2.waitKey(2)
    if key ==27:
        break
cap.release()
cv2.destroyAllWindows()

