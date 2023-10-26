# Real-time Object Detection using YOLOv8 and OpenCV

This code performs real-time object detection on a live video stream from the camera using the YOLOv8 object detection model. 

## Imports
The code first imports the necessary libraries:

- `cv2` - OpenCV library for computer vision and video processing
- `numpy` - For numerical processing on arrays
- `ultralytics` - Python library containing the YOLOv8 implementation
- `torch` - PyTorch for running deep learning models

## Video Capture
It then initializes video capture from the default camera device 0:

```python
cap = cv2.VideoCapture(0) 
```

This opens the video stream and allows reading frames.

## Load YOLOv8 Model
The pretrained YOLOv8 medium model is loaded which can detect a wide variety of objects:

```python
model = YOLO("yolov8m.pt")
``` 

This model runs on a GPU by default for efficient inference.

## Prediction Loop
A `while True` loop is used to read each frame and run object detection on it:

- `ret, frame = cap.read()` reads the next video frame.
- `results = model(frame)` runs object detection on the frame using YOLOv8.
- The detections are extracted and converted to bounding boxes and class IDs.
- A loop draws the bounding boxes and class name for each detected object.
- `cv2.imshow` displays the annotated frame.

This loop runs indefinitely performing real-time detection until aborted.

## Cleanup
Once finished, the video capture is released and windows closed cleanly.

This simple pipeline allows harnessing the power of YOLOv8 to detect objects in real-time on a video stream using just a few lines of code in OpenCV! The modular design makes it easy to integrate into more complex applications.
