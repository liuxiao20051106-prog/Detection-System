#coding:utf-8
import cv2
from ultralytics import YOLO

# 所需加载的模型目录
path = 'models/best.pt'
# 需要检测的图片地址
video_path = "TestFiles/1.mp4"

# Load the YOLOv8 model
model = YOLO(path)
cap = cv2.VideoCapture(video_path)
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        # annotated_frame = cv2.resize(annotated_frame, dsize=None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()