#coding:utf-8
from ultralytics import YOLO
import cv2

# 所需加载的模型目录
path = 'models/best.pt'
# 需要检测的图片地址
img_path = "UIProgram/ui_imgs/微信截图_20250222235847.png"

# 加载预训练模型
# conf	0.25	object confidence threshold for detection
# iou	0.7	intersection over union (IoU) threshold for NMS
model = YOLO(path, task='detect')
# model = YOLO(path, task='detect',conf=0.5)


# 检测图片
results = model(img_path)
res = results[0].plot()
res = cv2.resize(res,dsize=None,fx=2,fy=2,interpolation=cv2.INTER_LINEAR)
cv2.imshow("YOLOv8 Detection", res)
cv2.waitKey(0)