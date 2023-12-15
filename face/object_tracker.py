# -*- coding: UTF-8 -*-
# 调用所需库
from objecttracker.centroidtracker import CentroidTracker
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

# 设置参数
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
ap.add_argument("-v", "--video", type=str, help="path to optional input video file")
args = vars(ap.parse_args())

# 初始化CentroidTracker的对象，初始化图像高度与宽度
ct = CentroidTracker()
(H, W) = (None, None)

# 加载caffe模型
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# 如果没有提供视频文件的路径，则从摄像头中抓取图像
if not args.get("video", False):
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(1.0)

# 如果提供了视频文件路径，则从文件中抓取图像
else:
    vs = cv2.VideoCapture(args["video"])

# 启动帧率估计
fps = FPS().start()

# 循环
while True:
    # 抓取下一帧，并根据视频流的来源进行调整
    frame = vs.read()
    #frame = frame[1] if args.get("video", False) else frame
    if args.get("video", False):
        frame = frame[1]
    else:
        frame = frame
        print("no 'video' ")

    # 检查视频流是否结束
    if frame is None:
        break

    # 缩放图像帧并保持长宽比
    frame = imutils.resize(frame, width=400)

    # 如果H和W为空，则赋值
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # 将图像帧传递至DNN
    # detections是DNN的识别输出结果
    # 初始化rects，rects中存储的应当是有效识别
    blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    rects = []

    # 遍历输出结果detections
    for i in range(0, detections.shape[2]):
        # 过滤掉置信度不够的识别结果
        if detections[0, 0, i, 2] > args["confidence"]:
            # 计算外接矩形的坐标，并加入rects
            box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            rects.append(box.astype("int"))

            # 绘制外接矩形
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # 调用CentroidTracker的函数实现形心跟踪算法
    objects = ct.update(rects)

    # 遍历追踪列表的目标
    for (objectID, centroid) in objects.items():
        # 在形心附近绘制编号
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    # 显示图像
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # 按下q可以退出循环
    if key == ord("q"):
        break

    # 更新计数器
    fps.update()

# 停止计数器，并显示帧数
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# 如果我们使用的是摄像头，则释放调用
if not args.get("video", False):
    vs.stop()

# 如果使用的是视频文件，则释放调用
else:
    vs.release()

# 关闭所有窗口
cv2.destroyAllWindows()