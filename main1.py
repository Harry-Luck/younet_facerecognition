import argparse

import numpy as np
import cv2 as cv

from yunet import YuNet

def str2bool(v):
    if v.lower() in ['on', 'yes', 'true', 'y', 't']:
        return True
    elif v.lower() in ['off', 'no', 'false', 'n', 'f']:
        return False
    else:
        raise NotImplementedError

backends = [cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_BACKEND_CUDA]
targets = [cv.dnn.DNN_TARGET_CPU, cv.dnn.DNN_TARGET_CUDA, cv.dnn.DNN_TARGET_CUDA_FP16]
help_msg_backends = "Choose one of the computation backends: {:d}: OpenCV implementation (default); {:d}: CUDA"
help_msg_targets = "Chose one of the target computation devices: {:d}: CPU (default); {:d}: CUDA; {:d}: CUDA fp16"
try:
    backends += [cv.dnn.DNN_BACKEND_TIMVX]
    targets += [cv.dnn.DNN_TARGET_NPU]
    help_msg_backends += "; {:d}: TIMVX"
    help_msg_targets += "; {:d}: NPU"
except:
    print('This version of OpenCV does not support TIM-VX and NPU. Visit https://gist.github.com/fengyuentau/5a7a5ba36328f2b763aea026c43fa45f for more information.')
path = '1.jpg'
parser = argparse.ArgumentParser(description='YuNet: A Fast and Accurate CNN-based Face Detector (https://github.com/ShiqiYu/libfacedetection).')

parser.add_argument('--model', '-m', type=str, default='face_detection_yunet_2022mar.onnx', help='Path to the model.')
parser.add_argument('--backend', '-b', type=int, default=backends[0], help=help_msg_backends.format(*backends))
parser.add_argument('--target', '-t', type=int, default=targets[0], help=help_msg_targets.format(*targets))
parser.add_argument('--conf_threshold', type=float, default=0.9, help='Filter out faces of confidence < conf_threshold.')
parser.add_argument('--nms_threshold', type=float, default=0.3, help='Suppress bounding boxes of iou >= nms_threshold.')
parser.add_argument('--top_k', type=int, default=5000, help='Keep top_k bounding boxes before NMS.')
parser.add_argument('--save', '-s', type=str, default=False, help='Set true to save results. This flag is invalid when using camera.')
parser.add_argument('--vis', '-v', type=str2bool, default=True, help='Set true to open a window for result visualization. This flag is invalid when using camera.')
args = parser.parse_args()

url = 'rtsp://admin:@122.176.110.134:554/ch0_0.264'

def visualize(image, results, box_color=(0, 255, 0), text_color=(0, 0, 255), fps=None):
    output = image.copy()
    landmark_color = [
        (255,   0,   0), # right eye
        (  0,   0, 255), # left eye
        (  0, 255,   0), # nose tip
        (255,   0, 255), # right mouth corner
        (  0, 255, 255)  # left mouth corner
    ]

    if fps is not None:
        cv.putText(output, 'FPS: {:.2f}'.format(fps), (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, text_color)

    for det in (results if results is not None else []):
        bbox = det[0:4].astype(np.int32)
        cv.rectangle(output, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), box_color, 2)

        conf = det[-1]
        cv.putText(output, '{:.4f}'.format(conf), (bbox[0], bbox[1]+12), cv.FONT_HERSHEY_DUPLEX, 0.5, text_color)

        landmarks = det[4:14].astype(np.int32).reshape((5,2))
        # for idx, landmark in enumerate(landmarks):
        #     cv.circle(output, landmark, 2, landmark_color[idx], 2)

    return output


    # Instantiate YuNet
model = YuNet(modelPath=args.model,
                inputSize=[320, 320],
                confThreshold=args.conf_threshold,
                nmsThreshold=args.nms_threshold,
                topK=args.top_k,
                backendId=args.backend,
                targetId=args.target)

deviceId = 'rtsp://admin:@122.176.110.134:554/ch0_0.264'
cap = cv.VideoCapture(deviceId)
# w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
# h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
w = 600
h = 600
# w = int(w/2)
# h = int(h/2)
model.setInputSize([w, h])

tm = cv.TickMeter()
while cv.waitKey(1) < 0:
    hasFrame, frame = cap.read()
    if not hasFrame:
        print('No frames grabbed!')
        break
    frame = cv.resize(frame, (w, h))
    # Inference
    tm.start()
    results = model.infer(frame) # results is a tuple
    tm.stop()

    # Draw results on the input image
    frame = visualize(frame, results, fps=tm.getFPS())

    # Visualize results in a new Window
    cv.imshow('YuNet Demo', frame)

    tm.reset()

# Save results if save is true
