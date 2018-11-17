"""Webcam demo for YOLOv3"""
import argparse
import cv2
from models.keras_yolov3 import YOLOV3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cam-id', '-c', type=int, default=0, help='Webcam id (default=0)')
    args = parser.parse_args()
    cam_id = args.cam_id
    vc = cv2.VideoCapture()
    if not vc.open(cam_id):
        raise IOError("Error opening webcam {}".format(cam_id))

    detector = YOLOV3()
    while True:
        _, img = vc.read()
        detections = detector.detect(image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img_draw = detector.draw_detections(img, detections)
        img_draw = cv2.resize(img_draw, (1280, 960))
        cv2.imshow("Detections", img_draw)
        key = cv2.waitKey(1)
        if key & 0xFFFF == 27:
            vc.release()
            cv2.destroyAllWindows()
            exit(0)


if __name__ == '__main__':
    main()
