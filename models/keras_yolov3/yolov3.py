from models.keras_yolov3.src.yolo import YOLO
from PIL import Image

YOLOV3 = YOLO

if __name__ == '__main__':
    import cv2
    vid = cv2.VideoCapture(0)
    yolo = YOLO()
    while True:
        return_value, frame = vid.read()
        image = Image.fromarray(frame)
        detections = yolo.detect(image)
        print(detections)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    yolo.close_session()