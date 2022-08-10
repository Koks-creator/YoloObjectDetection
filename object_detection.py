from dataclasses import dataclass
from time import time
from typing import Tuple
import cv2
import numpy as np

np.random.seed(20)


@dataclass
class ObjectDetection:
    weights_path: str = r"yolov4-tiny.weights"  # https://github.com/AlexeyAB/darknet/releases
    cfg_path: str = r"yolov4-tiny.cfg"  # https://github.com/AlexeyAB/darknet/tree/master/cfg
    classes_file = r"classes.txt"
    nms_threshold: int = .3
    conf_threshold: int = .3
    image_w: int = 416  # multiple of 32, for yolov4-tiny it's 416, for normal yolov4 it's 608 but you can play around
    # with it, the bigger w and h the more accuracy at the expense of fps
    image_h: int = 416

    net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def __post_init__(self):
        self.classes_list, self.colors_list = self.load_classes()

    def load_classes(self):
        with open(self.classes_file, "rt") as f:
            class_names = f.read().rstrip('\n').split("\n")
            color_list = np.random.uniform(low=0, high=255, size=(len(class_names), 3))
            return class_names, color_list

    def detect(self, img: np.array, allowed_classes=False, draw=False) -> Tuple[np.array, list]:
        ih, iw, _ = img.shape
        bbox = []
        class_ids = []
        confs = []

        if allowed_classes is False:
            allowed_classes = [i for i in range(len(self.classes_list))]

        blob = cv2.dnn.blobFromImage(img, 1 / 255, (self.image_w, self.image_h), [0, 0, 0], 1, crop=False)
        self.net.setInput(blob)

        layer_names = self.net.getLayerNames()
        output_names = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        outputs = self.net.forward(output_names)

        for output in outputs:
            for det in output:
                # print(cos[5:])
                scores = det[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if class_id in allowed_classes:
                    if confidence > self.conf_threshold:
                        w, h = int(det[2] * iw), int(det[3] * ih)
                        x, y = int((det[0] * iw) - w / 2), int((det[1] * ih) - h / 2)

                        bbox.append([x, y, w, h])
                        class_ids.append(class_id)
                        # print(confidence)
                        confs.append(float(confidence))
                        # print(confs)

        # to wywala zbyt duza ilosc bboxow, zwraca id odpowiedniego bboxa w liscie bboxow
        indices = cv2.dnn.NMSBoxes(bbox, confs, self.conf_threshold, self.nms_threshold)

        bbox_list = []
        for i in indices:
            i = i[0]  # bo i jest lista, np [1]
            # print(confs[i])
            box = bbox[i]
            # print(class_ids)
            # print(classes_list)
            x, y, w, h = box[0], box[1], box[2], box[3]
            class_name = self.classes_list[class_ids[i]].upper()
            class_color = [int(c) for c in self.colors_list[class_ids[i]]]
            bbox_list.append([x, y, w, h, class_name, class_color])
            if draw:
                cv2.rectangle(img, (x, y), (x + w, y + h), class_color, 2)
                cv2.putText(img, f"{self.classes_list[class_ids[i]].upper()} {int(confs[i] * 100)}%", (x, y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, class_color, 2)
        return img, bbox_list


if __name__ == '__main__':
    od = ObjectDetection()

    # Example with image
    # img = cv2.imread(r"Medias\Livingroom.jpg")
    # img, boxes = od.detect(img, draw=True)
    # cv2.imshow("res", img)
    # cv2.waitKey(0)

    # Example with video
    cap = cv2.VideoCapture(r"Medias\kuchnia.mp4")

    p_time = 0
    while True:
        success, img = cap.read()
        img = cv2.resize(img, (1280, 720))
        img, boxes = od.detect(img, draw=True)
        for box in boxes:
            x, y, w, h = box[:4]
            cx, cy = x + w // 2, y + h // 2
            p1 = x, y
            p2 = x + w, y + h

        c_time = time()
        fps = int(1 / (c_time - p_time))
        p_time = c_time

        cv2.putText(img, f"FPS: {fps}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 50, 100), 2)

        cv2.imshow("Res", img)
        cv2.waitKey(1)
