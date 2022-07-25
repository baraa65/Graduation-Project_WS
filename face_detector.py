import numpy as np
from cv2 import cv2

class Detector:
    def __init__(self):
        self.faceModel = cv2.dnn.readNetFromCaffe('models/res10_300x300_ssd_iter_140000.prototxt', 'models/res10_300x300_ssd_iter_140000.caffemodel')

    def processImage(self, img):
        (height, width) = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 127.0, 123.0), swapRB=False, crop=False)
        self.faceModel.setInput(blob)
        predictions = self.faceModel.forward()

        results = []

        for i in range(predictions.shape[2]):
            if predictions[0, 0, i, 2] > 0.5:
                bbox = predictions[0, 0, i, 3:7] * np.array([width, height, width, height])
                results.append(bbox.astype('int'))

        return results
