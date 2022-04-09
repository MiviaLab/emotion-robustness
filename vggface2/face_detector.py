import cv2
import numpy as np
import os 
PATH = os.path.join( os.path.dirname(os.path.realpath(__file__)), "models", "res10_face")
modelFile = os.path.join(PATH, "res10_300x300_ssd_iter_140000_fp16.caffemodel")
configFile=os.path.join(PATH, "deploy.prototxt")

class FaceDetector:
    net = None
    def __init__(self, min_confidence=0.5):
        print ("FaceDetector -> init")
        self.net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
        self.min_confidence = min_confidence
        print ("FaceDetector -> init ok")
    
    def detect(self, image):
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
        # TODO provare a cambiare width e height 300 300
        # TODO esportare confidence
        # TODO la faccia dev'essere quadrata?
        frameHeight, frameWidth, channels = image.shape
        self.net.setInput(blob)
        detections = self.net.forward()
        faces_result=[]
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.min_confidence:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                f = (x1,y1, x2-x1, y2-y1)
                if f[2]>1 and f[3]>1:
                    faces_result.append({
                        'roi': f,
                        'type': 'face',
                        'img': image[f[1]:f[1]+f[3], f[0]:f[0]+f[2]],
                        'confidence' : confidence
                    })
        return faces_result
    
    def __del__(self):
        print ("FaceDetector -> bye")
