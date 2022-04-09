import cv2
import numpy as np
import os 
import dlib

PATH = os.path.join( os.path.dirname(os.path.realpath(__file__)), "models", "shape_predictor_68_face_landmarks.dat")

def _get_part(shape, n):
    l = shape.part(n)
    return (l.x, l.y)

class FaceAligner:
    predictor = None
    def __init__(self):
        print ("FaceAligner -> init")
        self.predictor = dlib.shape_predictor(PATH)
        print ("FaceAligner -> init ok")


    def get_landmarks(self, image, box):
        box = dlib.rectangle(box[0], box[1], box[0]+box[2], box[1]+box[3])
        shape = self.predictor(image, box)
        arr = []
        for i in range(64):
            arr.append(_get_part(shape, i))
        return arr
        #return [_get_part(shape,36), _get_part(shape,45), _get_part(shape,33), _get_part(shape,66)]
    
    def __del__(self):
        print ("FaceAligner -> bye")
