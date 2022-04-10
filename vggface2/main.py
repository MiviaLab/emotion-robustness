import os
import sys
sys.path.append( os.path.dirname(__file__) )


import numpy as np
import cv2

from tqdm import tqdm



#cap = cv2.VideoCapture(0)

def top_left(f):
    return (f['roi'][0], f['roi'][1])
def bottom_right(f):
    return (f['roi'][0]+f['roi'][2], f['roi'][1]+f['roi'][3])

def enclosing_square(rect):
    def _to_wh(s,l,ss,ll, width_is_long):
        if width_is_long:
            return l,s,ll,ss
        else:
            return s,l,ss,ll
    def _to_long_short(rect):
        x,y,w,h = rect
        if w>h:
            l,s,ll,ss = x,y,w,h
            width_is_long = True
        else:
            s,l,ss,ll = x,y,w,h
            width_is_long = False
        return s,l,ss,ll,width_is_long

    s,l,ss,ll,width_is_long = _to_long_short(rect)

    hdiff = (ll - ss)//2
    s-=hdiff
    ss = ll

    return _to_wh(s,l,ss,ll,width_is_long)

def add_margin(roi, qty):
    return (
     roi[0]-qty,
     roi[1]-qty,
     roi[2]+2*qty,
     roi[3]+2*qty )

def cut(frame, roi):
    pA = ( int(roi[0]) , int(roi[1]) )
    pB = ( int(roi[0]+roi[2]), int(roi[1]+roi[3]) ) #pB will be an internal point
    W,H = frame.shape[1], frame.shape[0]
    A0 = pA[0] if pA[0]>=0 else 0
    A1 = pA[1] if pA[1]>=0 else 0
    data = frame[ A1:pB[1], A0:pB[0] ]
    if pB[0] < W and pB[1] < H and pA[0]>=0 and pA[1]>=0:
        return data
    w,h = int(roi[2]), int(roi[3])
    img = np.zeros((h,w,frame.shape[2]), dtype=np.uint8)
    offX = int(-roi[0]) if roi[0]<0 else 0
    offY = int(-roi[1]) if roi[1]<0 else 0
    np.copyto( img[ offY:offY+data.shape[0], offX:offX+data.shape[1] ], data )
    return img


def findRelevantFace(objs, W,H):
    mindistcenter = None
    minobj = None
    for o in objs:
        cx = o['roi'][0] + (o['roi'][2]/2)
        cy = o['roi'][1] + (o['roi'][3]/2)
        distcenter = (cx-(W/2))**2 + (cy-(H/2))**2
        if mindistcenter is None or distcenter < mindistcenter:
            mindistcenter = distcenter
            minobj = o
    return minobj



            
if '__main__' == __name__:
    from face_detector import FaceDetector
    from face_aligner import FaceAligner

    fd = FaceDetector()
    al = FaceAligner()


    TIME_LOADING = 0
    TIME_DETECTING = 0
    TIME_CROPPING = 0
    faces_found = 0
    images_processed = 0
    
    import sys
    import os
    from time import time
    fin_name = sys.argv[1]
    img_path = os.path.join( os.path.dirname(fin_name), os.path.basename(fin_name)[:-4] )
    fout_name = os.path.basename(fin_name)[:-4]+'.detected1.csv'
    fin = open(fin_name, "r")
    fout = open(fout_name, "w")
    for line in tqdm(fin):
        #if images_processed >= 2000:
        #    break
        TSTART = time()
        line = line.strip()
        line = line.split(',')
        img_fname = os.path.join(img_path,line[2])
        frame = cv2.imread(img_fname)
        TIME_LOADING += time()-TSTART
        ##########
        TSTART = time()
        faces = fd.detect(frame)
        images_processed+=1
        if len(faces) == 0:
            continue
        faces_found+=1
        #for f in faces:
        #    cv2.rectangle(frame, top_left(f), bottom_right(f), (0, 255, 0), 2)
        f = findRelevantFace(faces, frame.shape[1], frame.shape[0])
        TIME_DETECTING += time()-TSTART
        
        # save original roi
        fout.write(','.join(line))
        fout.write(',%d,%d,%d,%d\n' % f['roi'] )
        #cv2.rectangle(frame, top_left(f), bottom_right(f), (255, 255, 0), 2)
        #cv2.imshow('img', frame)
        #cv2.waitKey(0)
        '''
        TSTART = time()
        f['roi'] = enclosing_square(f['roi'])
        f['roi'] = add_margin(f['roi'], 0.2)
        img = cut(frame, f['roi'])
        TIME_CROPPING += time()-TSTART
        '''
        #cv2.imshow('img',f['img'])
        #cv2.waitKey(0)
        #cv2.imshow('img',img)
        #cv2.waitKey(0)
        #    landmarks = al.get_landmarks(frame, f['roi'])
        #    for l in landmarks:
        #        cv2.circle(frame, l, 2, (0,255,0), -1)
    
        # Display the resulting frame
        #cv2.imshow('frame',frame)
        #cv2.imshow('img',img)
        #if cv2.waitKey(100) & 0xFF == ord('q'):
        #    break
    
    cv2.destroyAllWindows()
    
    print("Time loading: %f" % TIME_LOADING)
    print("Time detecting: %f" % TIME_DETECTING)
    print("Time cropping: %f" % TIME_CROPPING)
    print("Faces found: %d/%d" % (faces_found, images_processed) )
