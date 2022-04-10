import numpy as np
import time
from math import ceil
import random
import cv2
import sys
import os
from dataset_tools import enclosing_square, add_margin, cut
import keras
from tqdm import tqdm
from dataset_tools import DataGenerator
from dataset_tools import _readcsv
from dataset_tools import linear_balance_illumination, mean_std_normalize, equalize_hist
from six.moves import cPickle as pickle
from threading import Lock

NUM_CLASSES = 8          

def _fer_to_img(line_arr):
    data = [int(x) for x in line_arr[1].split(' ')]
    data = np.array(data, dtype=np.uint8)
    data = np.reshape(data, (48,48,1))
    #data = cv2.cvtColor(data, cv2.COLOR_GRAY2RGB)
    return data



def cntk_filtering(data):
    # remove outlier votes
    data = np.array([int(x) for x in data])
    outliers = data<=1
    data[outliers] = 0
    
    totalvotes = np.sum(data)
    
    # remove examples from class 9 or 10
    hardlabel = np.argmax(data)
    if hardlabel == 8 or hardlabel==9:
        return True, None

    # remove examples with more than two winners
    maxvotes = np.max(data)
    winners = data==maxvotes
    nwinners = np.sum(winners)
    if nwinners > 2:
        return True, None

    # remove examples where the winners have <=50% of all votes
    numwinnervotes = nwinners*maxvotes
    if numwinnervotes <= 0.5*totalvotes:
        return True, None

    # return normalized
    data = data.astype(float)/totalvotes
    return False, data[0:NUM_CLASSES]

def _load_ferplus(meta, images, partition):
    data = []
    n_discarded=0
    n_discarded_cntk=0
    for n,d in enumerate(tqdm(meta[1:])):
        if d[0]==partition:
            if d[1]=='':
                n_discarded+=1
            else:
                try:
                    drop, labels = cntk_filtering(d[2:])
                    #n = int(d[1][3:10])
                    #print(d[1],n)
                    if not drop:
                        example={
                            'img': _fer_to_img(images[n+1]),
                            'label': labels,
                            'roi': (0,0,48,48)
                            }
                        if np.max(example['img'])==np.min(example['img']):
                            print('Warning, blank image!')
                        else:
                            data.append(example)
                    else:
                        n_discarded_cntk+=1
                except (IndexError) as e:
                    print('not found: ' +str(e))
                    pass
    print("Data loaded. %d samples (%d+%d discarded)" % (len(data), n_discarded, n_discarded_cntk) )
    return data

class FerPlusDataset:
    def __init__(self, partition='train', csvdata='FERPlus/fer2013.csv', csvmeta='FERPlus/fer2013new.csv', target_shape=(48,48,1), augment=True, custom_augmentation=None, preprocessing='full_normalization', debug_max_num_samples=None):
        if partition.startswith('train'):
            partition='Training'
        elif partition.startswith('val'):
            partition='PrivateTest'
        elif partition.startswith('test'):
            partition='PublicTest'
        else:
            raise Exception("unknown partition")

        self.target_shape = target_shape
        self.custom_augmentation = custom_augmentation
        self.augment = augment
        self.gen = None
        self.preprocessing = preprocessing
        print('Loading data...')
        
        cache_file_name = '%s.%s%s.cache'%(csvdata,partition, '.'+str(debug_max_num_samples) if debug_max_num_samples is not None else '')
        try:
            with open(cache_file_name, 'rb') as f:
                self.data = pickle.load(f)
                self.data = self.data[:debug_max_num_samples]
                print("Data loaded. %d samples, from cache" % (len(self.data)) )
        except FileNotFoundError:
            meta = _readcsv(csvmeta)
            images = _readcsv(csvdata)
            print('csv read complete: %d, %d.' %(len(meta), len(images)))
            self.data = _load_ferplus(meta, images, partition)
            with open(cache_file_name, 'wb') as f:
                pickle.dump(self.data, f)

        
    
    
    def get_num_samples(self):
        return self.data.shape[0]
        
    def get_num_classes(self):
        return NUM_CLASSES
        
    def get_generator(self, batch_size=64):
        if self.gen is None:
            self.gen = DataGenerator(self.data, self.target_shape, with_augmentation=self.augment, custom_augmentation=self.custom_augmentation, batch_size=batch_size, preprocessing=self.preprocessing)
        return self.gen
     
           


CLASS_LABELS = ['neutral','happiness','surprise','sadness','anger','disgust','fear','contempt','unknown','NF']
def draw_emotion(y, w,h):
    EMOTIONS=CLASS_LABELS
    COLORS = [(120,120,120), (50,50,255), (0,255,255), (255,0,0), (0,0,140), (0,200,0), (42,42,165), (100,100,200), (170,170,170), (80,80,80)]
    emotionim = np.zeros((w,h,3), dtype=np.uint8)
    barh = h//len(EMOTIONS)
    MAXEMO = np.sum(y)
    for i,yi in enumerate(y):
        #print((EMOTIONS[i], yi))
        p1,p2 = (0,i*barh), (int(yi*w//MAXEMO), (i+1)*20)
        cv2.rectangle(emotionim, p1,p2, COLORS[i], cv2.FILLED)
        cv2.putText(emotionim, "%s: %.1f" % (EMOTIONS[i], yi), (0,i*20+14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255))
    return emotionim

            

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
def top_left(f):
    return (f['roi'][0], f['roi'][1])
def bottom_right(f):
    return (f['roi'][0]+f['roi'][2], f['roi'][1]+f['roi'][3])

def dump_ds(dataset,name,h5file):
    dt=dataset
    tgen = dt.get_generator(1)
    Xt=[]
    Yt=[]
    for batch in tqdm(tgen):
        Xt.append(batch[0])
        Yt.append(batch[1])
    Xt=np.squeeze(np.array(Xt))
    print(Xt.shape, Xt.dtype)
    Yt=np.squeeze(np.array(Yt))
    dsfile.create_dataset('X'+name, data=Xt)
    dsfile.create_dataset('Y'+name, data=Yt)

def test1():
    print('Training')
    dt = FerPlusDataset(target_shape=(224,224,3), preprocessing='full_normalization', debug_max_num_samples=None)
    
    print('Now generating from training set')
    gen = dt.get_generator()
    i=0
    while True:
        print(i)
        i+=1
        for batch in tqdm(gen):
            for im, identity in zip(batch[0], batch[1]):
                identity = np.argmax(identity)
                facemax = np.max(im)
                facemin = np.min(im)
                im = (255*( (im - facemin) / (facemax - facemin) )).astype(np.uint8)
                cv2.imshow('ferplus image', im)
                cv2.waitKey(0) 


if '__main__' == __name__:
    test1()

def dumpall():
    print('Training')
    dt = FerPlusDataset(target_shape=(224,224,3), preprocessing='full_normalization', debug_max_num_samples=None, augment=False)
    print('Validation')
    dv = FerPlusDataset('val',target_shape=(224,224,3), augment=False, preprocessing='full_normalization', debug_max_num_samples=None)
    print('Test')
    dp = FerPlusDataset('test',target_shape=(224,224,3), augment=False, preprocessing='full_normalization', debug_max_num_samples=None)
    
    print('Now generating from training set')
    
    import h5py
    dsfile=h5py.File('ds.h5','w')
    
    dump_ds(dt, 't',dsfile)
    dump_ds(dv, 'v',dsfile)
    dump_ds(dp, 's',dsfile)
    
    dsfile.close()
    '''
    from face_detector import FaceDetector
    from face_aligner import FaceAligner
    fd = FaceDetector(min_confidence=0.8)
    al = FaceAligner()

    for batch in gen:
        for x,y in zip(batch[0], batch[1]):
            window = np.zeros((400,200,3), dtype=np.uint8)
            x = ((x*127)+127).clip(0,255).astype(np.uint8)
            if len(x.shape)<=2 or x.shape[2]==1:
                x = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
            window[0:200,0:200,:] = x
            faces = fd.detect(x)
            #f = findRelevantFace(faces, x.shape[1], x.shape[0])
            for f in faces:
                print(f['roi'], f['confidence'])
                br = list(bottom_right(f))
                if br[1] > 200: br[1]=200
                if br[0] > 200: br[0]=200
                cv2.rectangle(window, top_left(f), tuple(br), (255, 255, 0), 2)
            window[200:400,0:200,:] = draw_emotion(y,200,200)
            cv2.imshow('im', window)
            cv2.waitKey(0)
            print("------------------------")
            print("------------------------")
            print("------------------------")
    '''
