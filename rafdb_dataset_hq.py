import numpy as np
import time
import random
import cv2
import sys
import os
from dataset_tools import enclosing_square, add_margin, cut
import keras
from tqdm import tqdm
from dataset_tools import _readcsv, cntk_filtering
from dataset_tools import linear_balance_illumination, mean_std_normalize, equalize_hist
from dataset_tools import draw_emotion
from dataset_tools import DataGenerator

from six.moves import cPickle as pickle

NUM_TRAINING_SAMPLES = 12271
NUM_CLASSES = 7

    


def _crop_align_rafdb_face(im, ann, TARGET_SIZE = (256,256)):
    p = []
    for i in range(5):
        p.append( [float(x) for x in ann[i].strip().replace(' ','\t').split('\t')] )
    pmouth=( (p[4][0]+p[3][0])//2 , (p[4][1]+p[3][1])//2 )
    srcTri = np.array( [ p[0], p[1], pmouth ] ).astype(np.float32)
    dstTri = np.array( [ [.286, .4], [.714, .4], [.50, .714] ] ).astype(np.float32) * (np.array(TARGET_SIZE, dtype=np.float32).reshape((1,2)))

    dh = TARGET_SIZE[0] - im.shape[0]
    dw = TARGET_SIZE[1] - im.shape[1]
    if dh > 0:
        pad = np.zeros((dh, im.shape[1], im.shape[2]), dtype=np.uint8)
        im = np.append(im, pad, axis=0)
    if dw > 0:
        pad = np.zeros((im.shape[0], dw, im.shape[2]), dtype=np.uint8)
        im = np.append(im, pad, axis=1)

    A = cv2.getAffineTransform(srcTri,dstTri)
    im = cv2.warpAffine(im, A, (im.shape[1], im.shape[0]))

    return im[0:TARGET_SIZE[0],0:TARGET_SIZE[1],:]


def _load_rafdb(meta, imagesdir, partition):
    data = []
    n_discarded=0
    n_discarded_cntk=0
    for n,d in enumerate(tqdm(meta)):
        actualpartition = 'train' if n<NUM_TRAINING_SAMPLES else 'test'
        if actualpartition == partition:
            imgid = '%05d'%(n+1) if n<NUM_TRAINING_SAMPLES else '%04d'%(n+1-NUM_TRAINING_SAMPLES)
            drop, labels = cntk_filtering(d, rowtotal=1, num_classes=NUM_CLASSES)
            if not drop:
                path = os.path.join(imagesdir,'%s_%s.jpg'%(partition,imgid))
                annpath= os.path.join(imagesdir, '..', '..', 'Annotation', 'manual','%s_%s_manu_attri.txt'%(partition,imgid))
                ann = [l for l in open(annpath)]
                img = cv2.imread(path)
                img = _crop_align_rafdb_face(img, ann)
                if img is not None:
                    example={
                        'img': img,
                        'label': labels,
                        'roi': (16,16,224,224)
                        }
                    if np.max(example['img'])==np.min(example['img']):
                        print('Warning, blank image!')
                    else:
                        data.append(example)
                else: # img is None
                    print("WARNING! Unable to read %s" % path)
                    n_discarded+=1
            else: # ambiguous label
                n_discarded_cntk+=1
    print("Data loaded. %d samples (%d+%d discarded)" % (len(data), n_discarded, n_discarded_cntk) )
    return data

class RAFDBDataset:
    def __init__(self, partition='train', imagesdir='RAF-DB/basic/Image/original', csvmeta='RAF-DB/distribute_basic.csv', target_shape=(224,224,3), augment=True, custom_augmentation=None, preprocessing='full_normalization', debug_max_num_samples=None):
        if partition.startswith('train'):
            partition='train'
        elif partition.startswith('val'):
            print('WARNING: this dataset only has one test partition for test and validation')
            partition='test'
        elif partition.startswith('test'):
            print('WARNING: this dataset only has one test partition for test and validation')
            partition='test'
        else:
            raise Exception("unknown partition")

        self.target_shape = target_shape
        self.custom_augmentation = custom_augmentation
        self.augment = augment
        self.gen = None
        self.preprocessing = preprocessing
        print('Loading data...')
        
        cache_file_name = 'hq%s.%s%s.cache'%(imagesdir.replace('/','_'),partition, '.'+str(debug_max_num_samples) if debug_max_num_samples is not None else '')
        try:
            with open(cache_file_name, 'rb') as f:
                self.data = pickle.load(f)
                self.data = self.data[:debug_max_num_samples]
                print("Data loaded. %d samples, from cache" % (len(self.data)) )
        except FileNotFoundError:
            meta = _readcsv(csvmeta)
            print('csv read complete: %d.' %(len(meta)))
            self.data = _load_rafdb(meta, imagesdir, partition)
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
     
           


def test1():
    print('Training')
    dt = RAFDBDataset(target_shape=(224,224,3), preprocessing='full_normalization', debug_max_num_samples=None, augment=False)
    print('Test')
    dv = RAFDBDataset('test',target_shape=(200,200,3), preprocessing='full_normalization', debug_max_num_samples=None)
    
    print('Now generating from training set')
    gen = dt.get_generator()
    i=0
    while True:
        print(i)
        i+=1
        for batch in tqdm(gen):
            pass


EMOTION_LABELS=['surprise','fear','disgust','happiness','sadness','anger','neutral']

def test2():
    dt = RAFDBDataset(target_shape=(200,200,3), preprocessing='full_normalization', debug_max_num_samples=None, augment=False)
    gen = dt.get_generator()
    for batch in gen:
        for x,y in zip(batch[0], batch[1]):
            window = np.zeros((400,200,3), dtype=np.uint8)
            MAX = np.amax(x)
            MIN = np.amin(x)
            x = 255*(x-MIN)/(MAX-MIN)
            x = x.clip(0,255).astype(np.uint8)
            if len(x.shape)<=2 or x.shape[2]==1:
                x = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
            window[0:200,0:200,:] = x
            window[200:400,0:200,:] = draw_emotion(y,200,200, emotion_labels=EMOTION_LABELS)
            cv2.imshow('im', window)
            k = cv2.waitKey(0)
            if k==27:
                sys.exit(0)

if '__main__' == __name__:
    test2()
