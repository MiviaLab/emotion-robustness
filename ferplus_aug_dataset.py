#!/usr/bin/python3
import os
import sys
from dataset_tools import draw_emotion
from ferplus_dataset import FerPlusDataset as Dataset, NUM_CLASSES

from tqdm import tqdm
import cv2
import numpy as np
from corruptions import motion_blur
from corruptions import zoom_blur, pixelate, defocus_blur, gaussian_noise, gaussian_blur, saturate, contrast_plus, contrast, brightness_plus, brightness_minus, elastic_transform, spatter, jpeg_compression, shot_noise
from PIL import Image


class MyCustomAugmentation():
    def __init__(self, corruption_types, corruption_qtys):
        self.corruption_types = corruption_types
        self.corruption_qtys = corruption_qtys
        assert(len(corruption_types)==len(corruption_qtys))
    def __str__(self):
        if max(self.corruption_qtys) != min(self.corruption_qtys):
            s=[]
            for t,q in zip(self.corruption_types, self.corruption_qtys):
                s.append( "%s.%d" % (t.__name__, q) )
            return '.'.join(s)
        else:
            return '.'.join([t.__name__ for t in self.corruption_types]) + "." + str(self.corruption_qtys[0])
    def before_cut(self, img, roi=None):
        for t,q in zip(self.corruption_types, self.corruption_qtys):
            #print(t,q)
            if q > 0:
                img = t(img, q)
            if len(img.shape)<3:
                img = np.expand_dims(img,2)
            if img.dtype != np.uint8:
                img = img.clip(0,255).astype(np.uint8)
        #print(img.shape, img.dtype)
        return img
    def augment_roi(self, roi):
        return roi
    def after_cut(self, img):
        return img



def contrast_brightness_plus(x, severity):
    sb, sc = [(1,1), (2,1), (2,2), (2,3), (3,4)][severity-1]
    return contrast(brightness_plus(x, sb), sc)
def contrast_brightness_minus(x, severity):
    sb, sc = [(1,1), (2,1), (2,2), (2,3), (3,4)][severity-1]
    return contrast(brightness_minus(x, sb), sc)
    
def gaussian_noise_contrast_brightness_minus(x, severity):
    sg, sb, sc = [(1,1,1), (2,2,1), (2,2,2), (3,2,3), (3,2,4)][severity-1]
    return contrast(brightness_minus(gaussian_noise(x, sg), sb), sc)

def pixelate_contrast_brightness_minus(x, severity):
    sp, sb, sc = [(1,1,1), (2,2,1), (3,2,2), (4,2,1), (4,3,3)][severity-1]
    return contrast(brightness_minus(pixelate(x, sp), sb), sc)
    
def motion_blur_contrast_brightness_minus(x, severity):
    sm, sb, sc = [(2,1,1), (3,1,1), (4,2,2), (5,2,1), (5,2,3)][severity-1]
    return contrast(brightness_minus(motion_blur(x, sm), sb), sc)

corruptions=[
    [gaussian_blur],
    [defocus_blur,],
    [zoom_blur,],
    [motion_blur,],
    
    [gaussian_noise,],
    [shot_noise],
    
    [contrast_plus],
    [contrast,],
    [brightness_plus,],
    [brightness_minus,],
    [spatter,],
    [pixelate,],
    [jpeg_compression],
    
    [contrast_brightness_plus],
    [contrast_brightness_minus],
    [gaussian_noise_contrast_brightness_minus],
    [motion_blur_contrast_brightness_minus],
    [pixelate_contrast_brightness_minus],
]


def show_one_image():
    TARGET_SHAPE= (48,48,3)
    P = 'test'
    print('Partition: %s'%P)
    while True:
        NUM_LEVELS = 6
        imout = np.zeros( (TARGET_SHAPE[0]*len(corruptions),TARGET_SHAPE[1]*NUM_LEVELS,3), dtype=np.uint8 )
        print(imout.shape)
        for ind1,ctypes in enumerate(corruptions):
            for ind2 in range(NUM_LEVELS):
                a = MyCustomAugmentation(ctypes, [ind2]*len(ctypes))
                
                dataset_test = Dataset(partition=P, target_shape=TARGET_SHAPE,
                            debug_max_num_samples=1, augment=False, custom_augmentation=a)
                imex = np.squeeze(dataset_test.get_generator(1).__getitem__(0)[0],0)
                imex = ((imex*127)+127).clip(0,255).astype(np.uint8)
                
                #imex_corrupted = a.before_cut(imex)
                imex_corrupted = imex
                off1=ind1*TARGET_SHAPE[0]
                off2=ind2*TARGET_SHAPE[1]
                imout[off1:off1+TARGET_SHAPE[0],off2:off2+TARGET_SHAPE[1],:] = imex_corrupted

        #imout = cv2.resize(imout, (TARGET_SHAPE[0]*2, TARGET_SHAPE[1]*2))
        cv2.imshow('imout', imout)
        k = cv2.waitKey(0)
        if k==27:
            sys.exit(0)
'''
def export_datasets():
    NUM_LEVELS = 6
    TARGET_SHAPE= (48,48,3)
    P = 'PublicTest'
    print('Partition: %s'%P)
    for corruption_types in corruptions:
        print(corruption_types)
        for corruption_qty in range(NUM_LEVELS):
            a = MyCustomAugmentation(corruption_types, [corruption_qty]*len(corruption_types))
            dataset_test = Dataset(partition=P, target_shape=TARGET_SHAPE, debug_max_num_samples=None,
                    augment=False, custom_augmentation=a)
            ld = np.asarray( list(dataset_test.get_generator()) )
            X = np.asarray([ item for sublist in ld[:,0] for item in sublist ])
            y = np.asarray([ item for sublist in ld[:,1] for item in sublist ])
            print(X.shape)
            print(y.shape)
            for im,lbl zip(X,y):
'''
def export_dataset(augmentation, csvout='corrupted_dataset/fer2013.%s.%s.csv', csvdata='FERPlus/fer2013.csv', partition='PublicTest', csvmeta='FERPlus/fer2013new.csv'):
    from dataset_tools import _readcsv
    from ferplus_dataset import _fer_to_img
    csvout=csvout%(partition,str(augmentation))
    meta = _readcsv(csvmeta)
    images = _readcsv(csvdata)
    print(csvout)
    with open(csvout, 'w') as outf:
        for i,data in enumerate(images):
            if meta[i][0]==partition:
                im = _fer_to_img(data)
                imo=augmentation.before_cut(im)
                outf.write('-1,')
                outf.write(' '.join([str(x) for x in imo.flatten()]) )
            outf.write('\n')
    return csvout
    
def export_datasets():
    NUM_LEVELS = 5
    for corruption_types in corruptions:
        print(corruption_types)
        for corruption_qty in range(NUM_LEVELS):
            a = MyCustomAugmentation(corruption_types, [1+corruption_qty]*len(corruption_types))
            fname = export_dataset(a)

def test_export_dataset():
    a = MyCustomAugmentation([motion_blur], [5])
    fname = export_dataset(a)
    dp = Dataset('PublicTest', csvdata=fname, target_shape=(200,200), debug_max_num_samples=None, augment=False)

    gen = dp.get_generator()
    for batch in gen:
        for x,y in zip(batch[0], batch[1]):
            window = np.zeros((400,200,3), dtype=np.uint8)
            x = ((x*127)+127).clip(0,255).astype(np.uint8)
            if len(x.shape)<=2 or x.shape[2]==1:
                x = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
            window[0:200,0:200,:] = x
            window[200:400,0:200,:] = draw_emotion(y,200,200)
            cv2.imshow('im', window)
            k = cv2.waitKey(0)
            if k==27:
                sys.exit(0)
    

def run_test():

    batch_size = 64
    P = 'test'
    print('Partition: %s'%P)
    
    NUM_LEVELS = 1

    DATE="2019-09-29 11:36:42.014910"
    EPOCH=164

    dirnm="out_training_fer/"+DATE
    filepath=os.path.join(dirnm, "checkpoint.{epoch:02d}.hdf5")
    model = keras.models.load_model(  filepath.format(epoch=EPOCH)  )

    INPUT_SHAPE = (224,224,3)
    res_dict = {}

    for repeat in range(10):
        for corruption_qty in range(NUM_LEVELS):
            dataset_test = Dataset(partition=P, target_shape=INPUT_SHAPE, debug_max_num_samples=None,
                                augment=False, custom_augmentation=MyCustomAugmentation([defocus_blur], [corruption_qty]))
            result = model.evaluate_generator(dataset_test.get_generator(batch_size), verbose=1, workers=4)
            try:
                res_dict[corruption_qty].append(result[1])
            except KeyError:
                res_dict[corruption_qty]=[result[1]]
        for corruption_qty in range(NUM_LEVELS):
            res_list = res_dict[corruption_qty]
            print( "Blur %d: acc %.2f%% +- %.2f%%" %(corruption_qty,100*np.mean(res_list), 100*np.std(res_list) ) )

if '__main__' == __name__:
    show_one_image()
    #export_datasets()
