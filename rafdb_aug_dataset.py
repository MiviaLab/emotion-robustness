#!/usr/bin/python3
import os
import sys

from glob import glob
from tqdm import tqdm
import cv2
import numpy as np
from corruptions import motion_blur
from corruptions import zoom_blur, defocus_blur, gaussian_noise, saturate, contrast, brightness, elastic_transform, spatter, jpeg_compression, shot_noise
from PIL import Image

from multiprocessing import Pool


from ferplus_aug_dataset import MyCustomAugmentation, corruptions


def show_one_image(dirin="RAF-DB/basic/Image/aligned"):
    impaths = glob(os.path.join(dirin, '*'))
    
    MARGIN=2
    OUT_IMG_SIZ = 100
    OUT_IMG_SPACE=OUT_IMG_SIZ+MARGIN
    CAPTION_SPACE = 0#20
    TARGET_SHAPE= (OUT_IMG_SIZ,OUT_IMG_SIZ,3)
    P = 'test'
    print('Partition: %s'%P)
    while True:
        NUM_LEVELS = 6
        imout = np.ones( (OUT_IMG_SPACE*len(corruptions),OUT_IMG_SPACE*NUM_LEVELS,3), dtype=np.uint8 )*255
        #names = ['gaus. blur', 'defocus', 'zoom', 'motion', 'gaus. noise', 'shot noise', 'contrast inc.', 
        #'contrast dec.', 'bright. inc.', 'bright. dec.', 'spatter', 'pixelation', 'jpeg', 'mixed']
        #for i,n in enumerate(names):
        #    cv2.putText(imout, n, (5+i*OUT_IMG_SPACE, CAPTION_SPACE//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        print(imout.shape)
        for ind1,ctypes in enumerate(corruptions):
            imex = cv2.imread(impaths[ind1])
            for ind2 in range(NUM_LEVELS):
                a = MyCustomAugmentation(ctypes, [ind2]*len(ctypes))
                imex_corrupted = a.before_cut(imex, None)
                off1=ind1*OUT_IMG_SPACE
                off2=ind2*OUT_IMG_SPACE
                imout[off1:off1+OUT_IMG_SIZ,CAPTION_SPACE+off2:CAPTION_SPACE+off2+OUT_IMG_SIZ,:] = imex_corrupted

        #imout = cv2.resize(imout, (TARGET_SHAPE[0]*2, TARGET_SHAPE[1]*2))
        cv2.imwrite('corruptions.png', imout)
        '''cv2.imshow('imout', imout)
        k = cv2.waitKey(0)
        if k==27:
            sys.exit(0)'''
        sys.exit(0)

def export_dataset(augmentation, dirout='corrupted_raf_dataset/rafdb.%s.%s/', dirin='RAF-DB/basic/Image/aligned', partition='test'):
    dirout=dirout%(partition,str(augmentation))
    if not os.path.exists(dirout):
        os.mkdir(dirout)
    images = [x for x in glob(os.path.join(dirin, '*')) if os.path.basename(x).startswith(partition+'_')]
    for inim in tqdm(images):
        im = cv2.imread(inim)
        imo=augmentation.before_cut(im)
        outim = os.path.join(dirout, inim[len(dirin)+1:] )
        cv2.imwrite(outim, imo)
    return dirout
    
def export_datasets():
    NUM_LEVELS = 5
    aug_arr = []
    for corruption_types in corruptions:
        print(corruption_types)
        for corruption_qty in range(NUM_LEVELS):
            a = MyCustomAugmentation(corruption_types, [1+corruption_qty]*len(corruption_types))
            aug_arr.append(a)
    p = Pool(5)
    p.map(export_dataset, aug_arr)

    
if '__main__' == __name__:
    if len(sys.argv)>1 and sys.argv[1].startswith('exp'):
        export_datasets()
    else:
        show_one_image()
