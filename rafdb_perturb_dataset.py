#!/usr/bin/python3
import os
import sys

from glob import glob
from tqdm import tqdm
import cv2
import numpy as np
from corruptions import custom_motion_blur
from corruptions import zoom_blur, defocus_blur, gaussian_noise, saturate, contrast, brightness, elastic_transform, spatter, jpeg_compression, shot_noise
from skimage.filters import gaussian
from PIL import Image

import concurrent.futures

#from ferplus_aug_dataset import MyCustomAugmentation, corruptions
IN_IMG_SIZ = 130
OUT_IMG_SIZ = 100
N_PERTUB_FRAMES = 30

def p_brightness(x,roi):
    outlist=[]
    for i in range(N_PERTUB_FRAMES):
        outx = brightness(x, c=(i - 15) * 2 / 100.)
        outlist.append((outx,roi))
    return outlist

def p_translate(x,roi):
    outlist=[]
    for i in range(N_PERTUB_FRAMES):
        outx = x
        roi = list(roi)
        roi[0]=i
        outlist.append((outx,tuple(roi)))
    return outlist

def p_rotate(x,roi):
    outlist=[]
    for i in range(N_PERTUB_FRAMES):
        cX = (roi[2]+2*roi[0])/2
        cY = (roi[3]+2*roi[1])/2
        M = cv2.getRotationMatrix2D((cX, cY), i-N_PERTUB_FRAMES//2, 1.0)
        outx = cv2.warpAffine(x, M, (roi[0]+roi[2], roi[1]+roi[3]))
        outlist.append((outx,roi))
    return outlist

def p_shear(x,roi):
    outlist=[]
    for i in range(N_PERTUB_FRAMES):
        cX = (roi[2]+2*roi[0])/2
        cY = (roi[3]+2*roi[1])/2
        shear=0.01*(i-N_PERTUB_FRAMES/2)
        M = np.float64([[1,shear,0], [shear,1,0]])
        outx = cv2.warpAffine(x, M, (roi[0]+roi[2], roi[1]+roi[3]))
        outlist.append((outx,roi))
    return outlist

def p_motion_blur(x,roi):
    outlist=[]
    for i in range(N_PERTUB_FRAMES):
        outx=custom_motion_blur(x,10,3,(i-N_PERTUB_FRAMES)*4)
        outlist.append((outx,roi))
    return outlist

def p_scale(x,roi):
    outlist=[]
    for i in range(N_PERTUB_FRAMES):
        s = int( IN_IMG_SIZ - 1.7*(IN_IMG_SIZ - OUT_IMG_SIZ)*(i/N_PERTUB_FRAMES) )
        off = (IN_IMG_SIZ - s)//2
        roi = (off,off,s,s)
        outlist.append((x,tuple(roi)))
    return outlist


def p_gaussian_blur(x,roi):
    outlist=[]
    for i in range(N_PERTUB_FRAMES):
        outx = np.uint8(255*gaussian(np.array(x, copy=True)/255., sigma=0.25 + 0.035*i, multichannel=True, truncate=6.0))
        outlist.append((outx,roi))
    return outlist

def p_spatter(x,roi):
    x = cv2.cvtColor(np.array(x, dtype=np.float32) / 255., cv2.COLOR_BGR2BGRA)

    liquid_layer = np.random.normal(size=x.shape[:2], loc=0.65, scale=0.27)
    liquid_layer = gaussian(liquid_layer, sigma=3.7)
    liquid_layer[liquid_layer < 0.69] = 0

    outlist=[]
    for i in range(N_PERTUB_FRAMES):
        liquid_layer_i = (liquid_layer * 255).astype(np.uint8)
        dist = 255 - cv2.Canny(liquid_layer_i, 50, 150)
        dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
        _, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)
        dist = cv2.blur(dist, (3, 3)).astype(np.uint8)
        dist = cv2.equalizeHist(dist)
        ker = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        dist = cv2.filter2D(dist, cv2.CV_8U, ker)
        dist = cv2.blur(dist, (3, 3)).astype(np.float32)
        m = cv2.cvtColor(liquid_layer_i * dist, cv2.COLOR_GRAY2BGRA)
        m /= np.max(m, axis=(0, 1))
        m *= 0.6
        color = np.concatenate((175 / 255. * np.ones_like(m[..., :1]),
                                238 / 255. * np.ones_like(m[..., :1]),
                                238 / 255. * np.ones_like(m[..., :1])), axis=2)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2BGRA)
        z = np.uint8(cv2.cvtColor(np.clip(x + m * color, 0, 1), cv2.COLOR_BGRA2BGR) * 255)
        liquid_layer = np.apply_along_axis(lambda mat:
                                        np.convolve(mat, np.array([0.05, 0.1, 0.15, 0.7]), mode='same'),
                                        axis=0, arr=liquid_layer)
        outlist.append((z,tuple(roi)))
    return outlist

def p_gaussian_noise(x,roi):
    outlist=[]
    for i in range(N_PERTUB_FRAMES):
        if i>0:
            outx=gaussian_noise(x,2)
        else:
            outx=x
        outlist.append((outx,roi))
    return outlist

def p_shot_noise(x,roi):
    outlist=[]
    for i in range(N_PERTUB_FRAMES):
        if i>0:
            outx=shot_noise(x,2)
        else:
            outx=x
        outlist.append((outx,roi))
    return outlist

perturbations = [
    p_gaussian_noise,
    p_shot_noise,
    p_gaussian_blur,
    p_motion_blur,
    p_spatter,
    p_brightness,
    p_translate,
    p_rotate,
    p_scale,
    p_shear,
]


from rafdb_dataset_hq import _crop_align_rafdb_face

def _cut_and_resize(inp):
    pim,proi = inp
    im = pim[proi[1]:proi[1]+proi[3], proi[0]:proi[0]+proi[2],:]
    if proi[2] != OUT_IMG_SIZ or proi[3] != OUT_IMG_SIZ:
        im = cv2.resize(im,(OUT_IMG_SIZ, OUT_IMG_SIZ))
    return im

def _process_ds(dirin="RAF-DB/basic/Image/original"):
    allims = glob(os.path.join(dirin, 'test_*'))
    def proc_im(impath):
        imname = os.path.basename(impath)[:-4]
        annpath= os.path.join(dirin, '..', '..', 'Annotation', 'manual','%s_manu_attri.txt'%(imname))
        ann = [l for l in open(annpath)]
        im = cv2.imread(impath)
        x = _crop_align_rafdb_face(im, ann, (IN_IMG_SIZ,IN_IMG_SIZ))
        OFFSET = (IN_IMG_SIZ-OUT_IMG_SIZ)//2
        roi = (OFFSET,OFFSET,OUT_IMG_SIZ,OUT_IMG_SIZ)
        pertubed_images = [ [_cut_and_resize(pimandroi) for pimandroi in p(x,roi)] for p in perturbations]
        return pertubed_images, impath
    for i in allims:
        yield proc_im(i)
    #with concurrent.futures.ThreadPoolExecutor(4) as executor:
    #   for i in executor.map(proc_im, allims):
    #       yield i

def show_one_image_anim():
    for pertubed_images, _ in _process_ds():
        break
    imout = np.ones((OUT_IMG_SIZ+20, len(perturbations)*OUT_IMG_SIZ,3), np.uint8)*255
    while True:
        #import imageio
        #with imageio.get_writer('perturbations.gif', mode='I') as writer:
        for fn in range(N_PERTUB_FRAMES):
            for i, o in enumerate(pertubed_images):
                pim = o[fn]
                # draw
                imout[0:OUT_IMG_SIZ, i*OUT_IMG_SIZ:(i+1)*OUT_IMG_SIZ, :] = pim
                cv2.putText(imout, perturbations[i].__name__, (5+i*OUT_IMG_SIZ,OUT_IMG_SIZ+14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 2)
            #writer.append_data(cv2.cvtColor(imout, cv2.COLOR_BGR2RGB))
            cv2.imshow('imout', imout)
            k = cv2.waitKey(50)
            if k==27:
                sys.exit(0)

def show_one_image():
    MARGIN=2
    OUT_IMG_SPACE=OUT_IMG_SIZ+MARGIN
    CAPTION_SPACE = 20
    DECIM = 4
    all_pims = []
    for pertubed_images, _ in _process_ds():
        all_pims.append(pertubed_images)
        if len(all_pims)==len(pertubed_images):
            break
    from math import ceil
    imout = np.ones( (len(perturbations)*(OUT_IMG_SPACE), ceil(N_PERTUB_FRAMES/DECIM)*(OUT_IMG_SPACE)+CAPTION_SPACE, 3), np.uint8)*255
    while True:
        for fi,fn in enumerate(range(0,N_PERTUB_FRAMES,DECIM)): # iterate frames
            for i in range(len(pertubed_images)): # iterate different perturbations
                print(fi, fn)
                o = all_pims[i][i]
                pim = o[fn]
                # draw
                imout[(i*OUT_IMG_SPACE):((i+1)*OUT_IMG_SPACE)-MARGIN, CAPTION_SPACE+OUT_IMG_SPACE*fi:CAPTION_SPACE+OUT_IMG_SPACE*(fi+1)-MARGIN, :] = pim
        captionimg= np.ones( (CAPTION_SPACE, len(perturbations)*OUT_IMG_SPACE, 3), np.uint8)*255
        names = [p.__name__[2:] for p in perturbations]
        names.reverse()
        for i,n in enumerate(names):
            cv2.putText(captionimg, n, (5+i*OUT_IMG_SPACE, CAPTION_SPACE//2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
        h,w,_ = captionimg.shape
        #captionimg = cv2.transpose(captionimg)
        #captionimg = cv2.flip(captionimg, 0)
        #imout[:, 0:CAPTION_SPACE, :] = captionimg
        cv2.imwrite('perturbations.png', imout)
        '''cv2.imshow('imout', imout)
        k = cv2.waitKey(0)
        if k==27:
            sys.exit(0)'''
        sys.exit(0)

def export_dataset(diroutpattern='perturbed_raf_dataset/rafdb.%s.%s/', dirin='RAF-DB/basic/Image/original', partition='test'):
    for pertubed_images, inim in tqdm(_process_ds()):
        for pid,pimages in enumerate(pertubed_images):
            perturbation = perturbations[pid].__name__
            dirout=diroutpattern%(partition,perturbation)
            if not os.path.exists(dirout):
                os.mkdir(dirout)
            for fn, pframe in enumerate(pimages):
                outim = os.path.join(dirout, inim[len(dirin)+1:-4]+ ('_%02d'%fn) + '.jpg' )
                cv2.imwrite(outim, pframe)
    return dirout
    
    
if '__main__' == __name__:
    if len(sys.argv)>1 and sys.argv[1].startswith('exp'):
        export_dataset()
    else:
        show_one_image()
