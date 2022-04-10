#!/usr/bin/python3
import keras
import sys, os
from glob import glob
#from rafdb_dataset import RAFDBDataset as Dataset
sys.path.append('keras_vggface/keras_vggface')
from antialiasing import BlurPool

import numpy as np
import cv2
from rafdb_perturb_dataset import N_PERTUB_FRAMES
from tqdm import tqdm
from dataset_tools import _readcsv, cntk_filtering, VGGFACE2_MEANS, mean_std_normalize
from rafdb_dataset import NUM_TRAINING_SAMPLES, NUM_CLASSES

from lbp_svm_train import LBPPredictor

from concurrent.futures import ThreadPoolExecutor
from psutil import virtual_memory
from time import sleep
from os.path import basename, dirname

def generate_perturbed_rafdb(imagesdir, target_shape=None, csvmeta='RAF-DB/distribute_basic.csv', preprocess=True):
    partition='test'
    meta = _readcsv(csvmeta)
    nmeta = list(enumerate(meta))
    nmeta = nmeta[NUM_TRAINING_SAMPLES:]
    print(len(nmeta))

    def _process(item):
        # Wait if memory full
        if virtual_memory().available < 2*1024*1024*1024:
            print('Low memory')
            sleep(2)
        # Do the actual task
        n,d = item
        actualpartition = 'train' if n<NUM_TRAINING_SAMPLES else 'test'
        if actualpartition == partition:
            imgid = '%05d'%(n+1) if n<NUM_TRAINING_SAMPLES else '%04d'%(n+1-NUM_TRAINING_SAMPLES)
            drop, labels = cntk_filtering(d, rowtotal=1, num_classes=NUM_CLASSES)
            if not drop:
                imgs = []
                for i in range(N_PERTUB_FRAMES):
                    path = os.path.join(imagesdir,'%s_%s_%02d.jpg'%(partition,imgid,i))
                    img = cv2.imread(path)
                    if img is not None:
                        if np.max(img)==np.min(img):
                            print('Warning, blank image!')
                        if preprocess:
                            #Preprocess a la vggface2
                            img = cv2.resize(img, target_shape[0:2])
                            img = mean_std_normalize(img, VGGFACE2_MEANS, None)
                        imgs.append(img)
                    else:
                        print("WARNING! Unable to read %s" % path)
                imgs = np.array(imgs)
                return imgs
                #print(imgs.shape, imgs.dtype)

    with ThreadPoolExecutor(max_workers=5) as executor:
        for imgs in executor.map(_process, nmeta):
            if imgs is not None:
                yield imgs
    #print("Finished. %d samples (%d+%d discarded)" % (len(data), n_discarded, n_discarded_cntk) )

from queue import Queue
def run_test(filepath):
    P = 'test'
    print('Partition: %s'%P)
    outf = open('perturbed_predictions/%s.txt'%basename(dirname(filepath)), "a+")
    if filepath.endswith('.pickle'):
        INPUT_SHAPE = None
        PREPROCESS = False
        model = LBPPredictor(filepath)
    else:
        model = keras.models.load_model(  filepath , custom_objects={'BlurPool':BlurPool} )
        INPUT_SHAPE = (299,299,3) if 'xception' in filepath else (224,224,3)
        PREPROCESS = True
    for d in list(glob('perturbed_raf_dataset/rafdb.*')):
        outf.write( '%s,%s\n' % (filepath,d) )
        print("Evaluating on %s" % d)
        results = []
        gen = generate_perturbed_rafdb(d, target_shape=INPUT_SHAPE, preprocess=PREPROCESS)
        for batch in tqdm(gen):
            result = model.predict(batch)
            if len(result.shape) > 1:
                result = np.argmax(result, axis=1)
            o = ','.join([str(x) for x in result])
            outf.write(o+'\n')
            results.append(result)
        #outf.write(o)
    outf.write('\n\n')
    outf.close()

import re
ep_re = re.compile('checkpoint.([0-9]+).hdf5')

def _find_latest_checkpoint(d):
    all_checks = glob(os.path.join(d, '*'))
    max_ep = 0
    max_c = None
    for c in all_checks:
        epoch_num = re.search(ep_re, c)
        if epoch_num is not None:
            epoch_num = int(epoch_num.groups(1)[0])
            if epoch_num > max_ep:
                max_ep = epoch_num
                max_c = c
    return max_c

def run_all(dirpath):
    alldirs = glob(os.path.join(dirpath, '*'))
    allchecks = [_find_latest_checkpoint(d) for d in alldirs]
    allchecks = [c for c in allchecks if c is not None]
    print(allchecks)
    for c in allchecks:
        print('\n Testing %s now...\n' % c)
        run_test(c)

if '__main__' == __name__:
    inpath = sys.argv[1]
    if inpath.endswith('.hdf5') or inpath.endswith('.pickle') :
        run_test(inpath)
    else:
        run_all(inpath)
