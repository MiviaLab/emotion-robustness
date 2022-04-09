#!/usr/bin/python3
import keras
import sys, os
from glob import glob
from rafdb_dataset import RAFDBDataset as Dataset
sys.path.append('keras_vggface/keras_vggface')
from antialiasing import BlurPool
import time

def run_test(filepath, batch_size=64):
    P = 'test'
    print('Partition: %s'%P)    
    outf = open(sys.argv[2], "a+")
    outf.write( 'Results for: %s\n' % filepath )
    model = keras.models.load_model(  filepath , custom_objects={'BlurPool':BlurPool} )
    model.compile(optimizer=keras.optimizers.SGD(), loss=keras.losses.CategoricalCrossentropy(), metrics=['categorical_accuracy'])

    INPUT_SHAPE = (299,299,3) if 'xception' in filepath else (224,224,3)
    
    for d in ['RAF-DB/basic/Image/aligned']+list(glob('corrupted_raf_dataset/rafdb.*')):
        dataset_test = Dataset(partition=P, target_shape=INPUT_SHAPE, preprocessing='vggface2', custom_augmentation=None, augment=False, imagesdir=d)
        total_time = -time.time()
        result = model.evaluate(dataset_test.get_generator(batch_size), verbose=1, workers=4)
        total_time += time.time()
        o =  "%s %f\n"%(d, result[1])
        print("\n\n RES "+o)
        print(f"#Time:{total_time}s")
        outf.write(o)
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
    if inpath.endswith('.hdf5') :
        run_test(inpath)
    else:
        run_all(inpath)
