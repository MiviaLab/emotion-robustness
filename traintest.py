#!/usr/bin/python3
import argparse


parser = argparse.ArgumentParser(description='FER Training and evaluation.')
parser.add_argument('--lpf', dest='lpf_size', type=int, choices=[0, 1, 3, 5, 7], default=1,
                    help='size of the lpf filter (1 means no filtering)')
parser.add_argument('--cropout', dest='cropout', type=bool, default=False,
                    help='use cropout augmentation')
parser.add_argument('--dataset', dest='dataset', type=str, choices=['fer2013', 'raf-db', 'vggface'], default='fer2013',
                    help='dataset to use for the training')
parser.add_argument('--mode', dest='mode', type=str, choices=['train', 'training', 'test'], default='train',
                    help='train or test')
parser.add_argument('--epoch', dest='test_epoch', type=int, default=None,
                    help='epoch to be used for testing, mandatory if mode=test')
args = parser.parse_args()


# IMPORT ------------------------

import sys
import os
import numpy as np
import tensorflow as tf
import keras
sys.path.append('keras_vggface')
from keras_vggface.vggface import VGGFace

if args.dataset=='fer2013':
    from ferplus_dataset import FerPlusDataset as Dataset, NUM_CLASSES
elif args.dataset=='raf-db':
    from rafdb_dataset import RAFDBDataset as Dataset, NUM_CLASSES
else:
    print('unknown dataset %s' % args.dataset)
    exit(1)
from cropout_test import CropoutAugmentation


# PARAMETERS ----------------------


initial_learning_rate = 0.0001
learning_rate_decay_factor = 0.4
learning_rate_decay_epochs = 40
n_training_epochs = 220
batch_size = 64

def step_decay_schedule(initial_lr, decay_factor, step_size):
    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch/step_size))
    return keras.callbacks.LearningRateScheduler(schedule,verbose=1)




# MODEL ----------------------

INPUT_SHAPE = (224,224,3)
model = VGGFace(model='senet50', input_shape=INPUT_SHAPE, classes=NUM_CLASSES, lpf_size=args.lpf_size)
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=[keras.metrics.categorical_accuracy])

# DIRECTORY -----------------

dirnm="out_training_fer"
if not os.path.isdir(dirnm):
  os.mkdir(dirnm)
dirnm+='/provafixed_nogen_A_%s_lpf%d'%(args.dataset,args.lpf_size)
if args.cropout: dirnm+='_cropout'
if not os.path.isdir(dirnm):
    os.mkdir(dirnm)
#import datetime
#dirnm=dirnm+"/"+str(datetime.datetime.today())
#if len(sys.argv)<=1:
#else:
#dirnm=dirnm+"/"+sys.argv[1]
filepath=os.path.join(dirnm, "checkpoint.{epoch:02d}.hdf5")
logdir=dirnm


if args.mode.startswith('train'):
    print("TRAINING %s" % dirnm)
    #dataset_training = Dataset('train', target_shape=INPUT_SHAPE, augment=True,
    #	custom_augmentation=CropoutAugmentation() if args.cropout else None)
    #dataset_validation = Dataset('val', target_shape=INPUT_SHAPE, augment=False)
    
    import h5py
    dsfile = h5py.File('ds.h5','r')
    Xt = dsfile.get('Xt').value
    Yt = dsfile.get('Yt').value
    Xv = dsfile.get('Xv').value
    Yv = dsfile.get('Yv').value
    print('loaded.')
    lr_sched = step_decay_schedule(initial_lr=initial_learning_rate,
                   decay_factor=learning_rate_decay_factor, step_size=learning_rate_decay_epochs)
    checkpoint = keras.callbacks.ModelCheckpoint(filepath, verbose=1, save_best_only=True)
    tbCallBack = keras.callbacks.TensorBoard(log_dir=logdir, write_graph=True, write_images=True)
    callbacks_list = [lr_sched, checkpoint, tbCallBack]

    model.fit( Xt, Yt, batch_size, n_training_epochs, validation_data=(Xv, Yv), callbacks=callbacks_list)
elif args.mode=='test':
    model.load_weights(  filepath.format(epoch=int(args.test_epoch))  )
    dataset_test = Dataset('test', target_shape=INPUT_SHAPE, augment=False)
    result = model.evaluate_generator(dataset_test.get_generator(batch_size), verbose=1, workers=4)
    print(result)
