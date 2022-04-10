#!/usr/bin/python3
from glob import glob
import sys
import os
import numpy as np
from tqdm import tqdm

import tensorflow as tf
import keras
print(keras.__version__)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
keras.backend.set_session(session)
from keras_senet.se_resnet import SEResNet
from vgg_dataset import VggDataset

INPUT_SHAPE = (224,224,3)


dataset = VggDataset('facedetect_vggface2/train.detected1.csv', 'facedetect_vggface2/test.detected1.csv', '/mnt/sdi1/vggface2/vggface2_train/train', '/mnt/sdi1/vggface2/vggface2_test/test', INPUT_SHAPE)#, debug_max_num_samples=1000)

model = SEResNet(classes=dataset.get_num_training_classes(), include_top=False, pooling='avg') # load SEResNet50 with no pretrained weights

model.summary()

dirnm="out_training"
filepath=os.path.join(dirnm, "checkpoint.{epoch:02d}.hdf5")
logdir=os.path.join(dirnm, 'tb_logs')

batch_size = 64

SIMILARITY_THRES = 0.5

def compute_cos_similarity(inp1, inp2):
        DESCR_SIZE = 512
        new_shape = (inp1.shape[0], DESCR_SIZE )
        inp1 = inp1.reshape( new_shape )
        inp2 = inp2.reshape( new_shape )
        # Normalizza
        v1 = inp1 / np.sqrt(np.sum(inp1 ** 2, -1, keepdims=True))
        v2 = inp2 / np.sqrt(np.sum(inp2 ** 2, -1, keepdims=True))
        # Calcola
        similarity_score = np.sum(v1 * v2, -1)
        return similarity_score
       
def compute_ROC(labels, scores):
    import sklearn.metrics as skm
    from scipy import interpolate
    fpr, tpr, thresholds = skm.roc_curve(labels, scores)
    fpr_levels = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    f_interp = interpolate.interp1d(fpr, tpr)
    tpr_at_fpr = [f_interp(x) for x in fpr_levels]
    for (far, tar) in zip(fpr_levels, tpr_at_fpr):
        print('TAR @ FAR = {} : {}'.format(far, tar))

for i in glob(os.path.join(dirnm, "*.hdf5")): 
    YP = []
    YT = []
    model.load_weights(i, by_name=True)
    for batch in tqdm(dataset.val_generator(batch_size)):
        x1, x2, y_true = batch
        out1 = model.predict(x1)
        out2 = model.predict(x2)
        y_pred = compute_cos_similarity(out1, out2)
        YP.append(y_pred)
        YT.append(y_true)
    YP = np.array(YP).flatten()
    YT = np.array(YT).flatten()
    print( "%s -> MAE %.3f" % (i, np.mean( np.abs(YP-YT) ) ) )
    compute_ROC(YT, YP)