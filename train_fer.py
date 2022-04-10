#!/usr/bin/python3
import argparse

parser = argparse.ArgumentParser(description='FER Training and evaluation.')
parser.add_argument('--lpf', dest='lpf_size', type=int, choices=[0, 1, 2, 3, 5, 7], default=1,
                    help='size of the lpf filter (1 means no filtering)')
parser.add_argument('--cutout', action='store_true', help='use cutout augmentation')
parser.add_argument('--center_loss', action='store_true', help='use center loss')
parser.add_argument('--weight_decay', type=float)
parser.add_argument('--lr', default='0.002', help='Initial learning rate or init:factor:epochs', type=str)
parser.add_argument('--momentum', action='store_true')
parser.add_argument('--dataset', dest='dataset', type=str, choices=['fer2013', 'raf-db', 'vggface2'], default='raf-db',
                    help='dataset to use for the training')
parser.add_argument('--mode', dest='mode', type=str, choices=['train', 'training', 'test'], default='train',
                    help='train or test')
parser.add_argument('--epoch', dest='test_epoch', type=int, default=None,
                    help='epoch to be used for testing, mandatory if mode=test')
parser.add_argument('--dir', dest='dir', type=str, default=None,
                    help='directory for reading/writing training data and logs')
parser.add_argument('--batch', dest='batch_size', type=int, default=64, help='batch size.')
parser.add_argument('--ngpus', dest='ngpus', type=int, default=1, help='Number of gpus to use.')
parser.add_argument('--net', type=str, default='senet50', choices=['senet50-100', 'senet50', 'vgg16-100', 'densenet121bc', 'densenet121bc-100', 'xception-100', 'mobilenet96', 'mobilenet224'], help='Network architecture')
parser.add_argument('--resume', type=bool, default=False, help='resume training')
parser.add_argument('--pretraining', type=str, default=None, help='Pretraining weights, do not set for None, can be vggface or imagenet or a file')
parser.add_argument('--preprocessing', type=str, default='full_normalization', choices=['z_normalization', 'full_normalization', 'vggface2'])
parser.add_argument('--augmentation', type=str, default='default', choices=['default', 'vggface2', 'autoaugment-rafdb', 'no'])
args = parser.parse_args()


# IMPORT ------------------------
import warnings; warnings.filterwarnings('ignore',category=FutureWarning)
import sys
import os
import numpy as np
from glob import glob
import re
import tensorflow as tf
import keras

if args.dataset=='fer2013':
    from ferplus_dataset import FerPlusDataset as Dataset, NUM_CLASSES, CLASS_LABELS
elif args.dataset=='raf-db':
    from rafdb_dataset import RAFDBDataset as Dataset, NUM_CLASSES
elif args.dataset=='vggface2':
    from vgg2_dataset import Vgg2Dataset as Dataset, NUM_CLASSES
else:
    print('unknown dataset %s' % args.dataset)
    exit(1)
from cropout_test import CropoutAugmentation


# PARAMETERS ----------------------
lr = args.lr.split(':')
initial_learning_rate = float(lr[0]) #0.002
learning_rate_decay_factor = float(lr[1]) if len(lr)>1 else 0.5
learning_rate_decay_epochs = int(lr[2]) if len(lr)>2 else 40
n_training_epochs = 220
batch_size = args.batch_size

def step_decay_schedule(initial_lr, decay_factor, step_size):
    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch/step_size))
    return keras.callbacks.LearningRateScheduler(schedule,verbose=1)




# MODEL ----------------------
INPUT_SHAPE=None
def get_model():
    global INPUT_SHAPE
    if args.net.startswith('senet') or args.net.startswith('resnet') or args.net.startswith('vgg'):
        INPUT_SHAPE = (100,100,3) if args.net.endswith('-100') else (224,224,3)
        if args.pretraining is not None and args.pretraining.startswith('imagenet') and args.net.startswith('resnet'):
            assert args.lpf_size==1 or args.lpf_size==0
            sys.path.append('keras-squeeze-excite-network')
            from keras.applications.resnet import ResNet50
            m1 = ResNet50(weights=args.pretraining, input_shape=INPUT_SHAPE, include_top=False, pooling='avg', weight_decay=0)#, lpf_size=args.lpf_size)i
            features = m1.output
            x = keras.layers.Dense(NUM_CLASSES, use_bias=False, activation='softmax')(features)
            model = keras.models.Model(m1.input, x)
            return model, features
        if args.pretraining is not None and args.pretraining.startswith('imagenet') and args.net.startswith('senet'):
            assert args.lpf_size==1 or args.lpf_size==0
            sys.path.append('keras-squeeze-excite-network')
            from keras_squeeze_excite_network.se_resnet import SEResNet
            m1 = SEResNet(weights=args.pretraining, input_shape=INPUT_SHAPE, include_top=False, pooling='avg', weight_decay=0)#, lpf_size=args.lpf_size)i
            features = m1.output
            x = keras.layers.Dense(NUM_CLASSES, use_bias=False, activation='softmax')(features)
            model = keras.models.Model(m1.input, x)
            return model, features
        else: # VGGFACE PRETRAINING
            sys.path.insert(0, 'keras_vggface')
            from keras_vggface.vggface import VGGFace
            return VGGFace(model=args.net, weights=args.pretraining, input_shape=INPUT_SHAPE, classes=NUM_CLASSES, lpf_size=args.lpf_size)
    elif args.net.startswith('mobilenet'):
        s = int(args.net[9:])
        INPUT_SHAPE = (s,s,3)
        m1 = keras.applications.mobilenet_v2.MobileNetV2(INPUT_SHAPE,0.75 if s<=96 else 1.0, include_top=False, weights=args.pretraining)
        x = m1.output
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dense(NUM_CLASSES, activation='softmax', use_bias=True, name='Logits')(x)
        return keras.Model(m1.input, x), m1.output
    elif args.net.startswith('densenet121bc'):
        INPUT_SHAPE = (100,100,3) if args.net.endswith('-100') else (224,224,3)
        sys.path.insert(0, 'keras_vggface')
        from keras_vggface.densenet import DenseNet121
        m1 = DenseNet121(include_top=False, input_shape=INPUT_SHAPE, weights=args.pretraining, pooling='avg', lpf_size=args.lpf_size)
        x = m1.output
        x = keras.layers.Dense(NUM_CLASSES, activation='softmax', use_bias=True, name='Logits')(x)
        return keras.Model(m1.input, x), m1.output
    elif args.net.startswith('xception'):
        INPUT_SHAPE = (100,100,3) if args.net.endswith('-100') else (299,299,3)
        sys.path.insert(0, 'keras_vggface')
        from keras_vggface.xception import Xception
        model = Xception(input_shape=INPUT_SHAPE, weights=args.pretraining, include_top=False, pooling='avg', lpf_size=args.lpf_size)
        features = model.output
        x = keras.layers.Dense(NUM_CLASSES, use_bias=False, activation='softmax')(features)
        model = keras.models.Model(model.input, x)
        return model, features
if args.ngpus <=1:
    model, feature_layer = get_model()
else:
    print("Using %d gpus" % args.ngpus)
    with tf.device('/cpu:0'):
        model, feature_layer = get_model()
    model = keras.utils.multi_gpu_model(model, args.ngpus)
model.summary()


if args.weight_decay:
    weight_decay=args.weight_decay #0.0005
    for layer in model.layers:
        print(layer.name, type(layer), end='')
        #if isinstance(layer, keras.layers.Conv2D) or isinstance(layer, keras.layers.Dense):
        #    layer.add_loss(keras.regularizers.l2(weight_decay)(layer.kernel))
        if hasattr(layer, 'kernel_regularizer'):
            layer.kernel_regularizer = keras.regularizers.l2(weight_decay)
            print('K', end='')
        if hasattr(layer, 'bias_regularizer') and layer.use_bias:
            #layer.add_loss(keras.regularizers.l2(weight_decay)(layer.bias))
            layer.bias_regularizer = keras.regularizers.l2(weight_decay)
            print('B', end='')
        print('')

if args.momentum:
    optimizer = keras.optimizers.sgd(momentum=0.9)
else:
    optimizer = 'sgd'

from center_loss import center_loss
if args.center_loss:
    loss = center_loss(feature_layer, keras.losses.categorical_crossentropy, 0.9, NUM_CLASSES, 0.01, features_dim=2048)
else:
    loss = keras.losses.categorical_crossentropy
model.compile(loss=loss,
              optimizer=optimizer,
              metrics=[keras.metrics.categorical_accuracy])


# DIRECTORY -----------------

from datetime import datetime
datetime = datetime.today().strftime('%Y%m%d_%H%M%S')
dirnm="out_training_fer"
if not os.path.isdir(dirnm):
  os.mkdir(dirnm)
argstring=''.join(sys.argv[1:]).replace('--','_').replace('=','').replace(':','_')
dirnm+='/%s_%s'%(argstring, datetime)
if args.cutout: dirnm+='_cutout'

if args.dir: dirnm=args.dir

if not os.path.isdir(dirnm):
    os.mkdir(dirnm)
#dirnm=dirnm+"/"+str(datetime.datetime.today())
#if len(sys.argv)<=1:
#else:
#dirnm=dirnm+"/"+sys.argv[1]
filepath=os.path.join(dirnm, "checkpoint.{epoch:02d}.hdf5")
logdir=dirnm



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
    return max_ep, max_c


def get_confusion(model, gen):
    from sklearn.metrics import confusion_matrix
    yp, yt = [], []
    for batch in gen:
        result = model.predict(batch[0])
        yt += np.argmax(batch[1], axis=1).tolist()
        yp += np.argmax(result, axis=1).tolist()
    yp = np.array(yp)
    yt = np.array(yt)
    accuracy = np.sum(np.equal(yp,yt))/len(yp)
    cm = confusion_matrix(yt, yp, labels=range(NUM_CLASSES))
    return accuracy, cm


from dataset_tools import DefaultAugmentation, VGGFace2Augmentation
if args.cutout:
    custom_augmentation=CropoutAugmentation()
elif args.augmentation =='autoaugment-rafdb':
    from autoaug_test import MyAutoAugmentation
    from autoaugment.rafdb_policies import rafdb_policies
    custom_augmentation=MyAutoAugmentation(rafdb_policies)
elif args.augmentation=='default':
    custom_augmentation=DefaultAugmentation()
elif args.augmentation=='vggface2':
    custom_augmentation=VGGFace2Augmentation()
else:
    custom_augmentation=None

if args.mode.startswith('train'):
    print("TRAINING %s" % dirnm)
    dataset_training = Dataset('train', target_shape=INPUT_SHAPE, augment=False, 
            preprocessing=args.preprocessing, custom_augmentation=custom_augmentation)
    dataset_validation = Dataset('val', target_shape=INPUT_SHAPE, augment=False, preprocessing=args.preprocessing)
    
    lr_sched = step_decay_schedule(initial_lr=initial_learning_rate,
                   decay_factor=learning_rate_decay_factor, step_size=learning_rate_decay_epochs)
    checkpoint = keras.callbacks.ModelCheckpoint(filepath, verbose=1, save_best_only=True, monitor='val_categorical_accuracy')
    tbCallBack = keras.callbacks.TensorBoard(log_dir=logdir, write_graph=True, write_images=True)
    callbacks_list = [lr_sched, checkpoint, tbCallBack]
    if args.resume:
        pattern=filepath.replace('{epoch:02d}','*')
        epochs = glob(pattern)
        print(pattern)
        print(epochs)
        epochs = [ int(x[-8:-5].replace('.','')) for x in epochs ]
        initial_epoch=max(epochs)
        print('Resuming from epoch %d...'%initial_epoch)
        model.load_weights(filepath.format(epoch=initial_epoch))
    else:
        initial_epoch=0
    model.fit_generator(dataset_training.get_generator(batch_size),
        validation_data=dataset_validation.get_generator(batch_size),
        verbose=1, callbacks=callbacks_list, epochs=n_training_epochs, workers=8, initial_epoch=initial_epoch)
elif args.mode=='test':
    if args.test_epoch is None:
        args.test_epoch, _ = _find_latest_checkpoint(dirnm)
        print("Using epoch %d" % args.test_epoch)
    model.load_weights(  filepath.format(epoch=int(args.test_epoch))  )
    def evalds(part):
        dataset_test = Dataset(part, target_shape=INPUT_SHAPE, augment=False, preprocessing=args.preprocessing) 
        #result = model.evaluate_generator(dataset_test.get_generator(batch_size), verbose=1, workers=4)
        acc, cm = get_confusion(model, dataset_test.get_generator(batch_size))
        print('%s accuracy: %f' % (part,acc))
        print("%s: predicted" % " ".join(["%7s"%x[:7] for x in CLASS_LABELS]))
        for i, line in enumerate(cm):
            lbl = CLASS_LABELS[i]
            for n in line:
                print("%7d"%n, end=' ')
            acc = line[i]/np.sum(line)
            print("%7s, acc:%.1f" %(lbl[:7], acc*100))
    evalds('test')
    evalds('val')
    evalds('train')

