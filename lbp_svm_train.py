from lbp_descriptor import lbp_hist
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import pickle, os, sys
from rafdb_dataset import RAFDBDataset as Dataset, NUM_CLASSES
from tqdm import tqdm
from glob import glob

def compute_features(dataset, part):
    cache_file_name = 'lbp_data.%s.cache'%part
    X = []
    y = []
    try:
        with open(cache_file_name, 'rb') as f:
            X,y = pickle.load(f)
            print("Data loaded from cache")
    except FileNotFoundError:
        for item in tqdm(dataset.data):
            img = item['img']
            lbl = np.argmax(item['label'])
            h, lbpimg=lbp_hist(img)
            X.append(h)
            y.append(lbl)
        X = np.array(X)
        y = np.array(y)
        pickle.dump( (X,y), open(cache_file_name, 'wb') )
    print(X.shape, X.dtype, y.shape, y.dtype)
    return X,y

def showds(dataset):
    for item in tqdm(dataset.data):
        img = item['img']
        lbl = np.argmax(item['label'])
        h, lbpimg=lbp_hist(img)
        import cv2
        cv2.imshow('orig', img)
        cv2.imshow('lbp', lbpimg*4)
        if cv2.waitKey(0) & 0xff == 27:
            sys.exit(0)

MODEL_FILE_NAME = 'out_training_fer/svc_c%f_g%f_bal.pickle'
INPUT_SHAPE=(150,110)
dataset_training = Dataset('train', target_shape=INPUT_SHAPE, augment=False, preprocessing='no', custom_augmentation=None)
dataset_validation = Dataset('val', target_shape=INPUT_SHAPE, augment=False, preprocessing='no', custom_augmentation=None)

bestc, bestg = 4, 3e-6

def load(c,g):
    print('Loading...')
    model = pickle.load(open(MODEL_FILE_NAME%(c,g), 'rb'))
    print('Loaded.')
    model.verbose=1
    print(model)
    return model

def evaluate(model, Xv, yv):
    print('Evaluating...')
    y_pred = model.predict(Xv)
    result = accuracy_score(yv, y_pred)
    cm = confusion_matrix(yv, y_pred)
    print(cm)
    print("Accuracy: %.3f"%result)
    return result

def train(Xt,yt, c, g):
    model = SVC(C=c, kernel='rbf', gamma=g, verbose=True, class_weight='balanced')
    print(model)
    print('Training...')
    model.fit(Xt,yt)
    print('Done.')
    pickle.dump(model, open(MODEL_FILE_NAME%(c,g), 'wb'))
    print('Saved.')
    return model

if __name__ == "__main__":
        
    if sys.argv[1] == 'search':
        Xt,yt = compute_features(dataset_training, 'train')
        Xv,yv = compute_features(dataset_validation, 'val')
        cs = [2**0, 2**2]
        gs = list(1/Xt.shape[1]/np.array([2**9, 2**11]))
        allres = np.zeros((len(cs),len(gs)))

        for ic, c in enumerate(cs):
            for ig, g in enumerate(gs):
                model = train(Xt,yt, c, g)
                result = evaluate(model, Xv, yv)
                allres[ic,ig] = result
        print(cs)
        print(gs)
        print(allres)

    elif sys.argv[1] == 'eval':
        Xv,yv = compute_features(dataset_validation, 'val')
        model = load(bestc, bestg)
        evaluate(model, Xv, yv)

    elif sys.argv[1] == 'eval_corruptions':
        outf = open('results_lbp.txt', "a+")
        outf.write( 'Results for: LBP_rbf\n')
        model = load(bestc, bestg)
        for d in ['RAF-DB/basic/Image/aligned']+list(glob('corrupted_raf_dataset/rafdb.*')):
            print('Evaluating %s' % d)
            dataset_corrupted = Dataset('val', imagesdir=d, target_shape=INPUT_SHAPE, augment=False, preprocessing='no', custom_augmentation=None)
            Xv,yv = compute_features(dataset_corrupted, 'val-%s'%d.replace('/','-'))
            result = evaluate(model, Xv, yv)
            o =  "%s %f\n"%(d, result)
            outf.write(o)
            outf.write('\n\n')
        outf.close()
        
    elif sys.argv[1] == 'show':
        showds(dataset_training)


class LBPPredictor():
    def __init__(self, fname):
        print("Loading %s..." % fname)
        self.model = pickle.load(open(fname, 'rb'))
        print("Loaded: %s" % str(self.model))
        
    def predict(self, images):
        Xv = []
        for img in images:
            h, _=lbp_hist(img)
            Xv.append(h)
        Xv = np.array(Xv)
        return self.model.predict(Xv)
