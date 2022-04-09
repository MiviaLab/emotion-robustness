#!/usr/bin/python3
import keras
import sys, os
from glob import glob
from ferplus_dataset import FerPlusDataset
sys.path.append('keras_vggface/keras_vggface')
from antialiasing import BlurPool

def run_test():
    batch_size = 32
    P = 'PublicTest'
    print('Partition: %s'%P)    
    #DATE="2019-09-29 11:36:42.014910"
    #EPOCH=164
    #dirnm="out_training_fer/"+DATE
    #filepath=os.path.join(dirnm, "checkpoint.{epoch:02d}.hdf5")
    filepath = sys.argv[1]
    outf = open('results.txt', "a+")
    outf.write( 'Results for: %s\n' % filepath )
    model = keras.models.load_model(  filepath , custom_objects={'BlurPool':BlurPool} )
    INPUT_SHAPE = (224,224,3)
    
    for d in ['FERPlus/fer2013.csv']+list(glob('corrupted_dataset/*.csv')):
        dataset_test = FerPlusDataset(partition=P, target_shape=INPUT_SHAPE, augment=False, csvdata=d)
        result = model.evaluate_generator(dataset_test.get_generator(batch_size), verbose=1, workers=4)
        o =  "%s %f\n"%(d, result[1])
        print("\n\n RES "+o)
        outf.write(o)
    outf.write('\n\n')
    outf.close()


if '__main__' == __name__:
    run_test()
