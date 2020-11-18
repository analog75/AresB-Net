import os
import sys
import numpy as np
import lmdb
import matplotlib.pyplot as plt
from protolmdb import definition_pb2
from PIL import Image

#instead of caffe, util in protolmdb is used!!
#caffe_root = #<CAFFE_ROOT>
#sys.path.append(os.path.join(caffe_root, 'python'))
#import caffe
def datum_to_array(datum):
	""" 
	transpose is applied for Channel Height Width -> Height Width Channel  
	"""
	if len(datum.data):
			return np.fromstring(datum.data, dtype=np.uint8).reshape(
					datum.channels, datum.height, datum.width).transpose(1,2,0)
	else:
			return np.array(datum.float_data).astype(float).reshape(
					datum.channels, datum.height, datum.width).transpose(1,2,0)

def read_images_from_lmdb(db_path) :
    """
    Loops over image data in the lmdb, and displays information about each datum
    Assumes that data dimensions are as follows: (channels, height, width)
    """
    #ax = plt.subplot(111)
    #plt.hold(False)
    lmdb_env = lmdb.open(db_path, readonly=True)    
    with lmdb_env.begin() as lmdb_txn :
        lmdb_cursor = lmdb_txn.cursor() 
        #for it in lmdb_cursor.iternext() :
        while lmdb_cursor.next() :
            value = lmdb_cursor.value()
            key = lmdb_cursor.key()
            
            #datum = caffe.proto.caffe_pb2.Datum()
            datum = definition_pb2.Datum()
            datum.ParseFromString(value)
            image = np.zeros((datum.channels, datum.height, datum.width))
            image = datum_to_array(datum)   
            #image = np.transpose(image, (1, 2, 0))    # -> height, width, channels
            image = image[:,:,::-1]                   # BGR -> RGB
              
            print("key: ", key) 
            print("image shape: " + str(image.shape) + ", data type: " + str(image.dtype) + ", random pixel value: " +  str(image[150,150,0]))
                        
            #ax.imshow(np.squeeze(image))
            #plt.draw()
            #plt.waitforbuttonpress()
            
    #plt.show() 
    lmdb_txn.abort()
    lmdb_env.close()

    return
    

def main():

    """
    Set the db_path, and you're good to go.
    """
    
    #db_path = "./data/torch_ilsvrc12_val_lmdb",
    db_path = "/home/ubuntu/database/ImageNet/torch_ilsvrc12_train_lmdb"
    if not os.path.exists(db_path) :
        raise Exception('db not found')

    read_images_from_lmdb(db_path)
        
    return
 
   
if __name__ == "__main__" :
    main()
