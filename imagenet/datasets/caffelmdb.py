import os
import sys
import string
import lmdb
import numpy as np
from .protolmdb import definition_pb2
from PIL import Image
#import cv2

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torchvision import transforms, datasets
# This segfaults when imported before torch: https://github.com/apache/arrow/issues/2637

class ImageFolderLMDB(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
    """
    def __init__(self, db_path=None, transform=None):
        self.transform = transform
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=os.path.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        self.lmdb_txn = self.env.begin()
        self.length = self.env.stat()['entries']

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        datum = definition_pb2.Datum()
        lmdb_cursor = self.lmdb_txn.cursor()
        key_index ='{:08}'.format(index)
        value = lmdb_cursor.get(key_index.encode('ascii'))
        datum.ParseFromString(value)
        data = self.datum_to_array(datum)
        data = Image.fromarray(data).convert('RGB')         
        if self.transform is not None:
            imgx = self.transform(data)
        target = datum.label

        return imgx, target

    def __len__(self):
        return self.length

    def datum_to_array(self, datum):
      """ 
      transpose is applied for Channel Height Width -> Height Width Channel  
      """
      if len(datum.data):
          return np.fromstring(datum.data, dtype=np.uint8).reshape(
              datum.channels, datum.height, datum.width).transpose(1,2,0)
      else:
          return np.array(datum.float_data).astype(float).reshape(
              datum.channels, datum.height, datum.width).transpose(1,2,0)


