'''
This module contains a helper class to handle the dataset you are
training one.
'''

import json
import os
import random
import sys
from os.path import join as ojoin
from pprint import pprint

import pandas as pd
from src.data.coco_video import COCOPriMatrix
from src.models.youthful_edison.preprocessing import (MetaDataGenerator,
                                                      MetaDataIterator)

ROOT_DIR = os.environ['ROOT_DIR']


class Data(object):
    """
    Base class to handle dataset for object detection.

    Heavily reli on the COCO API developed by microsoft
    """

    def __init__(self, data_name):
        self.data_name = data_name
        self.data_folder = os.path.join(ROOT_DIR, 'data', 'interim', data_name)
        self.train = COCOPriMatrix(annotation_file=ojoin(
            self.data_folder, 'annotations', 'instances_train.json'))
        self.validation = COCOPriMatrix(annotation_file=ojoin(
            self.data_folder, 'annotations', 'instances_validation.json'))
        if os.path.isfile(
                ojoin(self.data_folder, 'annotations', 'instances_test.json')):
            self.test = COCOPriMatrix(annotation_file=ojoin(
                self.data_folder, 'annotations', 'instances_test.json'))

    def pprint(self, m):
        print('*' * 50)
        print(m)
        print('*' * 50)

    def describe_dataset(self):
        for split in ['train', 'validation', 'test']:
            if hasattr(self, split):
                self.pprint(split)
                coco = getattr(self, split)
                coco.info()
                print('Nb images: {}'.format(len(coco.getImgIds())))
                pprint(coco.cats)
                print('\n')

    def init_generator(self,
                       dummy_keys=[],
                       dummy_pca=10,
                       real_keys=[],
                       real_scaler='RobustScaler',
                       normalize=True):
        kwargs = {
            key: val
            for key, val in locals().items() if key not in ['self']
        }
        return MetaDataGenerator(**kwargs)

    def init_iterator(self,
                      gen,
                      split='train',
                      batch_size=32,
                      shuffle=True,
                      seed=293,
                      **kwargs):

        return gen.flow_from_directory(
            directory=self.data_folder,
            split=split,
            batch_size=batch_size,
            seed=seed,
            shuffle=shuffle)
