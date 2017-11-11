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
from src.models.boring_leavitt.preprocessing import (VideoDataGenerator,
                                                     VideoDataIterator)

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
                       featurewise_center=False,
                       samplewise_center=False,
                       featurewise_std_normalization=False,
                       samplewise_std_normalization=False,
                       zca_whitening=False,
                       rotation_range=0.,
                       width_shift_range=0.,
                       height_shift_range=0.,
                       shear_range=0.,
                       zoom_range=0.,
                       channel_shift_range=0.,
                       fill_mode='nearest',
                       cval=0.,
                       horizontal_flip=False,
                       vertical_flip=False,
                       rescale=None,
                       data_format=None):
        """
        Generate minibatches with real-time data augmentation. Apply the same transformation
        to the image and the bunding box.

        # Arguments
        featurewise_center: set input mean to 0 over the dataset.
        samplewise_center: set each sample mean to 0.
        featurewise_std_normalization: divide inputs by std of the dataset.
        samplewise_std_normalization: divide each input by its std.
        zca_whitening: apply ZCA whitening.
        rotation_range: degrees (0 to 180).
        width_shift_range: fraction of total width.
        height_shift_range: fraction of total height.
        shear_range: shear intensity (shear angle in radians).
        zoom_range: amount of zoom. if scalar z, zoom will be randomly picked
            in the range [1-z, 1+z]. A sequence of two can be passed instead
            to select this range.
        channel_shift_range: shift range for each channels.
        fill_mode: points outside the boundaries are filled according to the
            given mode ('constant', 'nearest', 'reflect' or 'wrap'). Default
            is 'nearest'.
        cval: value used for points outside the boundaries when fill_mode is
            'constant'. Default is 0.
        horizontal_flip: whether to randomly flip images horizontally.
        vertical_flip: whether to randomly flip images vertically.
        rescale: rescaling factor. If None or 0, no rescaling is applied,
            otherwise we multiply the data by the value provided
            (before applying any other transformation).
        preprocessing_function: function that will be implied on each input.
            The function will run before any other modification on it.
            The function should take one argument: one image (Numpy tensor with rank 3),
            and should output a Numpy tensor with the same shape.
        dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
            (the depth) is at index 1, in 'tf' mode it is at index 3.
            It defaults to the `image_dim_ordering` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "th".
        """

        kwargs = {
            key: val
            for key, val in locals().items() if key not in ['self']
        }
        return VideoDataGenerator(**kwargs)

    def init_iterator(self,
                      gen,
                      split='train',
                      batch_size=32,
                      target_size=(16, 64, 64, 3),
                      shuffle=True,
                      seed=293,
                      **kwargs):

        # Clean up the temporary folder
        return gen.flow_from_directory(
            directory=self.data_folder,
            split=split,
            target_size=target_size,
            batch_size=batch_size,
            seed=seed,
            shuffle=shuffle)
