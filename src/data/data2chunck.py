from __future__ import print_function

import json
import os
import shutil
import sys
import time
import xml
from multiprocessing import Pool
from os.path import join as ojoin

import pandas as pd
import skvideo.io as skv
from boltons.iterutils import chunked_iter
from coco_video import *
from future.utils import iteritems
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import *

ROOT_DIR = os.environ['ROOT_DIR']
sys.path.append(ROOT_DIR)


class Data2ChunkData(object):
    def __init__(self, data_name, data_name_from, dataset_type='prima'):
        self.data_name = '{}_{}'.format(dataset_type, data_name)
        self.dataset_type = dataset_type
        self.dest_folder = ojoin(ROOT_DIR, 'data', 'interim', self.data_name)
        self.ori_folder = ojoin(ROOT_DIR, 'data', 'interim2', data_name_from)

    def init_folder(self, overwrite=False):
        '''
        Init the binary directory
        '''
        if os.path.isdir(self.dest_folder):
            if overwrite:
                shutil.rmtree(self.dest_folder)
            else:
                print('Dataset {} already exist but overwrite set to False'.
                      format(self.data_name))
        self._mkdir_recursive(self.dest_folder)
        self._mkdir_recursive(ojoin(self.dest_folder, 'annotations'))
        self._mkdir_recursive(ojoin(self.dest_folder, 'images'))

    def _mkdir_recursive(self, path):
        '''
        Small function to recursively create a folder
        '''
        sub_path = os.path.dirname(path)
        if not os.path.exists(sub_path):
            self._mkdir_recursive(sub_path)
        if not os.path.exists(path):
            os.mkdir(path)

    def create_dataset(self, overwrite=False):
        '''
        Literrally create the dataset.
        If df is provided, use it directly,
        if not, use the db.
        '''
        self.init_folder(overwrite=overwrite)
        splits = ['train', 'validation']
        for split in splits:
            print('\n' + '-' * 50)
            print('Processing  {}'.format(split))
            self.process_split(split)

    def process_split(self, split):
        data = COCOPriMatrix(
            ojoin(self.ori_folder, 'annotations',
                  'instances_{}.json'.format(split)))
        data.fdata_path = data.fdata_path.replace('interim', 'interim2')
        data.fdata = h5py.File(data.fdata_path, "r")

        fdata_path = ojoin(self.dest_folder, 'images',
                           'labels_{}.h5'.format(split))
        coco = COCOPriMatrixreator(
            classes=data.classes,
            dataset_type=self.dataset_type,
            dest_folder=self.dest_folder,
            fname='instances_{}'.format(split),
            fdata_path=fdata_path)

        coco.fdata.create_dataset("labels",
                                  (len(data.fdata['data']), len(data.classes)))

        chunk_size = 200
        total = len(data.getImgIds()) / chunk_size
        for batch_imgid in tqdm(
                chunked_iter(data.getImgIds(), size=chunk_size), total=total):
            chunk_data = data.fdata['data'][batch_imgid[0]:batch_imgid[-1] + 1]
            chunk_label = data.fdata['labels'][batch_imgid[0]:
                                               batch_imgid[-1] + 1]
            for i, imgid in enumerate(batch_imgid):
                idx_chunk = imgid - batch_imgid[0]
                img = data.imgs[imgid]
                fdata_path = ojoin(self.dest_folder, 'images',
                                   '{}_{}.h5'.format(split, imgid))
                img['fdata_path'] = fdata_path
                if os.path.isfile(fdata_path):
                    os.remove(fdata_path)
                fdata = h5py.File(fdata_path, "w")
                fdata['data'] = chunk_data[imgid - batch_imgid[0]]
                fdata['labels'] = chunk_label[imgid - batch_imgid[0]]
                fdata.close()

                coco.fdata['labels'][imgid] = chunk_label[imgid
                                                          - batch_imgid[0]]
                coco.update_images(**img)

            # Dump annotations
        coco.dump_annotations()


if __name__ == '__main__':
    dataset = Data2ChunkData(
        'd1_500_rectified',
        data_name_from='prima_d1_500',
        dataset_type='prima')
    dataset.create_dataset(overwrite=False)
