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
from coco_video import *
from future.utils import iteritems
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import *

ROOT_DIR = os.environ['ROOT_DIR']
sys.path.append(ROOT_DIR)


class DatasetCreator(object):
    def __init__(self,
                 data_name,
                 dataset_type='nano',
                 val_size=0.1,
                 seed=2231,
                 overfit=False):
        self.data_name = '{}_{}'.format(dataset_type, data_name)
        if overfit:
            self.data_name = '{}_overfit'.format(self.data_name)
        self.overfit = overfit
        self.dataset_type = dataset_type
        self.val_size = val_size
        self.dest_folder = ojoin(ROOT_DIR, 'data', 'interim', self.data_name)
        self.creation_date = time.strftime("%d/%m/%Y")
        self.creation_time = time.strftime("%H:%M:%S")
        self.seed = seed

    def split(self, df):
        """
        split the dataset based on frame_ids.
        if split_idd is True, then take every frame as idpt
        if not, take into acocunt the time.
        """
        N = 2000
        cls = df.columns[1:].tolist()
        dfs = []
        for cl in cls:
            df_sample = df[df[cl] == 1]
            if len(df_sample) > N:
                df_sample = df_sample.sample(N)
            dfs.append(df_sample)
        df = pd.concat(dfs)

        train_idx, validation_idx = train_test_split(
            df.filename.tolist(),
            test_size=self.val_size,
            random_state=self.seed)
        train = df[df.filename.isin(train_idx)]
        train.index = range(len(train))
        validation = df[df.filename.isin(validation_idx)]
        validation.index = range(len(validation))
        return train, validation

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
        data = pd.read_csv(
            os.path.join(ROOT_DIR, 'data', 'raw', 'train_labels.csv'))
        data = data[data.filename.isin(
            os.listdir(
                os.path.join(ROOT_DIR, 'data', 'raw', self.dataset_type)))]

        self.classes = data.columns[1:].tolist()
        self.init_folder(overwrite=overwrite)
        train, validation = self.split(data)

        test = pd.read_csv(
            os.path.join(ROOT_DIR, 'data', 'raw', 'submission_format.csv'))
        splits = zip(['test'], [test])
        for split, data in splits:
            print('\n' + '-' * 50)
            print('Processing  {}'.format(split))
            self.process_split(split, data)

    def get_video(self, reader):
        """
        We downsample to 16 frame - most video are 24fps.
        So we sample 1 frame every 24 during 10s
        """
        T, M, N, C = reader.getShape()

        videodata = np.zeros((T, 224, 224, C), dtype=np.uint8)
        for idx, frame in enumerate(reader.nextFrame()):
            frame = np.array(Image.fromarray(frame).resize((224, 224)))
            videodata[idx, :, :, :] = frame
        video = videodata[list(np.arange(0, 360, 24)) + [len(videodata) - 1]]

        return video

    def get_metadata(self, reader):
        meta_data = {
            k.replace('@', ''): v
            for k, v in iteritems(reader.probeInfo['video']) if type(v) == str
        }
        meta_data.update({
            'disposition_{}'.format(k.replace('@', '')): v
            for k, v in iteritems(reader.probeInfo['video']['disposition'])
        })
        meta_data.update({
            k: v
            for k, v in iteritems(reader.__dict__)
            if type(v) == str or type(v) == int
        })
        return meta_data

    def process_split(self, split, data):
        fdata_path = ojoin(self.dest_folder, 'images',
                           'data_{}.h5'.format(split))
        coco = COCOPriMatrixreator(
            classes=self.classes,
            dataset_type=self.dataset_type,
            dest_folder=self.dest_folder,
            fname='instances_{}'.format(split),
            fdata_path=fdata_path)

        coco.fdata.create_dataset("labels", (len(data), len(self.classes)))
        idx = 0
        err = []

        total = len(data)
        if self.overfit:
            total = 5
        for i, row in tqdm(data.iterrows(), total=total):
            img = dict()
            img['id'] = idx
            img['video_id'] = row['filename']
            img['cat_ids'] = [
                coco.cat2id[c]
                for c in row[self.classes].index[row[self.classes] == 1]
                .tolist()
            ]
            coco.fdata['labels'][idx] = np.array(row[self.classes].tolist())
            try:
                p = ojoin(ROOT_DIR, 'data', 'raw', self.dataset_type,
                          row['filename'])
                reader = skv.FFmpegReader(p)
                video = self.get_video(reader)
                meta_data = self.get_metadata(reader)
            except Exception as e:
                err.append(row['filename'])
                print(p, file=sys.stderr)
                print(e, file=sys.stderr)
                continue

            fpaths = []
            for i in range(len(video)):
                fpath = ojoin(self.dest_folder, 'images',
                              '{}_{}_{}.jpg'.format(split, idx, i))
                Image.fromarray(video[i].astype('uint8')).save(fpath)
                fpaths.append(fpath)
            img['files'] = fpaths
            img['meta_data'] = meta_data
            img['video_path'] = p
            img['nframes'] = video.shape[0]
            img['height'] = video.shape[1]
            img['width'] = video.shape[2]

            coco.update_images(**img)
            idx += 1

            if self.overfit:
                if i > total:
                    break

            # Dump annotations
        coco.dump_annotations()


if __name__ == '__main__':
    dataset = DatasetCreator('d1_2000', dataset_type='prima', overfit=False)
    dataset.create_dataset()
