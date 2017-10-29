import datetime
import itertools
import json
import os
import time
from collections import defaultdict
from os.path import join as ojoin

import h5py
import numpy as np
import pandas as pd


class COCOPriMatrix():
    def __init__(self, annotation_file=None, verbose=False):
        """
        Constructor of COCOPCD helper class for reading and visualizing annotated point cloud.
        :param annotation_file (str): location of annotation file
        :return:
        """
        # load dataset
        self.dataset, self.cats, self.imgs = dict(), dict(), dict()
        self.catToImgs = defaultdict(list)
        self.verbose = verbose
        if not annotation_file == None:
            if self.verbose:
                print('loading annotations into memory...')
            tic = time.time()
            dataset = json.load(open(annotation_file, 'r'))
            assert type(
                dataset
            ) == dict, 'annotation file format {} not supported'.format(
                type(dataset))
            if self.verbose:
                print('Done (t={:0.2f}s)'.format(time.time() - tic))
            self.dataset = dataset
            self.createIndex()
            self.info()
            self.fdata = h5py.File(self.fdata_path, "r")

    def createIndex(self):
        # create index
        if self.verbose:
            print('creating index...')
        cats, imgs = {}, {}
        catToImgs = defaultdict(list)

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img
                for cat_id in img['cat_ids']:
                    catToImgs[cat_id].append(img['id'])

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if self.verbose:
            print('index created!')

        # create class members
        self.imgs = imgs
        self.cats = cats
        self.catToImgs = catToImgs

    def info(self):
        """
        Print information about the annotation file.
        :return:
        """
        for key, value in self.dataset['info'].items():
            print('{}: {}'.format(key, value))
            setattr(self, key, value)

    def getVidData(self, video_ids):
        catNms = catNms if type(catNms) == list else [catNms]
        pass

    def getCatIds(self, catNms=[], catIds=[]):
        """
        filtering parameters. default skips that filter.
        :param catNms (str array)  : get cats for given cat names
        :param catIds (int array)  : get cats for given cat ids
        :return: ids (int array)   : integer array of cat ids
        """
        catNms = catNms if type(catNms) == list else [catNms]
        catIds = catIds if type(catIds) == list else [catIds]

        if len(catNms) == len(supNms) == len(catIds) == 0:
            cats = self.dataset['categories']
        else:
            cats = self.dataset['categories']
            cats = cats if len(catNms) == 0 else [
                cat for cat in cats if cat['name'] in catNms
            ]
            cats = cats if len(catIds) == 0 else [
                cat for cat in cats if cat['id'] in catIds
            ]
        ids = [cat['id'] for cat in cats]
        return ids

    def getImgIds(self, imgIds=[], catIds=[]):
        '''
        Get pcd ids that satisfy given filter conditions.
        :param catIds (int array) : get pcds with all given cats
        :param imgIds (int array) : get pcds with all given img
        :return: ids (int array)  : integer array of pcd ids
        '''
        catIds = catIds if type(catIds) == list else [catIds]
        imgIds = imgIds if type(imgIds) == list else [imgIds]

        if len(catIds) == len(imgIds) == 0:
            ids = self.imgs.keys()
        else:
            ids = set(imgIds)
            for i, catId in enumerate(catIds):
                if i == 0 and len(ids) == 0:
                    ids = set(self.catToImgs[catId])
                else:
                    ids &= set(self.catToImgs[catId])

        return list(ids)

    def loadCats(self, ids=[]):
        """
        Load cats with the specified ids.
        :param ids (int array)       : integer ids specifying cats
        :return: cats (object array) : loaded cat objects
        """
        if type(ids) == list:
            return [self.cats[id] for id in ids]
        elif type(ids) == int:
            return [self.cats[ids]]

    def loadImgs(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying img
        :return: imgs (object array) : loaded img objects
        """
        if type(ids) == list:
            return [self.imgs[id] for id in ids]
        elif type(ids) == int:
            return [self.imgs[ids]]

    def loadRes(self, resFile):
        """
        Load result file and return a result api object.
        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object
        """
        res = COCOPriMatrix()
        res.dataset['imgs'] = [img for img in self.dataset['imgs']]
        if self.verbose:
            print('Loading and preparing results...')
        tic = time.time()
        if type(resFile) == str or type(resFile) == unicode:
            preds = json.load(open(resFile))
        else:
            preds = resFile

        if 'imgs' in preds.keys():
            print('Found annotations in the dump')
            imgs = preds['imgs']
            assert type(imgs) == list, 'results in not an array of objects'
            imgIds = [img['id'] for img in imgs]
            assert set(imgIds) == (set(imgIds) & set(self.getImgIds())), \
                'Results do not correspond to current cocopcd set'
            for id, img in enumerate(imgs):
                img['id'] = id + 1
            res.dataset['imgs'] = img

        if self.verbose:
            print('DONE (t={:0.2f}s)'.format(time.time() - tic))

        res.createIndex()
        return res


class COCOPriMatrixreator(object):
    '''
    Object to help the creation of dataset.
    '''

    def __init__(self, classes, dest_folder, fname, dataset_type, fdata_path):
        self.dest_folder = dest_folder
        if not os.path.isdir(ojoin(self.dest_folder, 'annotations')):
            os.mkdir(ojoin(self.dest_folder, 'annotations'))
        self.annot = dict()
        self.annot['info'] = dict(
            fdata_path=fdata_path,
            contributor='clement',
            dataset_type=dataset_type,
            classes=classes,
            data_created=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            verson=1.0,
            year=2017)
        self.classes = classes
        self.annot['images'] = []
        self.annot['categories'] = []
        self.init_categories()
        self.fname = fname
        self.fdata_path = fdata_path
        if os.path.isfile(fdata_path):
            os.remove(fdata_path)
        self.fdata = h5py.File(fdata_path, "w")

    def init_categories(self):
        names = self.classes
        for idx, name in zip(range(len(names)), names):
            self.annot['categories'].append(dict(id=idx, name=name))
        self.cat2id = dict(zip(names, range(len(names))))

    def update_images(self, **kwargs):
        img = kwargs
        self.annot['images'].append(img)

    def dump_annotations(self):
        print(self.dest_folder)
        fname = ojoin(self.dest_folder, 'annotations', '{}.json'.format(
            self.fname))
        json.dump(self.annot, open(fname, 'w+'))
        self.fdata.close()
