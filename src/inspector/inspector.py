import json
import os
import pprint
import time
from os.path import join as ojoin

import h5py
import numpy as np
import pandas as pd
from src.data.coco_video_eval import *
from tabulate import tabulate
from tqdm import *

ROOT_DIR = os.environ['ROOT_DIR']


class PriMatrixInspector(object):
    '''
    A general inspector to inspect the result for the competition
    '''

    def __init__(self, data, model):
        self.data = data
        self.model = model
        self.project = 'prima'
        self.modeltype = 'models'

    def gen_prediction(self, split='train', batch_size=16):
        """
        Gen prediction and dump then to disk
        """
        reader = getattr(self.data, split)

        predictions = self.model.predict(
            reader=reader, batch_size=batch_size, verbose=True)

        # Writing the detections
        print('Writing the detections in the PriMatrix format')
        df = pd.DataFrame(
            predictions, columns=['id'] + self.data.train.classes)
        fdata_path = ojoin(self.model.model_folder, '{}_{}_preds.h5'.format(
            self.model.expname, split))
        if os.path.isfile(fdata_path):
            os.remove(fdata_path)
        fdata = h5py.File(fdata_path, "a")
        fdata.create_dataset("preds",
                             getattr(self.data, split).fdata['labels'].shape)
        imgs = []
        cat2id = {v['name']: v['id'] for v in reader.cats.values()}
        for i, row in df.iterrows():
            img = dict()
            img['id'] = int(row['id'])
            img['fdata_path'] = fdata_path
            img['cat_ids'] = [
                cat2id[c]
                for c in row[self.data.train.classes].index[
                    row[self.data.train.classes] > .5].tolist()
            ]
            fdata['preds'][row['id']] = np.array(
                row[self.data.train.classes].tolist())
            imgs.append(img)
        jsondump = dict(imgs=imgs)
        output_name = '{}_{}_preds.json'.format(self.model.expname, split)
        output_file = ojoin(self.model.model_folder, output_name)
        json.dump(jsondump, open(output_file, 'w+'))
        return df

    def gen_score(self, split='train', verbose=True):

        if verbose:
            print('*' * 50)
            print('{} set: {}'.format(split, self.model.model_name).upper())
            print('*' * 50 + '\n')
            print('Dataset: {} \n'.format(self.data.data_name))

        resFile = ojoin(self.model.model_folder, '{}_{}_preds.json'.format(
            self.model.expname, split))
        Gt = getattr(self.data, split)
        Dt = Gt.loadRes(resFile)
        cocoEval = COCOPriMatrixEval(cocoGt=Gt, cocoDt=Dt)
        score = cocoEval.evaluate()
        score = pd.DataFrame(score, columns=Gt.classes)
        return score.mean()

    def detailed_report(self, train, val, test=None):
        p = ojoin(ROOT_DIR, 'reports', self.modeltype, self.model.model_task,
                  self.model.model_name)
        self._mkdir_recursive(p)
        fname = ojoin(ROOT_DIR, 'reports', self.modeltype,
                      self.model.model_task, self.model.model_name,
                      '{}.md'.format(self.model.expname))
        with open(fname, 'w+') as f:
            f.write('# Best model \n\n')
            f.write('- model task : {} \n'.format(self.model.model_task))
            f.write('- model name : {} \n'.format(self.model.model_name))
            f.write('- data name : {} \n'.format(self.data.data_name))
            f.write('- model description : {} \n'.format(
                self.model.model_description))
            f.write('- experiment : {} \n'.format(self.model.expname))
            f.write('- model folder : {} \n'.format(self.model.model_folder))
            f.write('- creation time : {} \n\n'.format(
                time.strftime("%d/%m/%Y")))
            f.write('# Best model - training summary \n\n')
            f.write('## Training set \n\n')
            f.write(
                tabulate(
                    train.values.reshape(1, -1),
                    headers=train.index,
                    tablefmt="pipe"))
            f.write('\n\n')
            f.write('## Validation set \n\n')
            f.write(
                tabulate(
                    val.values.reshape(1, -1),
                    headers=val.index,
                    tablefmt="pipe"))
            f.write('\n\n')
            if test is not None:
                f.write('# Best model - test summary \n\n')
                f.write(
                    tabulate(
                        test.values.reshape(1, -1),
                        headers=test.index,
                        tablefmt="pipe"))
                f.write('\n\n')

            if hasattr(self.model, 'experiment'):
                f.write('\n\n')
                f.write('# Exepriment description \n\n')
                f.write('```python \n')
                f.write(pprint.pformat(self.model.experiment, indent=4))
                f.write('\n```')

    def _mkdir_recursive(self, path):
        '''
        Small function to recursively create a folder
        '''
        sub_path = os.path.dirname(path)
        if not os.path.exists(sub_path):
            self._mkdir_recursive(sub_path)
        if not os.path.exists(path):
            os.mkdir(path)

    def summary_report(self, train, val, test=None, ref='clement'):
        p = ojoin(ROOT_DIR, 'reports', self.modeltype, self.model.model_task,
                  self.model.model_name)
        self._mkdir_recursive(p)

        test_score = None
        if test is not None:
            test_score = test.mean()
        data = dict(
            dataset=self.data.data_name,
            expname=self.model.expname,
            name=self.model.model_name,
            train_score=train.mean(),
            val_score=val.mean(),
            test_score=test_score,
            ref=ref,
            report='[report]({})'.format(
                ojoin(
                    'https://github.com/cthorey/{0}/blob/master/reports/{1}/{2}/{3}/{4}.md'.
                    format(self.project, self.modeltype, self.model.model_task,
                           self.model.model_name, self.model.expname))),
            description=self.model.model_description,
            created=time.strftime("%d/%m/%Y"))
        # Make sure we preserve this order
        header = [
            'dataset', 'expname', 'name', 'train_score', 'val_score',
            'test_score', 'ref', 'report', 'description', 'created'
        ]
        df = pd.DataFrame([[data[f] for f in header]], columns=header)
        folder = self.model.model_folder
        df.to_csv(
            ojoin(folder, 'report_{}.csv'.format(self.model.expname)),
            index=None)
        df.to_csv(
            ojoin(
                ojoin(ROOT_DIR, 'reports', self.modeltype,
                      self.model.model_task, self.model.model_name,
                      'reports_{}.csv'.format(self.model.expname))),
            index=None)

    def save_report(self, ref='clement'):
        '''
        Save a report for that model that can can be used
        as a summary of the model performance on this dataset.

        parameters:
        ref: Name of the person training the models
        report: Notebook where to find the report
        finetuned: Boolean to specify if the model has been fine tunned or not
        '''

        # detail report
        train = self.gen_score('train', verbose=False)
        val = self.gen_score('validation', verbose=False)
        try:
            test = self.gen_score('test', verbose=False)
        except:
            test = None

        print('Generating a detailed report')
        self.detailed_report(train, val, test)
        print('Generating a summary report')
        self.summary_report(train, val, test, ref=ref)
