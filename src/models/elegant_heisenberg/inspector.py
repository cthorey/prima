import json
import os
from os.path import join as ojoin

import h5py
import numpy as np
import pandas as pd


class PriMatrixInspector(object):
    '''
    A general inspector to inspect the result for the competition
    '''

    def __init__(self, data, model):
        self.data = data
        self.model = model

    def gen_prediction(self, split='train', batch_size=16):
        """
        Gen prediction and dump then to disk
        """
        reader = getattr(self.data, split)
        img_ids = reader.getImgIds()

        predictions = self.model.predict(
            image_ids=img_ids,
            fdata=reader.fdata['data'],
            batch_size=batch_size,
            verbose=True)

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
