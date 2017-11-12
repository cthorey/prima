import time
from os.path import join as ojoin

import pandas as pd
from keras.preprocessing.image import *
from PIL import Image
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.pipeline import FeatureUnion, Pipeline
from src.data.coco_video import COCOPriMatrix
from src.models.youthful_edison import transformer


class MetaDataGenerator(object):
    def __init__(self,
                 dummy_keys=[],
                 dummy_pca=10,
                 real_keys=[],
                 real_scaler='RobustScaler',
                 normalize=True):
        self.dummy_keys = dummy_keys
        self.dummy_pca = dummy_pca
        self.real_keys = real_keys
        self.real_scaler = real_scaler
        self.normalize = normalize

    def standardize(self, u):
        # dummy varialbe
        dummy_pipelines = []
        for key in self.dummy_keys:
            p = Pipeline([('extraction', getattr(
                transformer,
                'FeatureSelector')(cols=[key])), ('encoding', getattr(
                    transformer, 'OneHotEncoder')(sparse=False))])
            p = (key, p)
            dummy_pipelines.append(p)
        dummy_features = FeatureUnion(dummy_pipelines)
        if self.dummy_pca is not None:
            dummy_pipeline = Pipeline([('dummy_feature_extraction',
                                        dummy_features), ('PCA', PCA(
                                            self.dummy_pca))])
        else:
            dummy_pipeline = Pipeline([('dummy_feature_extraction',
                                        dummy_features)])

        # real variable
        real_pipelines = []
        for key in self.real_keys:
            p = Pipeline([('extraction', getattr(
                transformer, 'FeatureSelector')(cols=[key])),
                          ('scaler', getattr(transformer,
                                             self.real_scaler)())])
            p = (key, p)
            real_pipelines.append(p)
        real_features = FeatureUnion(real_pipelines)
        real_pipeline = Pipeline([('real_feature', real_features)])

        pipeline = FeatureUnion([('dummy_feature', dummy_pipeline),
                                 ('real_feature', real_pipeline)])

        if self.normalize:
            pipeline = Pipeline([('feature', pipeline), ('normalize', getattr(
                transformer, 'FeatureNormalizer')())])

        return pipeline.fit_transform(u)

    def flow_from_directory(self,
                            directory,
                            split,
                            batch_size=32,
                            shuffle=True,
                            seed=None):
        return MetaDataIterator(
            directory,
            self,
            split,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed)


class MetaDataIterator(Iterator):
    def __init__(self,
                 directory,
                 data_generator,
                 split,
                 batch_size=32,
                 shuffle=True,
                 seed=None):

        self.directory = directory
        self.data_generator = data_generator
        self.data = COCOPriMatrix(
            ojoin(self.directory, 'annotations',
                  'instances_{}.json'.format(split)))
        self.filenames = self.data.getImgIds()
        self.nb_sample = len(self.filenames)
        self.num_classes = len(self.data.cats)
        super(MetaDataIterator, self).__init__(self.nb_sample, batch_size,
                                               shuffle, seed)
        df = pd.DataFrame([
            self.data.imgs[k]['meta_data'] for k in range(len(self.data.imgs))
        ])
        self.X = self.data_generator.standardize(df)
        self.y = self.data.fdata['labels'][:]

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(
                self.index_generator)

        batch_x = self.X[index_array]
        batch_y = self.y[index_array]

        return batch_x, batch_y
