import os
import sys
from os.path import join as ojoin

import keras.backend as K
import tensorflow as tf
from ai_platform.src.model import pretrained_models
from ai_platform.src.model.common.base_model import *
from keras.layers import (Activation, BatchNormalization, Conv2D, Dense,
                          Dropout, Flatten, MaxPooling2D, TimeDistributed)
from keras.layers.recurrent import LSTM
from keras.models import Sequential, model_from_json
from PIL import Image
from src.models.youthful_edison.preprocessing import *
from tqdm import *

ROOT_DIR = os.environ['ROOT_DIR']
sys.path.append(ROOT_DIR)


class Model(BaseModel):
    """
    Description
    """

    def __init__(self,
                 model_name='youthful_edison',
                 model_task='video_classification',
                 model_description="meta data model",
                 stage='training',
                 bmodel=None,
                 input_shape=(12),
                 expname=None,
                 weights_file='min',
                 serve_model_from=None,
                 **kwargs):
        self.model_name = model_name
        self.model_task = model_task
        self.model_description = model_description
        self.serve_model_from = serve_model_from
        self.stage = stage
        self.input_shape = input_shape
        self.setup_foldertree()
        self.load_exp(expname, weights_file, bmodel)

    def preprocessing_input(self, df, prepro_kwargs=None):
        """
        Pre-process a multiview image (i.e. a set of images)

        :param video: a video
        :param prepro_kwargs:
        :return:
        """
        if prepro_kwargs is None:
            assert hasattr(self,
                           'experiment'), 'Set up the prepro_kwargs argument'
            if 'feature_config' in self.experiment.keys():
                prepro_kwargs = self.experiment.feature_config.validation_augmentation
            elif self.experiment['data_config'] is not None:
                prepro_kwargs = self.experiment.data_config.validation_augmentation

        prepro = MetaDataGenerator(**prepro_kwargs)
        return prepro.standardize(df)

    def decode_predictions(self, y):
        pass

    def predict(self, reader, batch_size, verbose=False):
        df = pd.DataFrame(
            [reader.imgs[k]['meta_data'] for k in range(len(reader.imgs))])
        X = self.preprocessing_input(df)
        preds = self.model.predict(X, batch_size=batch_size, verbose=verbose)
        ids = np.array(reader.getImgIds())
        return np.hstack((np.expand_dims(ids, -1), preds))

    def load_model(self,
                   nlayers=1,
                   input_shape=(12),
                   dropout=0.5,
                   nb_neuron=64,
                   num_classes=24,
                   **kwargs):

        input_data = Input(shape=input_shape, name='input')
        x = input_data

        for i in range(nlayers):
            x = Dense(nb_neuron, name='dense_{}'.format(i))(x)
            x = Dropout(dropout)(x)
            x = Activation('relu')(x)

        x = Dense(num_classes, name='classification')(x)
        x = Activation('sigmoid')(x)

        model = KerasModel(input_data, x)

        return model
