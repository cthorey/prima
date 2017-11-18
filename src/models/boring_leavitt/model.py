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
from src.models.elegant_heisenberg.preprocessing import VideoDataGenerator
from tqdm import *

ROOT_DIR = os.environ['ROOT_DIR']
sys.path.append(ROOT_DIR)


class Model(BaseModel):
    """
    Description
    """

    def __init__(self,
                 model_name='boring_leavitt',
                 model_task='video_classification',
                 model_description="Based on RESNET 50",
                 stage='training',
                 bmodel=None,
                 input_shape=(224, 224, 3),
                 expname=None,
                 weights_file='min',
                 serve_model_from=None,
                 **kwargs):
        self.model_name = model_name
        self.model_task = model_task
        self.model_description = model_description
        self.serve_model_from = serve_model_from
        self.input_shape = input_shape
        self.stage = stage
        self.setup_foldertree()
        self.load_exp(expname, weights_file, bmodel)

    def preprocessing_input(self, image, prepro_kwargs=None):
        """
        Pre-process a multiview image (i.e. a set of images)
        """
        if prepro_kwargs is None:
            assert hasattr(self,
                           'experiment'), 'Set up the prepro_kwargs argument'
            if 'feature_config' in self.experiment.keys():
                prepro_kwargs = self.experiment.feature_config.validation_augmentation
            elif self.experiment['data_config'] is not None:
                prepro_kwargs = self.experiment.data_config.validation_augmentation

        target_size = self.experiment.data_config.validation_target_size
        video = np.expand_dims(np.array(image), 0).astype('float')
        prepro = VideoDataGenerator(**prepro_kwargs)
        video = prepro.standardize(video)
        return video[0]

    def decode_predictions(self, y):
        pass

    def predict(self, reader, batch_size, verbose=False):

        image_ids = np.array(reader.getImgIds())

        N = len(image_ids) / batch_size
        idxs = range(len(image_ids))
        preds = []
        dis = not verbose
        for batch_idx in tqdm(
                chunked_iter(idxs, size=batch_size), disable=dis, total=N):
            batch_ids = image_ids[batch_idx]
            X = np.stack([
                self.preprocessing_input(
                    Image.open(reader.imgs[idx]['files'][0]))
                for idx in batch_ids
            ])
            pred = self.model.predict(X, batch_size=batch_size, verbose=False)
            pred = np.hstack((batch_ids.reshape(-1, 1), pred))
            preds.append(pred)
        preds = np.vstack(preds)
        return preds

    def load_head(self,
                  plug_on_top=True,
                  nb_neuron=2048,
                  num_classes=24,
                  **kwargs):

        if plug_on_top is not None:
            base = plug_on_top
            input_data = base.output
        else:
            base = self.load_basemodel(**kwargs)
            input_data = Input(shape=base.output_shape[1:], name='Input')

        x = Conv2D(
            nb_neuron,
            (input_data._keras_shape[1], input_data._keras_shape[2]),
            name='head_conv_1')(input_data)
        x = BatchNormalization(axis=3, name='head_batchnorm_1')(x)
        x = Activation('relu')(x)

        x = Conv2D(nb_neuron, (1, 1), name='head_conv_2')(x)
        x = BatchNormalization(axis=3, name='head_batchnorm_2')(x)
        x = Activation('relu')(x)

        x = Flatten()(x)
        x = Dense(num_classes)(x)
        x = Activation('sigmoid')(x)

        return x

    def load_model(self, **kwargs):

        base = self.load_basemodel(**kwargs)
        head = self.load_head(plug_on_top=base, **kwargs)
        model = KerasModel(base.input, head)

        return model
