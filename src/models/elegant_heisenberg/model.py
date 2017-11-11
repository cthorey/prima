import os
import sys
from os.path import join as ojoin

import keras.backend as K
import tensorflow as tf
from ai_platform.src.model import pretrained_models
from ai_platform.src.model.common.base_model import *
from keras.layers import (Conv2D, Dense, Dropout, Flatten, MaxPooling2D,
                          TimeDistributed)
from keras.layers.recurrent import LSTM
from keras.models import Sequential, model_from_json
from src.models.elegant_heisenberg.preprocessing import VideoDataGenerator
from tqdm import *

ROOT_DIR = os.environ['ROOT_DIR']
sys.path.append(ROOT_DIR)


class Model(BaseModel):
    """
    Description
    """

    def __init__(self,
                 model_name,
                 model_task,
                 model_description="",
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

    def preprocessing_input(self, arr, prepro_kwargs=None):
        """
        Pre-process a multiview image (i.e. a set of images)

        :param arr: a multiview image
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

        prepro = VideoDataGenerator(**prepro_kwargs)
        return prepro.standardize(arr)

    def decode_predictions(self, y):
        pass

    def predict(self, image_ids, fdata, batch_size, verbose=False):
        if type(image_ids) != list:
            image_ids = [image_ids]

        image_ids = np.array(image_ids)

        N = len(image_ids) / batch_size
        idxs = range(len(image_ids))
        preds = []
        dis = not verbose
        for batch_idx in tqdm(
                chunked_iter(idxs, size=batch_size), disable=dis, total=N):
            batch_ids = image_ids[batch_idx]
            X = np.stack(
                [self.preprocessing_input(fdata[idx]) for idx in batch_ids])
            pred = self.model.predict_proba(
                X, batch_size=batch_size, verbose=False)
            pred = np.hstack((batch_ids.reshape(-1, 1), pred))
            preds.append(pred)
        preds = np.vstack(preds)
        return preds

    def load_model(self, num_frames, width, height, num_classes, *args,
                   **kwargs):

        model = Sequential()

        # add three time-distributed convolutional layers for feature extraction
        model.add(
            TimeDistributed(
                Conv2D(64, (3, 3), activation='relu'),
                input_shape=(num_frames, width, height, 3)))
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(1, 1))))
        model.add(TimeDistributed(Conv2D(128, (4, 4), activation='relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))
        model.add(TimeDistributed(Conv2D(256, (4, 4), activation='relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))
        # extract features and dropout
        model.add(TimeDistributed(Flatten()))
        model.add(Dropout(0.5))
        # input to LSTM
        model.add(LSTM(256, return_sequences=False, dropout=0.5))
        # classifier with sigmoid activation for multilabel
        model.add(Dense(num_classes, activation='sigmoid'))
        # compile the model with binary_crossentropy loss for multilabel
        model.compile(optimizer='rmsprop', loss='binary_crossentropy')
        # look at the params before training
        model.summary()

        return model

    def forward_pass_from_generator(self, generator, N):
        """
        Customize the forward pass to ease the dumping
        of features from an architecture (egg: VGG16).
        To use instead of **predict_generator**.

        In contrast to predict_generator, this method return not
        only the prediction, but also the labels, a dictionary that
        maps the label to their indices and the class mode
        use.
        """
        # forward pass
        for i in range(N):
            X_batch, y_batch = generator.next()
            features = self.model.predict_on_batch(X_batch)
            labels = y_batch

            yield features, labels
