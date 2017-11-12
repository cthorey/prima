import copy
import operator
import os
import sys
from functools import reduce  # forward compatibility for Python 3

import numpy as np
import tensorflow as tf
from ai_platform.src.experiment.experiment import Experiment
from ai_platform.src.utils.oscar_tunning import *
from future.utils import iteritems
from src.models.youthful_edison.data import Data
from src.models.youthful_edison.model import Model

ROOT_DIR = os.environ['ROOT_DIR']


class Config():
    OSCAR_API_TOKEN_ACCES = '40ZGpGE1neCsubL2rB0TwcXDhgqLOuuroqGWhGUtag9zfnNlbnNvdXQtb3NjYXJyEQsSBFVzZXIYgICAgN7PjQoM'


if __name__ == '__main__':
    """
    Train a model for regression
    """
    data = Data(data_name='prima_d1_500_image')
    model = Model(stage='training')
    experiment = Experiment(data=data, model=model)

    npca = 10

    fit_config = dict(steps_per_epoch=1000, epochs=150)
    training_config = dict(
        model_config=dict(
            nlayers=3,
            input_shape=(2 + npca, ),
            dropout=0.5,
            nb_neuron=512,
            num_classes=24),
        optimizer='RMSprop',
        optimizer_config=dict(lr=0.0001),
        loss='binary_crossentropy',
        metrics=['binary_crossentropy'],
        finetunning=None)

    callback_config = dict(
        patience=25,
        monitor='val_loss',
        mode='min',
        keep_only_n=5,
        cycliclr=False,
        reducelr=True,
        write_graph=True,
        write_images=False)
    data_config = dict(
        base_augmentation=dict(
            dummy_keys=[
                'coded_height', 'coded_width', 'level', 'nb_frames',
                'start_time', 'duration'
            ],
            dummy_pca=npca,
            real_keys=['bit_rate', 'size'],
            real_scaler='RobustScaler',
            normalize=False),
        train_aug_specific=dict(),
        batch_size=64,
        shuffle=True)

    # setup
    exp = dict(
        fit_config=fit_config,
        training_config=training_config,
        callback_config=callback_config,
        data_config=data_config)

    parameters = dict(
        training_config__model_config__nlayers=[1, 2, 3, 4],
        training_config__model_config__nb_neuron=[32, 64, 128, 512, 1024],
        training_config__model_config__dropout=[0.25, 0.5, 0.75],
        training_config__optimizer=['RMSprop', 'Adam'],
        training_config__optimizer_config__lr=[
            1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4
        ],
        data_config__batch_size=[64, 128, 256, 512],
        data_config__base_augmentation__real_scaler=[
            'MinMaxScaler', 'StandardScaler', 'RobustScaler'
        ],
        data_config__base_augmentation__normalize=[False, True])
    oscar_exp = dict(
        name='prima_2', description='metadata', parameters=parameters)
    config = Config()
    oscar = OscarTunning(config)
    oscar.finetune_from_experiment(
        experiment, exp, oscar_exp, inspector=None, max_attempt=450)
