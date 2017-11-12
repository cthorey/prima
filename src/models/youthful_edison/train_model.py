"""
Object detector using the ported version of SSD300 for keras.

In this model, we are trying to reproduce the result from the PAPER on the VOC dataset.
"""
import os
import sys

import tensorflow as tf
from ai_platform.src.experiment.experiment import Experiment
from src.models.youthful_edison.data import Data
from src.models.youthful_edison.model import Model

ROOT_DIR = os.environ['ROOT_DIR']

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
        patience=35,
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
    expname = 'exp1'
    exp = dict(
        fit_config=fit_config,
        training_config=training_config,
        callback_config=callback_config,
        data_config=data_config)
    experiment.fit_experiment(exp, expname=expname, overwrite=True)
