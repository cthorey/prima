"""
Object detector using the ported version of SSD300 for keras.

In this model, we are trying to reproduce the result from the PAPER on the VOC dataset.
"""
import os
import sys

import tensorflow as tf
from ai_platform.src.experiment.experiment import Experiment
from src.models.elegant_heisenberg.data import Data
from src.models.elegant_heisenberg.model import Model

ROOT_DIR = os.environ['ROOT_DIR']

if __name__ == '__main__':
    """
    Train a model for regression
    """
    data = Data(data_name='prima_d1_500')
    model = Model(
        model_task='video_classification',
        model_name='elegant_heisenberg',
        stage='training',
        bmodel=None,
        model_description='benchmark')
    experiment = Experiment(data=data, model=model)

    fit_config = dict(steps_per_epoch=150, epochs=150)
    training_config = dict(
        model_config={
            'num_classes': len(data.train.cats),
            'input_shape': (6, 128, 128, 3)
        },
        optimizer='RMSprop',
        optimizer_config=dict(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0),
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
        base_augmentation=dict(rescale=1. / 255),
        train_aug_specific=dict(
            rotation_range=15,
            width_shift_range=0.05,
            height_shift_range=0.05,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest'),
        batch_size=5,
        shuffle=True)

    # setup
    expname = 'exp2'
    exp = dict(
        fit_config=fit_config,
        training_config=training_config,
        callback_config=callback_config,
        data_config=data_config)
    experiment.fit_experiment(exp, expname=expname, overwrite=True)
