"""
Object detector using the ported version of SSD300 for keras.

In this model, we are trying to reproduce the result from the PAPER on the VOC dataset.
"""
import os
import sys

import tensorflow as tf
from ai_platform.src.experiment.experiment import Experiment
from src.models.boring_leavitt.data import Data
from src.models.boring_leavitt.model import Model

ROOT_DIR = os.environ['ROOT_DIR']

if __name__ == '__main__':
    """
    Train a model for regression
    """
    data = Data(data_name='prima_d1_500_image')
    model = Model(
        stage='training', bmodel='Xception', model_description='Xception')
    experiment = Experiment(data=data, model=model)

    finetunning_step0 = dict(
        layer_name='head_conv_1',
        fit_config=dict(epochs=40),
        training_config=dict())

    finetunning_step1 = dict(
        layer_name='block14_sepconv1_act',
        fit_config=dict(epochs=40),
        training_config=dict(
            optimizer='SGD',
            optimizer_config=dict(
                lr=1e-3, momentum=0.9, decay=5e-3, nesterov=True)))
    finetunning_step2 = dict(
        layer_name='block13_sepconv1_act',
        fit_config=dict(epochs=50),
        training_config=dict(
            optimizer='SGD',
            optimizer_config=dict(
                lr=1e-3, momentum=0.9, decay=5e-3, nesterov=True)))
    finetunning_step3 = dict(
        layer_name='block12_sepconv1_act',
        fit_config=dict(epochs=50),
        training_config=dict(
            optimizer='SGD',
            optimizer_config=dict(
                lr=1e-4, momentum=0.9, decay=5e-3, nesterov=True)))
    finetunning_step4 = dict(
        layer_name='block11_sepconv1_act',
        fit_config=dict(epochs=50),
        training_config=dict(
            optimizer='SGD',
            optimizer_config=dict(
                lr=1e-4, momentum=0.9, decay=5e-3, nesterov=True)))
    finetunning_step5 = dict(
        layer_name='block10_sepconv1_act',
        fit_config=dict(epochs=500),
        training_config=dict(
            optimizer='SGD',
            optimizer_config=dict(
                lr=1e-4, momentum=0.9, decay=5e-3, nesterov=True)))

    finetunning = dict(
        step0=finetunning_step0,
        step1=finetunning_step1,
        step2=finetunning_step2,
        step3=finetunning_step3,
        step4=finetunning_step4,
        step5=finetunning_step5)

    fit_config = dict(steps_per_epoch=150, epochs=1000)
    training_config = dict(
        model_config=dict(
            input_shape=(224, 224, 3),
            include_top=False,
            num_classes=24,
            last_layer=None,
            nb_neuron=1024),
        optimizer='Adam',
        optimizer_config=dict(lr=0.001),
        loss='binary_crossentropy',
        metrics=['binary_crossentropy'],
        finetunning=finetunning)

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
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest'),
        batch_size=32,
        shuffle=True)

    # setup
    expname = 'exp14'
    exp = dict(
        fit_config=fit_config,
        training_config=training_config,
        callback_config=callback_config,
        data_config=data_config)
    experiment.fit_experiment(exp, expname=expname, overwrite=True)
