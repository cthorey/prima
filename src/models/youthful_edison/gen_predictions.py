"""
Object detector using the ported version of SSD300 for keras.

In this model, we are trying to reproduce the result from the PAPER on the VOC dataset.
"""
import os
import sys

import tensorflow as tf
from ai_platform.src.experiment.experiment import Experiment
from ai_platform.src.utils.report import Reports
from src.inspector.inspector import PriMatrixInspector
from src.models.youthful_edison.data import Data
from src.models.youthful_edison.model import Model

ROOT_DIR = os.environ['ROOT_DIR']

if __name__ == '__main__':
    """
    Train a model for regression
    """
    data = Data(data_name='prima_d1_500_image')
    for exp in ['exp0', 'exp1']:
        model = Model(stage='prediction', expname=exp)
        inspector = PriMatrixInspector(data=data, model=model)
        inspector.gen_prediction(split='train', batch_size=32)
        inspector.gen_prediction(split='validation', batch_size=32)
        inspector.save_report()
    report = Reports(monitor='val_score')
    report.reports2md()
