import itertools

import numpy as np
import pandas as pd
import tensorflow as tf
from boltons.iterutils import chunked_iter
from box import Box
from keras import backend as K
from tqdm import *


class COCOPriMatrixEval(object):
    def __init__(self, cocoGt=None, cocoDt=None, scoreType='agglogloss'):
        '''
        Initialize CocoEval using coco APIs for gt and dt

        Args:
            cocoGt (obj): coco object with ground truth annotations
            cocoDt (obj): coco object with detection results
            scoreType (str): String indicating against which we evaluate the gt
            pc: point classification

        Returns:
            None
        '''

        assert scoreType in ['agglogloss'], 'Score type not valid'
        self.scoreType = scoreType
        self.cocoGt = cocoGt  # ground truth COCO API
        self.cocoDt = cocoDt  # detections COCO API

    def evaluate(self, batch_size=2):
        with tf.Session() as sess:
            labels = tf.constant(self.cocoGt.fdata['labels'])
            preds = tf.constant(self.cocoDt.fdata['preds'])
            score = K.binary_crossentropy(labels, preds).eval()

        return score
