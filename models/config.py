#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：config.py
@Author  ：wkml4996
@Date    ：2021/9/23 21:28 
"""
import os
import mindspore
import numpy as np


class Config(object):
    """hyperparameter configuration"""

    def __init__(self, datasetdir, outputdir, embedding):
        self.model_name = 'HyperText'
        self.train_path = os.path.join(datasetdir, 'train.txt')
        self.dev_path = os.path.join(datasetdir, 'dev.txt')
        self.test_path = os.path.join(datasetdir, 'test.txt')
        self.class_list = []
        self.vocab_path = os.path.join(datasetdir, 'vocab.txt')
        self.labels_path = os.path.join(datasetdir, 'labels.txt')
        self.save_path = os.path.join(outputdir, self.model_name + '.ckpt')
        self.log_path = os.path.join(outputdir, self.model_name + '.log')
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
        # pretrained word embedding 预训练的词向量
        self.embedding_pretrained = mindspore.Tensor(
            np.load(os.path.join(datasetdir, embedding))["embeddings"].astype('float32')) \
            if embedding != 'random' else None
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # cuda or cpu
        self.dropout = 0.5
        self.require_improvement = 2500  # max patient steps when early stoping
        self.num_classes = len(self.class_list)  # label number
        self.n_vocab = 0
        self.num_epochs = 30
        self.wordNgrams = 2
        self.batch_size = 32
        self.max_length = 1000
        self.learning_rate = 1e-2
        self.embed = self.embedding_pretrained.size(1) \
            if self.embedding_pretrained is not None else 100
        self.bucket = 20000  # word and ngram vocab size
        self.lr_decay_rate = 0.96
