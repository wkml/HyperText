#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：main.py
@Author  ：wkml4996
@Date    ：2021/9/24 14:09 
"""
from utils import build_dataset, build_dataloader
from config import Config
from model import Model
from mindspore import context

context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")


dataset = './data/tnews_public'
embedding = 'random'
outputdir = './output'

config = Config(dataset, outputdir, embedding)

vocab, train_data, dev_data, test_data = build_dataset(config, False, min_freq=1)
train_data.initial()
train_iter = build_dataloader(train_data, config.batch_size, False)

config.n_vocab = len(vocab)

model = Model(config)

for i, data in enumerate(train_iter.create_dict_iterator()):
    output = model(data)
    print(output)
