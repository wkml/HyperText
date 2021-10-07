#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：main.py
@Author  ：wkml4996
@Date    ：2021/9/24 14:09 
"""
from mindspore.nn import SoftmaxCrossEntropyWithLogits,WithLossCell,WithEvalCell,WithGradCell
from utils import build_dataset, build_dataloader
from config import Config
from model import Model
from mindspore import context, nn
from step_utils import TrainOneStepCell,TrainWithLossCell,GradNetWithWrtParams

if __name__ == '__main__':
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    dataset = './data/tnews_public'
    embedding = 'random'
    outputdir = './output'

    config = Config(dataset, outputdir, embedding)

    vocab, train_data, dev_data, test_data = build_dataset(config, False, min_freq=1)
    train_data.initial()
    train_iter = build_dataloader(train_data, config.batch_size, False)

    config.n_vocab = len(vocab)

    net = Model(config)
    loss_func = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    opt = nn.Adam(params=net.trainable_params())

    loss_net = TrainWithLossCell(net, loss_func)
    # train_net = TrainOneStepCell(loss_net, opt)
    train_net = GradNetWithWrtParams(net)
    for i, data in enumerate(train_iter.create_dict_iterator()):
        # output = train_net(data['id'],data['ngram'])
        print(net(data['id'],data['ngram']))
        # loss = loss_net(data['id'], data['ngram'], data['label'])
        # train_net(data['id'], data['ngram'], data['label'])
        # print(loss)
        # print('\n')

    # for param in net.trainable_params():
    #     print(param)
