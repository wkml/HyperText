#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：utils.py
@Author  ：wkml4996
@Date    ：2021/9/23 21:30 
"""

import os
import mindspore
import time
import random
from datetime import timedelta

import numpy as np
from mindspore.dataset import GeneratorDataset
from mindspore import Tensor

MAX_VOCAB_SIZE = 5000000
UNK, PAD = '<UNK>', '<PAD>'


def hash_str(gram_str):
    """
    哈希函数
    :param gram_str:
    :return:
    """
    gram_bytes = bytes(gram_str, encoding='utf-8')
    hash_size = 18446744073709551616
    h = 2166136261
    for gram in gram_bytes:
        h = h ^ gram
        h = (h * 1677619) % hash_size
    return h


def addWordNgrams(hash_list, n, bucket):
    """
    为每一句话添加ngram语法
    :param hash_list:
    :param n:
    :param bucket:
    :return:
    """
    ngram_hash_list = []
    len_hash_list = len(hash_list)
    for index, hash_val in enumerate(hash_list):
        bound = min(len_hash_list, index + n)

        for i in range(index + 1, bound):
            hash_val = hash_val * 116049371 + hash_list[i]
            ngram_hash_list.append(hash_val % bucket)

    return ngram_hash_list


def load_vocab(vocab_path, max_size, min_freq):
    """
    加载词汇表（语料库）
    :param vocab_path:
    :param max_size:
    :param min_freq:
    :return:
    """
    vocab = {}
    with open(vocab_path, 'r', encoding="utf-8") as fhr:
        for line in fhr:
            line = line.strip()
            line = line.split(' ')
            if len(line) != 2:
                continue
            token, count = line
            vocab[token] = int(count)
    # 对每个字用频率排名代替
    sorted_tokens = sorted([item for item in vocab.items() if item[1] >= min_freq], key=lambda x: x[1], reverse=True)
    sorted_tokens = sorted_tokens[:max_size]
    all_tokens = [[PAD, 0], [UNK, 0]] + sorted_tokens
    vocab = {item[0]: i for i, item in enumerate(all_tokens)}
    return vocab


def load_labels(label_path):
    """
    加载标签
    :param label_path:
    :return:
    """
    labels = []
    with open(label_path, 'r', encoding="utf-8") as fhr:
        for line in fhr:
            line = line.strip()
            if line not in labels:
                labels.append(line)
    return labels


def build_vocab(file_path, tokenizer, max_size, min_freq):
    """
    建立语料库（字粒度）
    :param file_path:
    :param tokenizer:
    :param max_size:
    :param min_freq:
    :return:
    """
    vocab_dic = {}
    label_set = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # line_splits = line.split("\t")
            # if len(line_splits) != 2:
            #     print(line)
            # content, label = line_splits
            # label_set.add(label.strip())

            line_splits = line.split("_!_")
            if len(line_splits) <= 3:
                print(line)
            content, label = line_splits[3], line_splits[1]
            label_set.add(label.strip())

            for word in tokenizer(content.strip()):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1

        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[
                     :max_size]

        vocab_list = [[PAD, 111101], [UNK, 111100]] + vocab_list
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}

    base_datapath = os.path.dirname(file_path)
    with open(os.path.join(base_datapath, "vocab.txt"), "w", encoding="utf-8") as f:
        for w, c in vocab_list:
            f.write(str(w) + " " + str(c) + "\n")
    with open(os.path.join(base_datapath, "labels.txt"), "w", encoding="utf-8") as fr:
        labels_list = list(label_set)
        labels_list.sort()
        for l in labels_list:
            fr.write(l + "\n")
    return vocab_dic, list(label_set)


def build_dataset(config, use_word, min_freq=5):
    """建立数据集"""
    print("use min words freq:%d" % (min_freq))
    if use_word:
        tokenizer = lambda x: x.split(' ')  # word-level 分为字符串
    else:
        tokenizer = lambda x: [y for y in x]  # char-level 分为单个字符
    # 先建立语料库
    _ = build_vocab(config.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=min_freq)

    vocab = load_vocab(config.vocab_path, max_size=MAX_VOCAB_SIZE, min_freq=min_freq)
    print(f"Vocab size: {len(vocab)}")
    # 所有句子的label(set)
    labels = load_labels(config.labels_path)
    print(f"label size: {len(labels)}")

    train = TextDataset(
        file_path=config.train_path,
        vocab=vocab,
        labels=labels,
        tokenizer=tokenizer,
        wordNgrams=config.wordNgrams,
        buckets=config.bucket,
        max_length=config.max_length,
        nraws=80000,
        shuffle=True
    )

    dev = TextDataset(
        file_path=config.dev_path,
        vocab=vocab,
        labels=labels,
        tokenizer=tokenizer,
        wordNgrams=config.wordNgrams,
        buckets=config.bucket,
        max_length=config.max_length,
        nraws=80000,
        shuffle=False
    )

    test = TextDataset(
        file_path=config.test_path,
        vocab=vocab,
        labels=labels,
        tokenizer=tokenizer,
        wordNgrams=config.wordNgrams,
        buckets=config.bucket,
        max_length=config.max_length,
        nraws=80000,
        shuffle=False
    )

    config.class_list = labels
    config.num_classes = len(labels)

    return vocab, train, dev, test


class TextDataset:
    """
    建立数据集
    """

    def __init__(self, file_path, vocab, labels, tokenizer, wordNgrams,
                 buckets, max_length=32, nraws=80000, shuffle=False):

        file_raws = 0
        with open(file_path, 'r', encoding="utf-8") as f:
            for _ in f:
                file_raws += 1
        self.file_path = file_path
        self.file_raws = file_raws
        if file_raws < 2000000:
            self.nraws = file_raws
        else:
            self.nraws = nraws
        self.shuffle = shuffle
        self.vocab = vocab
        self.labels = labels
        self.tokenizer = tokenizer
        self.wordNgrams = wordNgrams
        self.buckets = buckets
        self.max_length = max_length

    def process_oneline(self, line):
        """
        处理单独一句输入
        :param line:
        :return:
        """
        line = line.strip()
        line_splits = line.split("_!_")
        content, label = line_splits[3], line_splits[1]
        if len(content.strip()) == 0:
            content = "0"
        # 字粒度分解
        tokens = self.tokenizer(content.strip())
        seq_len = len(tokens)
        if seq_len > self.max_length:
            tokens = tokens[:self.max_length]
        # 将每个字用哈希函数处理
        token_hash_list = [hash_str(token) for token in tokens]
        # ngram语法列表
        ngram = addWordNgrams(token_hash_list, self.wordNgrams, self.buckets)
        ngram_pad_size = int((self.wordNgrams - 1) * (self.max_length - self.wordNgrams / 2))

        if len(ngram) > ngram_pad_size:
            ngram = ngram[:ngram_pad_size]
        # 对每个字用频率排名代替，id为排名
        tokens_to_id = [self.vocab.get(token, self.vocab.get(UNK)) for token in tokens]
        # y是标签
        y = self.labels.index(label.strip())

        return tokens_to_id, ngram, y

    def initial(self):
        """
        初始化，添加sample， 格式为为(token_id,ngram,label) ->(list,list,int)
        :return:
        """
        self.finput = open(self.file_path, 'r', encoding="utf-8")
        self.samples = list()

        for _ in range(self.nraws):
            line = self.finput.readline()
            if line:
                preprocess_data = self.process_oneline(line)
                self.samples.append(preprocess_data)
            else:
                break
        self.current_sample_num = len(self.samples)
        self.index = list(range(self.current_sample_num))
        if self.shuffle:
            random.shuffle(self.samples)

    def __len__(self):
        return self.current_sample_num

    def __getitem__(self, item):
        """原文处理照搬，每调用一次，index后移，长度-1"""
        idx = self.index[0]
        # one_sample = (token_id,ngram,label) ->(list,list,int)
        one_sample = self.samples[idx]
        data1 = np.array(one_sample[0])
        data2 = np.array(one_sample[1])
        label = np.array(one_sample[2])

        self.index = self.index[1:]
        self.current_sample_num -= 1

        if self.current_sample_num <= 0:

            for _ in range(self.nraws):
                line = self.finput.readline()
                if line:
                    preprocess_data = self.process_oneline(line)
                    self.samples.append(preprocess_data)
                else:
                    break
            self.current_sample_num = len(self.samples)
            self.index = list(range(self.current_sample_num))
            if self.shuffle:
                random.shuffle(self.samples)

        return data1, data2, label


def _pad(data, width=-1):
    """
    padding
    :param data:list
    :param pad_id:
    :param width:
    :return:
    """
    if (width == -1):
        width = max(d.shape[0] for d in data)
    # rtn_data = [d.tolist() + [pad_id] * (width - len(d)) for d in data]
    for d in data:
        d.resize(width, refcheck=False)
    return data


def text_collate_fn(x_1, x_2,batchInfo):
    x_1 = _pad(x_1)
    x_2 = _pad(x_2)

    return (x_1, x_2)


def build_dataloader(dataset, batch_size, shuffle=False):
    dataloader = GeneratorDataset(
        source=dataset,
        column_names=['id', 'ngram', 'label'],
    ).batch(batch_size=batch_size, input_columns=['id', 'ngram'], per_batch_map=text_collate_fn)

    return dataloader
