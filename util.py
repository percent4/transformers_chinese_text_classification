# -*- coding: utf-8 -*-
# @Time : 2021/1/15 11:53
# @Author : Jclian91
# @File : util.py
# @Place : Yangpu, Shanghai

# 模型参数配置
EPOCH = 5
BATCH_SIZE = 16
MAX_SEQ_LENGTH = 300

# 数据集配置
dataset = "sougou_mini"
train_file_path = "./data/{}/train.csv".format(dataset)
test_file_path = "./data/{}/test.csv".format(dataset)
