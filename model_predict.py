# -*- coding: utf-8 -*-
# @Time : 2021/1/15 16:13
# @Author : Jclian91
# @File : model_predict.py
# @Place : Yangpu, Shanghai
import json
import numpy as np

from util import dataset
from model_train import convert_example_to_feature
from transformers import TFBertForSequenceClassification

with open("{}_label.json".format(dataset), "r", encoding="utf-8") as f:
    labels = json.loads(f.read())

cls_model = model = TFBertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=len(labels))
cls_model.load_weights("{}_cls.h5".format(dataset))

text = "在2020年的广州车展亮相之后，新一代现代名图终于在1月12日宣布开启预售。新车的优越先享价，也就是预售价格为13.58-17.58万元，" \
       "将于2021年1季度正式上市。作为一款定位于A+级别的车型，它的前辈凭借出色的空间、丰富的配置和相对实惠的价格取得了不错的销售" \
       "成绩与用户口碑，只是换代不及时导致名图的关注度日渐低迷。现如今新款车型来袭，不知道还能不能取得曾经的成绩。"

# model predict
bert_input = convert_example_to_feature(text)
input_ids_list = [bert_input['input_ids']]
token_type_ids_list = [bert_input['token_type_ids']]
attention_mask_list = [bert_input['attention_mask']]

test_ds = [np.array(input_ids_list), np.array(attention_mask_list), np.array(token_type_ids_list)]
print(labels[np.argmax(cls_model.predict(test_ds).logits, axis=1)[0]])
