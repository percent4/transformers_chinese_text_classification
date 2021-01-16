# -*- coding: utf-8 -*-
# @Time : 2021/1/16 11:17
# @Author : Jclian91
# @File : model_evaluate.py
# @Place : Yangpu, Shanghai
import json
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from util import dataset, test_file_path
from model_train import convert_example_to_feature
from transformers import TFBertForSequenceClassification

with open("{}_label.json".format(dataset), "r", encoding="utf-8") as f:
    labels = json.loads(f.read())

cls_model = model = TFBertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=len(labels))
cls_model.load_weights("{}_cls.h5".format(dataset))

test_df = pd.read_csv(test_file_path).fillna(value="")

test_data = []
for i in range(test_df.shape[0]):
    label, content = test_df.iloc[i, :]
    test_data.append((content, label))

input_ids_list = []
token_type_ids_list = []
attention_mask_list = []
true_cls_labels = []

for review, label in test_data:
    bert_input = convert_example_to_feature(review)
    input_ids_list.append(bert_input['input_ids'])
    token_type_ids_list.append(bert_input['token_type_ids'])
    attention_mask_list.append(bert_input['attention_mask'])
    true_cls_labels.append(label)

print("first 10 true cls labels: ", true_cls_labels[:10])
test_ds = [np.array(input_ids_list), np.array(attention_mask_list), np.array(token_type_ids_list)]
predictions = cls_model.predict(test_ds).logits
label_ids = np.argmax(cls_model.predict(test_ds).logits, axis=1)
predict_cls_labels = [labels[_] for _ in label_ids]
print("first 10 predict cls labels: ", predict_cls_labels[:10])
print("result for model evaluate: \n")
print(classification_report(true_cls_labels, predict_cls_labels, digits=4))
