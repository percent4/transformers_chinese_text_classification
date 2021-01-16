# -*- coding: utf-8 -*-
# @Time : 2021/1/15 15:00
# @Author : Jclian91
# @File : model_train.py
# @Place : Yangpu, Shanghai
import json
import pandas as pd
import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer

from util import MAX_SEQ_LENGTH, EPOCH, BATCH_SIZE
from util import dataset, train_file_path, test_file_path

# Load the fast tokenizer from saved file
tokenizer = BertTokenizer("bert-base-chinese/vocab.txt", lowercase=True)


# combine step for tokenization, WordPiece vector mapping, adding special
# tokens as well as truncating reviews longer than the max length
def convert_example_to_feature(context):
    return tokenizer.encode_plus(context,
                                 add_special_tokens=True,  # add [CLS], [SEP]
                                 max_length=MAX_SEQ_LENGTH,  # max length of the text that can go to BERT
                                 pad_to_max_length=True,  # add [PAD] tokens
                                 return_attention_mask=True,  # add attention mask to not focus on pad tokens
                                 )


# map to the expected input to TFBertForSequenceClassification, see here
def map_example_to_dict(input_ids, attention_masks, token_type_ids, data_label):
    return {
               "input_ids": input_ids,
               "token_type_ids": token_type_ids,
               "attention_mask": attention_masks,
           }, data_label


# prepare list, so that we can build up final TensorFlow dataset from slices.
def encode_examples(train_sample):
    input_ids_list = []
    token_type_ids_list = []
    attention_mask_list = []
    label_list = []

    for review, label in train_sample:
        bert_input = convert_example_to_feature(review)
        input_ids_list.append(bert_input['input_ids'])
        token_type_ids_list.append(bert_input['token_type_ids'])
        attention_mask_list.append(bert_input['attention_mask'])
        label_list.append([label])

    return tf.data.Dataset.from_tensor_slices(
        (input_ids_list, attention_mask_list, token_type_ids_list, label_list)).map(map_example_to_dict)


if __name__ == '__main__':
    # read data
    print("begin data processing...")
    train_df = pd.read_csv(train_file_path).fillna(value="")
    test_df = pd.read_csv(test_file_path).fillna(value="")

    labels = list(train_df["label"].unique())
    with open("{}_label.json".format(dataset), "w", encoding="utf-8") as f:
        f.write(json.dumps(labels, ensure_ascii=False, indent=2))

    train_data = []
    test_data = []
    for i in range(train_df.shape[0]):
        label, content = train_df.iloc[i, :]
        label_id = labels.index(label)
        train_data.append((content, label_id))

    for i in range(test_df.shape[0]):
        label, content = test_df.iloc[i, :]
        label_id = labels.index(label)
        test_data.append((content, label_id))

    print("finish data processing!")

    # review first 3 samples
    print("\nreview first 3 samples of train data...\n")
    for _ in train_data[:3]:
        print("label: {}, content: {}".format(_[1], _[0]))
    print("\nreview first 3 samples of test data...\n")
    for _ in test_data[:3]:
        print("label: {}, content: {}".format(_[1], _[0]))

    # tokenize
    ds_train = encode_examples(train_data).shuffle(50000).batch(BATCH_SIZE)
    ds_test = encode_examples(test_data).batch(BATCH_SIZE)
    # model initialization
    model = TFBertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=len(labels))
    # optimizer Adam
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-08)
    # we do not have one-hot vectors, we can use sparse categorical cross entropy and accuracy
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    model.summary()

    # train model and save model
    model.fit(x=ds_train,
              validation_data=ds_test,
              epochs=EPOCH,
              verbose=1)

    model.save_weights("{}_cls.h5".format(dataset))
    print("model saved!")
