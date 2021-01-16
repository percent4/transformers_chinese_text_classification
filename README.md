本项目采用transformers模块，使用bert-base-chinese模型实现文本多分类。

### 维护者

- jclian91

### 数据集

- sougou小分类数据集

sougou小分类数据集，共有5个类别，分别为体育、健康、军事、教育、汽车。

划分为训练集和测试集，其中训练集每个分类800条样本，测试集每个分类100条样本。

- THUCNews数据集

使用THUCNews数据集进行训练与测试，10个分类，每个分类6500条数据。
类别如下：
体育, 财经, 房产, 家居, 教育, 科技, 时尚, 时政, 游戏, 娱乐
数据集划分如下：
训练集: 5000 * 10
测试集: 1000 * 10

### 代码结构

运行环境：

- Python: 3.7.9
- Cuda: 10.2
- Cudnn: 7.6.5
- Tensorflow: 2.3.2

其余第三方模块见requirements.txt.

代码结构如下：

```
.
├── bert-base-chinese（transformers提供的BERT中文预训练模型）
│   ├── config.json
│   ├── tf_model.h5
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   └── vocab.txt
├── data（数据集）
│   └── sougou_mini
│       ├── test.csv
│       └── train.csv
├── model_predict.py（模型预测脚本）
├── model_train.py（模型训练脚本）
├── model_evaluate.py（模型评估脚本）
├── sougou_mini_cls.h5（保存模型）
├── sougou_mini_label.json（数据集分类标签）
└── util.py（配置脚本）
```

### 模型结构

- 预训练模型：bert-base-chinese
- 多分类模型：TFBertForSequenceClassification

### 模型评估

- sougou小分类数据集

模型参数：MAX_SEQ_LENGTH=300, BATCH_SIZE=16, EPOCH=5

```
              precision    recall  f1-score   support

          体育     1.0000    1.0000    1.0000        99
          健康     0.9592    0.9495    0.9543        99
          军事     0.9802    1.0000    0.9900        99
          教育     0.9400    0.9495    0.9447        99
          汽车     0.9794    0.9596    0.9694        99

    accuracy                         0.9717       495
   macro avg     0.9718    0.9717    0.9717       495
weighted avg     0.9718    0.9717    0.9717       495
```
- THUCNews数据集

模型参数：MAX_SEQ_LENGTH=300, BATCH_SIZE=16, EPOCH=3

```
              precision    recall  f1-score   support

          体育     1.0000    0.9940    0.9970      1000
          娱乐     0.9871    0.9910    0.9890      1000
          家居     0.9780    0.8890    0.9314      1000
          房产     0.9180    0.9070    0.9125      1000
          教育     0.9896    0.9490    0.9689      1000
          时尚     0.9550    0.9970    0.9755      1000
          时政     0.9514    0.9780    0.9645      1000
          游戏     0.9960    0.9890    0.9925      1000
          科技     0.9726    0.9950    0.9837      1000
          财经     0.9405    0.9950    0.9670      1000

    accuracy                         0.9684     10000
   macro avg     0.9688    0.9684    0.9682     10000
weighted avg     0.9688    0.9684    0.9682     10000
```

### 项目启动

1. 将BERT中文预训练模型tf_model.h5放在bert-base-chinese文件夹下
2. 所需Python第三方模块参考requirements.txt文档
3. 自己需要分类的数据按照data/sougou_mini的格式准备好
4. 调整模型参数，运行model_train.py进行模型训练
5. 运行model_evaluate.py进行模型评估
6. 运行model_predict.py对新文本进行评估

### 参考网址

1. Text classification with transformers in Tensorflow 2: BERT, XLNet: https://atheros.ai/blog/text-classification-with-transformers-in-tensorflow-2
2. tensorflow 2.0+ 基于BERT模型的文本分类: https://zhuanlan.zhihu.com/p/199238483?utm_source=wechat_session
3. HUGGING FACE: https://huggingface.co/models