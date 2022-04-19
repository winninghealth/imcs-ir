## CBLUE 智能对话诊疗意图识别 IMCS-IR

### 任务背景

任务名称：智能对话诊疗数据集-对话意图识别（Intent Recognization）

任务简介：针对互联网医患在线对话问诊的记录，该任务的目标是识别出对话的意图。IMCS21数据集中标注了医患对话行为，共包含16类对话意图，标注方式采用句子级标注。任务采用Macro-F1值作为评价指标， 对于测试集中每份对话段落的每条句子，预测其对应的标签。

方案思路：https://zhuanlan.zhihu.com/p/501295857

方案结果：79.08%（Macro-F1）

相关比赛：第一届智能对话诊疗评测比赛（第二十届中国计算语言学大会 CCL2021）

比赛官网：http://www.fudan-disc.com/sharedtask/imcs21/index.html

### 数据集

IMCS21数据集由复旦大学大数据学院在复旦大学医学院专家的指导下构建。本次评测任务使用的IMCS-IR数据集在中文医疗信息处理挑战榜CBLUE持续开放下载，地址：https://tianchi.aliyun.com/dataset/dataDetail?dataId=95414

CBLUE挑战榜公开的3,052条数据包括1,824条训练数据、616条验证数据和612条测试数据。请将下载后的数据保存在文件夹`data`中。

### 环境依赖

- 主要基于 Python (3.7.3+) & AllenNLP 实现

- 实验使用 GPU ：GeForce GTX 1080Ti

- Python 版本依赖：

```
torch==1.7.1+cu101
transformers==4.4.2
allennlp==2.4.0
```

### 快速开始

#### 预训练模型

实验中选择了6种不同的开源预训练模型：

1. chinese-bert-wwm，下载地址：https://huggingface.co/hfl/chinese-bert-wwm
2. chinese-bert-wwm-ext，下载地址：https://huggingface.co/hfl/chinese-bert-wwm-ext
3. chinese-macbert-base，下载地址：https://huggingface.co/hfl/chinese-macbert-base
4. chinese-roberta-wwm-ext，下载地址：https://huggingface.co/hfl/chinese-roberta-wwm-ext
5. chinese-pert-base，下载地址：https://huggingface.co/hfl/chinese-pert-base
6. PCL-MedBERT，下载地址：https://code.ihub.org.cn/projects/1775

请将下载后的模型权重`pytorch_model.bin`保存在`plms`路径下相应名称的模型文件夹中。

#### 模型训练

```python
python trainer.py --train_file ./data/IMCS_train.json --dev_file ./data/IMCS_dev.json --pretrained_model_dir ./plms/chinese_bert_wwm --output_model_dir ./save_model/chinese_bert_wwm --cuda_id cuda:0 --batch_size 1 --num_epochs 10 --patience 3
```

- 参数：{train_file}: 训练数据集路径，{dev_file}: 验证数据集路径，{pretrained_model_dir}: 预训练语言模型路径，{output_model_dir}: 模型保存路径

#### 模型预测

```python
python predict.py --test_input_file ./data/IMCS_test.json --test_output_file IMCS-IR_test.json --model_dir ./save_model/chinese_bert_wwm --pretrained_model_dir ./plms/chinese_bert_wwm --cuda_id cuda:0
```

- 参数：{test_input_file}: 测试数据集路径，{test_output_file}: 预测结果输出路径，{model_dir}: 加载已训练模型的路径，{pretrained_model_dir}: 预训练语言模型的路径

### 如何引用

```
@Misc{Jiang2022Shared,
      author={Yiwen Jiang},
      title={Solutions of Intent Recognization Task within Online Medical Dialogues},
      year={2022},
      howpublished={GitHub},
      url={https://github.com/winninghealth/imcs-ir},
}
```

### 版权

MIT License - 详见 [LICENSE](LICENSE)

