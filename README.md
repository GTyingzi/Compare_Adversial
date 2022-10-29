# Compare_Adversial
在中文文本分类常场景下，用TextCNN、TextRNN为基准模型，综合对比不同对抗训练方法（FGSM、FGM、PGD、FreeAT）的效果，基于PyTorch实现

## 中文数据集

## 运行环境
- python 3.7
- pytorch 1.1
- tqdm
- sklearn
- tensorboardX

## 实验参数
- 预训练词向量：搜狗新闻
- batch_size 128
- 训练集：18w
- 验证集：1w
- 测试集：1w

## 效果

### TextCNN + attack_train

| baseline+attack_train | precision | recall | F1     |
| --------------------- | --------- | ------ | ------ |
| TextCNN               | 0.9083    | 0.9078 | 0.9079 |
| TextCNN + FGSM        | 0.9105    | 0.9103 | 0.9103 |
| TextCNN + FGM         | 0.9110    | 0.9104 | 0.9105 |
| TextCNN + PGD         | 0.9103    | 0.9098 | 0.9099 |
| TextCNN + FreeAT      | 0.9104    | 0.9097 | 0.9096 |



### TextRNN + attack_train

| baseline+attack_train | precision | recall | F1     |
| --------------------- | --------- | ------ | ------ |
| TextRNN               | 0.9046    | 0.9034 | 0.9038 |
| TextRNN + FGSM        | 0.9068    | 0.9055 | 0.9058 |
| TextRNN + FGM         | 0.9160    | 0.9161 | 0.9160 |
| TextRNN + PGD         | 0.9144    | 0.9142 | 0.9140 |
| TextRNN + FreeAT      | 0.9064    | 0.9062 | 0.9059 |

## 使用说明

```
# 训练并测试
# TextCNN
python run.py --model TextCNN

# TextCNN + FGSM
python run.py --model TextCNN --attack_train fgsm

# TextCNN + FGM
python run.py --model TextCNN --attack_train fgm

# TextCNN + PGD
python run.py --model TextCNN --attack_train pgd

# TextCNN + FreeAT
python run.py --model TextCNN --attack_train FreeAT

-------------------------------------------------------------
# TextRNN
python run.py --model TextRNN

# TextRNN + FGSM
python run.py --model TextRNN --attack_train fgsm

# TextRNN + FGM
python run.py --model TextRNN --attack_train fgm

# TextRNN + PGD
python run.py --model TextRNN --attack_train pgd

# TextRNN + FreeAT
python run.py --model TextRNN --attack_train FreeAT
```
