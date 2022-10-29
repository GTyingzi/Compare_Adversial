# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, TextRNN, TextRCNN, TextRNN_Att, DPCNN, Transformer')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
parser.add_argument('--attack_train', type=str, required=False,default='', help='choose a mode:pgd,FreeAT,fgsm,fgm')
args = parser.parse_args()

if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集

    # 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
    embedding = 'embedding_SougouNews.npz'
    if args.embedding == 'random':
        embedding = 'random'
    model_name = args.model  # 'TextRCNN'  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer

    x = import_module('models.' + model_name)
    config = x.Config('dataset/' + dataset, embedding)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    vocab, train_data, dev_data, test_data = build_dataset(config, args.word)
    train_iter = build_iterator(train_data, config) # 训练数据集迭代器
    dev_iter = build_iterator(dev_data, config) # 验证数据集迭代器
    test_iter = build_iterator(test_data, config) # 测试数据集迭代器
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    config.n_vocab = len(vocab)  # 词表大小赋予模型的config
    config.attack_train = args.attack_train
    model = x.Model(config).to(config.device)  # 训练模型
    if model_name != 'Transformer':
        init_network(model) # 初始化网络
    print(model.parameters) # 打印模型参数
    train(config, model, train_iter, dev_iter, test_iter) # 训练模型（模型配置config，模型，训练数据集迭代器，验证数据集迭代器，测试数据集迭代器）
