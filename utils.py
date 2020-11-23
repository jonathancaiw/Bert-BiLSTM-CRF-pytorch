# coding=utf-8
import datetime
import os
import sys
from data import *

sys.path.append("../..")
from global_util import *


class InputFeatures(object):
    def __init__(self, input_id, label_id, input_mask):
        self.input_id = input_id
        self.label_id = label_id
        self.input_mask = input_mask


def load_vocab(vocab_file, user_define=False):
    """Loads a vocabulary file into a dictionary."""
    if user_define:
        vocab_file = './data/tag_c.txt'
    vocab = {}
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


def read_corpus(path, max_length, label_dic, vocab, user_define=False):
    """
    :param path:数据文件路径
    :param max_length: 最大长度
    :param label_dic: 标签字典
    :return:
    """
    if user_define:
        return read_my_corpus(path, max_length, label_dic, vocab)

    file = open(path, encoding='utf-8')
    content = file.readlines()
    file.close()
    result = []
    for line in content:
        text, label = line.strip().split('|||')
        tokens = text.split()
        label = label.split()
        if len(tokens) > max_length - 2:
            tokens = tokens[0:(max_length - 2)]
            label = label[0:(max_length - 2)]
        tokens_f = ['[CLS]'] + tokens + ['[SEP]']
        label_f = ["<start>"] + label + ['<eos>']
        input_ids = [int(vocab[i]) if i in vocab else int(vocab['[UNK]']) for i in tokens_f]
        label_ids = [label_dic[i] for i in label_f]
        input_mask = [1] * len(input_ids)
        while len(input_ids) < max_length:
            input_ids.append(0)
            input_mask.append(0)
            label_ids.append(label_dic['<pad>'])
        assert len(input_ids) == max_length
        assert len(input_mask) == max_length
        assert len(label_ids) == max_length
        feature = InputFeatures(input_id=input_ids, input_mask=input_mask, label_id=label_ids)
        result.append(feature)
    return result


def read_my_corpus(path, max_length, label_dic, vocab):
    paths = path.split('/')
    filenames = paths[-1].split('.')
    paths[-1] = filenames[0] + '_c.pt'
    filename = '/'.join(paths)
    x, y = torch.load(filename)

    result = []
    for i in range(len(x)):
        tokens, label = x[i], y[i]
        if len(tokens) > max_length - 2:
            tokens = tokens[0:(max_length - 2)]
            label = label[0:(max_length - 2)]
        tokens_f = ['[CLS]'] + tokens + ['[SEP]']
        label_f = ["<start>"] + label + ['<eos>']
        input_ids = [int(vocab[i]) if i in vocab else int(vocab['[UNK]']) for i in tokens_f]
        label_ids = [label_dic[i] for i in label_f]
        input_mask = [1] * len(input_ids)
        while len(input_ids) < max_length:
            input_ids.append(0)
            input_mask.append(0)
            label_ids.append(label_dic['<pad>'])
        assert len(input_ids) == max_length
        assert len(input_mask) == max_length
        assert len(label_ids) == max_length
        feature = InputFeatures(input_id=input_ids, input_mask=input_mask, label_id=label_ids)
        result.append(feature)
    return result


def save_model(model, epoch, path='result', **kwargs):
    """
    默认保留所有模型
    :param model: 模型
    :param path: 保存路径
    :param loss: 校验损失
    :param last_loss: 最佳epoch损失
    :param kwargs: every_epoch or best_epoch
    :return:
    """
    if not os.path.exists(path):
        os.mkdir(path)
    if kwargs.get('name', None) is None:
        cur_time = datetime.datetime.now().strftime('%Y-%m-%d#%H:%M:%S')
        name = cur_time + '--epoch:{}'.format(epoch)
        full_name = os.path.join(path, name)
        torch.save(model.state_dict(), full_name)
        write_log('Saved model at epoch {} successfully {} '.format(epoch, full_name))
        with open('{}/checkpoint'.format(path), 'w') as file:
            file.write(name)
            print('Write to checkpoint')


def load_model(model, path='result', **kwargs):
    if kwargs.get('name', None) is None:
        with open('{}/checkpoint'.format(path)) as file:
            content = file.read().strip()
            name = os.path.join(path, content)
    else:
        name = kwargs['name']
        name = os.path.join(path, name)
    model.load_state_dict(torch.load(name, map_location=lambda storage, loc: storage))
    print('load model {} successfully'.format(name))
    return model
