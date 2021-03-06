# coding=utf-8
import torch

GPU = 0
DEVICE = torch.device('cuda:%d' % GPU if torch.cuda.is_available() else 'cpu')
TRAIN_SIZE = 0.9  # 训练数据占比
DEV_SIZE = 0.05  # 验证数据占比
TEST_SIZE = 0.05  # 测试数据占比


class Config(object):
    def __init__(self):
        self.label_file = './data/tag.txt'
        self.train_file = './data/train.txt'
        self.dev_file = './data/dev.txt'
        self.test_file = './data/test.txt'
        self.vocab = './data/bert/vocab.txt'
        self.max_length = 300
        self.use_cuda = torch.cuda.is_available()
        self.gpu = GPU
        self.batch_size = 50
        self.bert_path = './data/bert'
        self.rnn_hidden = 500
        self.bert_embedding = 768
        self.dropout1 = 0.5
        self.dropout_ratio = 0.5
        self.rnn_layer = 1
        self.lr = 0.0001
        self.lr_decay = 0.00001
        self.weight_decay = 0.00005
        self.checkpoint = 'result/'
        self.optim = 'Adam'
        self.load_model = False
        self.load_path = None
        self.base_epoch = 100

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):
        return '\n'.join(['%s:%s' % item for item in self.__dict__.items()])


if __name__ == '__main__':
    con = Config()
    con.update(gpu=8)
    print(con.gpu)
    print(con)
