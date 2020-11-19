from sklearn.model_selection import train_test_split
from tqdm import tqdm
from config import *


def convert(filename, skip_o=True):
    x = []
    y = []
    chars, labels = init_x_y()

    with open(filename, 'r', encoding='UTF-8') as f:
        for line in tqdm(f.readlines()):
            if line == '\n':
                # 换行
                append_x_y(x, y, chars, labels, skip_o)
                chars, labels = init_x_y()
            else:
                chars.append(line[0])
                labels.append(line[2:-1])

        append_x_y(x, y, chars, labels, skip_o)

    return x, y


def init_x_y():
    return [], []


def append_x_y(x, y, chars, labels, skip_o=True):
    if len(chars) > 0:
        if skip_o:
            s = set(labels)
            if len(s) == 1 and s.pop() == 'O':
                return

        x.append(chars)
        y.append(labels)


def generate_dataset(filename):
    x, y = convert(filename)

    train_x, left_x, train_y, left_y = get_train_test_dataset(x, y,
                                                              (DEV_SIZE + TEST_SIZE) / (TRAIN_SIZE + DEV_SIZE + TEST_SIZE))
    dev_x, test_x, dev_y, test_y = get_train_test_dataset(left_x, left_y, TEST_SIZE / (DEV_SIZE + TEST_SIZE))

    torch.save((train_x, train_y), 'data/train_c.pt')
    torch.save((dev_x, dev_y), 'data/dev_c.pt')
    torch.save((test_x, test_y), 'data/test_c.pt')


def get_train_test_dataset(x, y, test_size, shuffle=True):
    """拆分训练集与测试集"""
    return train_test_split(x, y, test_size=test_size, shuffle=shuffle)


if __name__ == '__main__':
    generate_dataset('data/label.txt')
