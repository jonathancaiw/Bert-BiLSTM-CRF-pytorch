from sklearn.model_selection import train_test_split
from tqdm import tqdm
from config import *

conf = Config()


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


def distinct_dataset(x, y):
    """
    数据集去重
    :param x:
    :param y:
    :return:
    """
    distinct_x = []
    distinct_y = []
    s = set()

    for index in range(len(y)):
        str_x = ''.join(x[index])
        if str_x not in s:
            s.add(str_x)
            distinct_x.append(x[index])
            distinct_y.append(y[index])

    return distinct_x, distinct_y


def trim_dataset(x, y, len_limit):
    """
    过滤掉超长数据
    :param x:
    :param y:
    :param len_limit:
    :return:
    """
    if not len_limit > 0:
        return x, y

    filtered_x = []
    filtered_y = []

    for index in range(len(y)):
        if len(y[index]) <= len_limit:
            filtered_x.append(x[index])
            filtered_y.append(y[index])

    return filtered_x, filtered_y


def generate_dataset(filename):
    x, y = convert(filename)

    x, y = distinct_dataset(x, y)

    # 最大有效长度=最大长度-开始位-结束位
    x, y = trim_dataset(x, y, conf.max_length - 2)

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
