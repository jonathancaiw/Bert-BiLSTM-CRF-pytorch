# coding=utf-8
from torch.autograd import Variable
from tqdm import tqdm
from config import *
from model import BERT_LSTM_CRF
import torch.optim as optim
from utils import load_vocab, read_corpus, load_model, save_model
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from statistics import *

USER_DEFINE = False
PRINT_COUNT = 5


def train(**kwargs):
    config = Config()
    config.update(**kwargs)
    print('current config:\n', config)
    if config.use_cuda:
        torch.cuda.set_device(config.gpu)
    print('loading corpus')
    vocab = load_vocab(config.vocab)
    label_dic = load_vocab(config.label_file, user_define=USER_DEFINE)
    tagset_size = len(label_dic)
    train_data = read_corpus(config.train_file, max_length=config.max_length, label_dic=label_dic, vocab=vocab, user_define=USER_DEFINE)
    dev_data = read_corpus(config.dev_file, max_length=config.max_length, label_dic=label_dic, vocab=vocab, user_define=USER_DEFINE)

    train_ids = torch.LongTensor([temp.input_id for temp in train_data])
    train_masks = torch.LongTensor([temp.input_mask for temp in train_data])
    train_tags = torch.LongTensor([temp.label_id for temp in train_data])

    train_dataset = TensorDataset(train_ids, train_masks, train_tags)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size)

    dev_ids = torch.LongTensor([temp.input_id for temp in dev_data])
    dev_masks = torch.LongTensor([temp.input_mask for temp in dev_data])
    dev_tags = torch.LongTensor([temp.label_id for temp in dev_data])

    dev_dataset = TensorDataset(dev_ids, dev_masks, dev_tags)
    dev_loader = DataLoader(dev_dataset, shuffle=True, batch_size=config.batch_size)
    model = BERT_LSTM_CRF(config.bert_path, tagset_size, config.bert_embedding, config.rnn_hidden, config.rnn_layer, dropout_ratio=config.dropout_ratio,
                          dropout1=config.dropout1, use_cuda=config.use_cuda)
    if config.load_model:
        assert config.load_path is not None
        model = load_model(model, name=config.load_path)
    if config.use_cuda:
        model.cuda()
    model.train()
    optimizer = getattr(optim, config.optim)
    optimizer = optimizer(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    eval_loss = 10000
    for epoch in range(config.base_epoch):
        step = 0
        for i, batch in enumerate(tqdm(train_loader)):
            step += 1
            model.zero_grad()
            inputs, masks, tags = batch
            inputs, masks, tags = Variable(inputs), Variable(masks), Variable(tags)
            if config.use_cuda:
                inputs, masks, tags = inputs.cuda(), masks.cuda(), tags.cuda()
            feats = model(inputs, masks)
            loss = model.loss(feats, masks, tags)
            loss.backward()
            optimizer.step()
            if step % 50 == 0:
                print('step: {} |  epoch: {}|  loss: {}'.format(step, epoch, loss.item()))
        loss_temp = dev(model, dev_loader, epoch, config)
        if loss_temp < eval_loss:
            eval_loss = loss_temp
            save_model(model, epoch)


def dev(model, dev_loader, epoch, config):
    model.eval()
    eval_loss = 0
    true = []
    pred = []
    length = 0
    for i, batch in enumerate(tqdm(dev_loader)):
        inputs, masks, tags = batch
        length += inputs.size(0)
        inputs, masks, tags = Variable(inputs), Variable(masks), Variable(tags)
        if config.use_cuda:
            inputs, masks, tags = inputs.cuda(), masks.cuda(), tags.cuda()
        feats = model(inputs, masks)
        path_score, best_path = model.crf(feats, masks.byte())
        loss = model.loss(feats, masks, tags)
        eval_loss += loss.item()
        pred.extend([t for t in best_path])
        true.extend([t for t in tags])
    write_log('eval  epoch: {}|  loss: {}'.format(epoch, eval_loss / length))
    model.train()
    return eval_loss


def test():
    config = Config()
    vocab = load_vocab(config.vocab)
    label_dic = load_vocab(config.label_file, user_define=USER_DEFINE)
    tagset_size = len(label_dic)
    test_data = read_corpus(config.test_file, max_length=config.max_length, label_dic=label_dic, vocab=vocab, user_define=USER_DEFINE)

    test_ids = torch.LongTensor([temp.input_id for temp in test_data])
    test_masks = torch.LongTensor([temp.input_mask for temp in test_data])
    test_tags = torch.LongTensor([temp.label_id for temp in test_data])

    test_dataset = TensorDataset(test_ids, test_masks, test_tags)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)
    model = BERT_LSTM_CRF(config.bert_path, tagset_size, config.bert_embedding, config.rnn_hidden, config.rnn_layer, dropout_ratio=config.dropout_ratio,
                          dropout1=config.dropout1, use_cuda=config.use_cuda)

    model = load_model(model, name=config.load_path)
    model.to(DEVICE)

    model.eval()
    eval_loss = 0
    true = []
    pred = []
    length = 0
    for i, batch in enumerate(tqdm(test_loader)):
        inputs, masks, tags = batch
        length += inputs.size(0)
        inputs, masks, tags = Variable(inputs), Variable(masks), Variable(tags)
        if config.use_cuda:
            inputs, masks, tags = inputs.cuda(), masks.cuda(), tags.cuda()
        feats = model(inputs, masks)
        path_score, best_path = model.crf(feats, masks.byte())
        loss = model.loss(feats, masks, tags)
        eval_loss += loss.item()
        pred.extend([t for t in best_path])
        true.extend([t for t in tags])
    write_log('test loss: {}'.format(eval_loss / length))

    print_statistics(label_dic, test_data, true, pred)


def print_statistics(label_dic, test_data, true, pred):
    tag_to_label = {value: key for key, value in label_dic.items()}

    mismatch = 0
    total = 0
    len_categories = len(tag_to_label)

    tp_list = numpy.zeros(len_categories, dtype=int)
    fn_list = numpy.zeros(len_categories, dtype=int)
    fp_list = numpy.zeros(len_categories, dtype=int)
    tn_list = numpy.zeros(len_categories, dtype=int)

    for i in tqdm(range(len(test_data))):
        for j in range(len(test_data[i].input_id)):
            if test_data[i].input_id[j] == 0:
                break

        p = pred[i][:j]
        y = true[i][:j]

        total += 1

        tp, fn, fp, tn = get_confusion_matrices(tag_to_label, y, p)
        tp_list += tp
        fn_list += fn
        fp_list += fp
        tn_list += tn

        for j in range(len(p)):
            if not p[j] == y[j]:
                mismatch += 1
                break

        if total < PRINT_COUNT:
            true_labels = [tag_to_label[x.item()] for x in y]
            pred_labels = [tag_to_label[x.item()] for x in p]
            write_log('#%d true: %s' % (i, true_labels))
            write_log('#%d pred: %s' % (i, pred_labels))

    accuracy = (total - mismatch) / total
    write_log('Accuracy: match %d, mismatch %d, total %d, accuracy %.4f' % (
        total - mismatch, mismatch, total, accuracy))

    get_evaluation_metrics(tag_to_label, tp_list, fn_list, fp_list, tn_list)


if __name__ == '__main__':
    train()
    # test()
