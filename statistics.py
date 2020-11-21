import sys

sys.path.append("../..")
from global_util import *


def get_confusion_matrices(CATEGORIES, label, prediction):
    """
    根据分类数据，生成混淆矩阵列表
    :param CATEGORIES:
    :param label:
    :param prediction:
    :return:
    """
    tp_list = numpy.zeros(len(CATEGORIES), dtype=int)
    fn_list = numpy.zeros(len(CATEGORIES), dtype=int)
    fp_list = numpy.zeros(len(CATEGORIES), dtype=int)
    tn_list = numpy.zeros(len(CATEGORIES), dtype=int)

    for index in range(len(CATEGORIES)):
        tp, fn, fp, tn = get_confusion_matrix(label, prediction, index)
        tp_list[index] += tp
        fn_list[index] += fn
        fp_list[index] += fp
        tn_list[index] += tn

    return tp_list, fn_list, fp_list, tn_list


def get_confusion_matrix(label, prediction, category):
    """
    生成混淆矩阵
    :param label:
    :param prediction:
    :param category:
    :return:
    """
    tp = 0
    fn = 0
    fp = 0
    tn = 0

    for index in range(len(label)):
        if label[index] == category:
            if prediction[index] == category:
                tp += 1
            else:
                fn += 1
        else:
            if prediction[index] == category:
                fp += 1
            else:
                tn += 1

    return tp, fn, fp, tn


def get_evaluation_metrics(categories, tp_list, fn_list, fp_list, tn_list):
    """
    计算PRF
    :param categories:
    :param tp_list:
    :param fn_list:
    :param fp_list:
    :param tn_list:
    :return:
    """
    len_categories = len(categories)
    precision_list = numpy.zeros(len_categories, dtype=float)
    recall_list = numpy.zeros(len_categories, dtype=float)
    f1_list = numpy.zeros(len_categories, dtype=float)

    # PRF
    for index in range(len_categories):
        precision = tp_list[index] / (tp_list[index] + fp_list[index])
        recall = tp_list[index] / (tp_list[index] + fn_list[index])
        f1 = 2 * precision * recall / (precision + recall)

        write_log('%s: Precision: %.4f, Recall %.4f, F1 Score %.4f' % (categories[index], precision, recall, f1))

        precision_list[index] += precision
        recall_list[index] += recall
        f1_list[index] += f1

    # Macro PRF
    macro_precission = precision_list.sum() / len_categories
    macro_recall = recall_list.sum() / len_categories
    macro_f1 = f1_list.sum() / len_categories
    write_log(
        'Macro Precision: %.4f, Macro Recall %.4f, Macro F1 Score %.4f' % (macro_precission, macro_recall, macro_f1))

    # Micro PRF
    micro_precision = tp_list.sum() / (tp_list.sum() + fp_list.sum())
    micro_recall = tp_list.sum() / (tp_list.sum() + fn_list.sum())
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)

    write_log(
        'Micro Precision: %.4f, Micro Recall %.4f, Micro F1 Score %.4f' % (micro_precision, micro_recall, micro_f1))
