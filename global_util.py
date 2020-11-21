import logging
import numpy
from global_config import *

file_handler = logging.FileHandler(LOG_FILENAME, encoding='utf-8')
stream_handler = logging.StreamHandler()
logging.basicConfig(format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT, level=logging.INFO,
                    handlers=[file_handler, stream_handler])


def write_log(line, level=logging.INFO, newline=False):
    """
    写日志并输出到控制台
    :param line:
    :param level:
    :param newline:
    :return:
    """
    if newline:
        line = '\n' + str(line)

    if level == logging.CRITICAL:
        logging.critical(line)
    elif level == logging.ERROR:
        logging.error(line)
    elif level == logging.WARNING:
        logging.warning(line)
    elif level == logging.INFO:
        logging.info(line)
    elif level == logging.DEBUG:
        logging.debug(line)


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
