# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import argparse
import os
import sys

sys.path.append("../")

import pycorrector
from pycorrector.utils import eval
pwd_path = os.path.abspath(os.path.dirname(__file__))


def demo():
    idx_errors = pycorrector.detect('少先队员因该为老人让坐')
    print(idx_errors)


def main(args):
    if args.data == 'sighan_15' and args.model == 'rule':
        # demo()
        # Sentence Level: acc:0.173225, precision:0.979592, recall:0.148541, f1:0.257965, cost time:230.92 s
        eval.eval_sighan2015_by_model(pycorrector.correct)
    if args.data == 'sighan_15' and args.model == 'bert':
        # right_rate:0.37623762376237624, right_count:38, total_count:101;
        # recall_rate:0.3645833333333333, recall_right_count:35, recall_total_count:96, spend_time:503 s
        from pycorrector.bert.bert_corrector import BertCorrector
        model = BertCorrector()
        eval.eval_sighan2015_by_model(model.bert_correct)
    if args.data == 'sighan_15' and args.model == 'macbert':
        # Sentence Level: acc:0.914885, precision:0.995199, recall:0.916446, f1:0.954200, cost time:29.47 s
        from pycorrector.macbert.macbert_corrector import MacBertCorrector
        model = MacBertCorrector()
        eval.eval_sighan2015_by_model(model.macbert_correct)
    if args.data == 'sighan_15' and args.model == 'ernie':
        # right_rate:0.297029702970297, right_count:30, total_count:101;
        # recall_rate:0.28125, recall_right_count:27, recall_total_count:96, spend_time:655 s
        from pycorrector.ernie.ernie_corrector import ErnieCorrector
        model = ErnieCorrector()
        eval.eval_sighan2015_by_model(model.ernie_correct)

    if args.data == 'corpus500' and args.model == 'rule':
        demo()
        # right_rate:0.486, right_count:243, total_count:500;
        # recall_rate:0.18, recall_right_count:54, recall_total_count:300, spend_time:78 s
        eval.eval_corpus500_by_model(pycorrector.correct)
    if args.data == 'corpus500' and args.model == 'bert':
        # right_rate:0.586, right_count:293, total_count:500;
        # recall_rate:0.35, recall_right_count:105, recall_total_count:300, spend_time:1760 s
        from pycorrector.bert.bert_corrector import BertCorrector
        model = BertCorrector()
        eval.eval_corpus500_by_model(model.bert_correct)
    if args.data == 'corpus500' and args.model == 'macbert':
        # Sentence Level: acc:0.724000, precision:0.912821, recall:0.595318, f1:0.720648, cost time:6.43 s
        from pycorrector.macbert.macbert_corrector import MacBertCorrector
        model = MacBertCorrector()
        eval.eval_corpus500_by_model(model.macbert_correct)
    if args.data == 'corpus500' and args.model == 'ernie':
        # right_rate:0.598, right_count:299, total_count:500;
        # recall_rate:0.41333333333333333, recall_right_count:124, recall_total_count:300, spend_time:6960 s
        from pycorrector.ernie.ernie_corrector import ErnieCorrector
        model = ErnieCorrector()
        eval.eval_corpus500_by_model(model.ernie_correct)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='sighan_15', help='evaluate dataset, sighan_15/corpus500')
    parser.add_argument('--model', type=str, default='rule', help='which model to evaluate, rule/bert/macbert/ernie')
    args = parser.parse_args()
    main(args)
