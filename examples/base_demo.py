# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import sys

sys.path.append("..")


import pycorrector

if __name__ == '__main__':
    corrected_sent, detail = pycorrector.correct('一只小鱼船浮在平净的河面上。')
    #corrected_sent, detail = pycorrector.correct('希望你们好好的跳无。')
    #corrected_sent, detail = pycorrector.correct('我跟我朋唷打算去法国玩儿。')
    # corrected_sent, detail = pycorrector.correct('家里的歪发信号不好')
    #corrected_sent, detail = pycorrector.correct('下面请消峰分享项目经验')
    #corrected_sent, detail = pycorrector.correct('视讯打屏计费确认页分省贯控需求')
    # corrected_sent, detail = pycorrector.correct('一只小鱼船浮在平静的河面上')
    #corrected_sent, detail = pycorrector.correct('少现队员音该为闹人让桌。希望你们好好的跳无')
    #corrected_sent, detail = pycorrector.correct('机七学习是人工智能领遇最能体现智能的一个分知')
    print(corrected_sent, detail)

    error_sentences = [
        '真麻烦你了。希望你们好好的跳无',
        '机七学习是人工智能领遇最能体现智能的一个分知',
        '一只小鱼船浮在平净的河面上',
        '我的家乡是有明的渔米之乡',
    ]
    for line in error_sentences:
        correct_sent, err = pycorrector.correct(line)
        print("{} => {} {}".format(line, correct_sent, err))