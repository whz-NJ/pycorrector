# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 汉字处理的工具:判断unicode是否是汉字，数字，英文，或者其他字符。以及全角符号转半角符号。
"""

import re

import six
from pypinyin import pinyin, Style,lazy_pinyin
from pycorrector.utils.langconv import Converter
import Levenshtein
# https://blog.csdn.net/u012762625/article/details/43371625
lis_dp = [0 for i in range(100)] # LIS算法中，dp[i]表示输入序列中，长度为i的上升（序列值顺序增加）子序列中最小末尾数
lis_pos = [0 for i in range(100)] #LIS算法中，pos[i]记录输入序列第i个值在dp数组中的下标
def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def is_chinese(uchar):
    """判断一个unicode是否是汉字"""
    return '\u4e00' <= uchar <= '\u9fa5'


def is_chinese_string(string):
    """判断是否全为汉字"""
    return all(is_chinese(c) for c in string)


def is_number(uchar):
    """判断一个unicode是否是数字"""
    return '\u0030' <= uchar <= '\u0039'


def is_alphabet(uchar):
    """判断一个unicode是否是英文字母"""
    return '\u0041' <= uchar <= '\u005a' or '\u0061' <= uchar <= '\u007a'


def is_alphabet_string(string):
    """判断是否全部为英文字母"""
    return all(is_alphabet(c) for c in string)


def is_other(uchar):
    """判断是否非汉字，数字和英文字符"""
    return not (is_chinese(uchar) or is_number(uchar) or is_alphabet(uchar))


def B2Q(uchar):
    """半角转全角"""
    inside_code = ord(uchar)
    if inside_code < 0x0020 or inside_code > 0x7e:  # 不是半角字符就返回原来的字符
        return uchar
    if inside_code == 0x0020:  # 除了空格其他的全角半角的公式为:半角=全角-0xfee0
        inside_code = 0x3000
    else:
        inside_code += 0xfee0
    return chr(inside_code)


def Q2B(uchar):
    """全角转半角"""
    inside_code = ord(uchar)
    if inside_code == 0x3000:
        inside_code = 0x0020
    else:
        inside_code -= 0xfee0
    if inside_code < 0x0020 or inside_code > 0x7e:  # 转完之后不是半角字符返回原来的字符
        return uchar
    return chr(inside_code)


def stringQ2B(ustring):
    """把字符串全角转半角"""
    return "".join([Q2B(uchar) for uchar in ustring])


def uniform(ustring):
    """格式化字符串，完成全角转半角，大写转小写的工作"""
    return stringQ2B(ustring).lower()


def remove_punctuation(strs):
    """
    去除标点符号
    :param strs:
    :return:
    """
    return re.sub("[\s+\.\!\/<>“”,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", strs.strip())


def traditional2simplified(sentence):
    """
    将sentence中的繁体字转为简体字
    :param sentence: 待转换的句子
    :return: 将句子中繁体字转换为简体字之后的句子
    """
    return Converter('zh-hans').convert(sentence)


def simplified2traditional(sentence):
    """
    将sentence中的简体字转为繁体字
    :param sentence: 待转换的句子
    :return: 将句子中简体字转换为繁体字之后的句子
    """
    return Converter('zh-hant').convert(sentence)


def get_homophones_by_char(input_char):
    """
    根据汉字取同音字
    :param input_char:
    :return:
    """
    result = []
    # CJK统一汉字区的范围是0x4E00-0x9FA5,也就是我们经常提到的20902个汉字
    for i in range(0x4e00, 0x9fa6):
        if pinyin([chr(i)], style=pypinyin.NORMAL)[0][0] == pinyin(input_char, style=pypinyin.NORMAL)[0][0]:
            result.append(chr(i))
    return result


def get_homophones_by_pinyin(input_pinyin):
    """
    根据拼音取同音字
    :param input_pinyin:
    :return:
    """
    result = []
    # CJK统一汉字区的范围是0x4E00-0x9FA5,也就是我们经常提到的20902个汉字
    for i in range(0x4e00, 0x9fa6):
        if pinyin([chr(i)], style=pypinyin.TONE2)[0][0] == input_pinyin:
            # TONE2: 中zho1ng
            result.append(chr(i))
    return result

def _get_update_pinyin_similarity(pinyin_similarity_map, pinyin1, pinyin2):
    similarity_map = pinyin_similarity_map.get(pinyin1, None)
    if similarity_map is not None:
        similarity = similarity_map.get(pinyin2, None)
        if similarity is not None:
            return similarity
        # pinyin1 是老的，pinyin2 是新增的
        distance = Levenshtein.distance(pinyin1, pinyin2)
        result = 1.0 - distance/max(len(pinyin1), len(pinyin2))
        pinyin_similarity_map[pinyin1][pinyin2] = result
        pinyin_similarity_map[pinyin2] = {}
        pinyin_similarity_map[pinyin2][pinyin2] = 1.0
        pinyin_similarity_map[pinyin2][pinyin1] = result
        for py in pinyin_similarity_map: #计算pinyin2和其他汉字拼音的相似度
            if py != pinyin1 and py != pinyin2:
                distance = Levenshtein.distance(py, pinyin2)
                similarity = 1.0 - distance/max(len(py), len(pinyin2))
                pinyin_similarity_map[py][pinyin2] = similarity
                pinyin_similarity_map[pinyin2][py] = similarity
    else:
        similarity_map = pinyin_similarity_map.get(pinyin2, None)
        if similarity_map is not None: # pinyin1 是新增的，pinyin2 是老的
            distance = Levenshtein.distance(pinyin1, pinyin2)
            result = 1.0 - distance / max(len(pinyin1), len(pinyin2))
            pinyin_similarity_map[pinyin2][pinyin1] = result
            pinyin_similarity_map[pinyin1] = {}
            pinyin_similarity_map[pinyin1][pinyin1] = 1.0
            pinyin_similarity_map[pinyin1][pinyin2] = result
            for py in pinyin_similarity_map: #计算pinyin1和其他汉字拼音的相似度
                if py != pinyin1 and py != pinyin2:
                    distance = Levenshtein.distance(py, pinyin1)
                    similarity = 1.0 - distance / max(len(py), len(pinyin2))
                    pinyin_similarity_map[py][pinyin1] = similarity
                    pinyin_similarity_map[pinyin1][py] = similarity
        else: # pinyin1 和 pinyin2 都是新增的
            distance = Levenshtein.distance(pinyin1, pinyin2)
            result = 1.0 - distance / max(len(pinyin1), len(pinyin2))
            pinyin_similarity_map[pinyin1] = {}
            pinyin_similarity_map[pinyin1][pinyin1] = 1.0
            pinyin_similarity_map[pinyin1][pinyin2] = result
            pinyin_similarity_map[pinyin2] = {}
            pinyin_similarity_map[pinyin2][pinyin2] = 1.0
            pinyin_similarity_map[pinyin2][pinyin1] = result
            for py1 in [pinyin1, pinyin2]: #计算其他汉字拼音和pinyin1/pinyin2的相似度
                for py2 in pinyin_similarity_map:
                    if py2 != pinyin1 and py2 != pinyin2:
                        distance = Levenshtein.distance(py1, py2)
                        similarity = 1.0 - distance / max(len(py1), len(py2))
                        pinyin_similarity_map[py1][py2] = similarity
                        pinyin_similarity_map[py2][py1] = similarity
    return result

def bin_search(dp, dp_len, target):
    #对dp数组[0, dp_len) 进行二分查找
    #返回数组元素需要插入的位置
    left,right = 0, dp_len - 1
    while left <= right:
        mid = (left + right)//2
        if dp[mid] > target:
            right = mid - 1
        elif dp[mid] < target:
            left = mid + 1
        else: #找到该元素，直接返回
            return mid
    return left #dp数组不存在该元素，返回该元素应该插入的位置


# 获取LCS串(longest common subsquence)，转换为LIS，通过二分法查找
def lcs(sentence_pinyin_map, words_pinyin, threshold=0.7):
    if len(words_pinyin) > 100:
        print("too long words")
        return None

    max_unmatched_cnt = int(len(words_pinyin)) * (1-threshold)
    min_matched_score = int(len(words_pinyin)) * threshold
    unmatched_cnt = 0
    gross_matched_score = 0
    pos_similarities_list = []
    for word_py in words_pinyin:
        pos_similarity_list = sentence_pinyin_map.get(word_py, None)
        if pos_similarity_list is None:
            unmatched_cnt += 1
            # 先粗略过滤掉不可能匹配的热词
            if unmatched_cnt > max_unmatched_cnt:
                return None
        else:
            max_score = 0
            for pos_similarity in pos_similarity_list: #一个拼音只会在一个位置出现，算最大匹配值
                if max_score < pos_similarity[1]:
                    max_score = pos_similarity[1]
            gross_matched_score += max_score
            pos_similarities_list.extend(pos_similarity_list)
    if len(pos_similarities_list) == 1:
        return {'maxMatchedLen': 1, 'matchedScore': pos_similarities_list[0][1],
                'range': [pos_similarities_list[0][0], pos_similarities_list[0][0]]}
    if gross_matched_score < min_matched_score:
        return None

    lis_dp[0] = pos_similarities_list[0][0]
    lis_pos[0] = 0
    lis_len = 1
    for idx in range(1, len(pos_similarities_list)):
        # 如果大于dp中最大的元素，则直接插入到dp数组末尾
        if lis_dp[lis_len - 1] < pos_similarities_list[idx][0]:
            lis_dp[lis_len] = pos_similarities_list[idx][0]
            lis_pos[idx] = lis_len
            lis_len += 1
        else:
            insert_pos = bin_search(lis_dp, lis_len, pos_similarities_list[idx][0])
            lis_dp[insert_pos] = pos_similarities_list[idx][0]
            lis_pos[idx] = insert_pos
    if lis_len < min_matched_score:
        return None

    stack = []
    i = len(pos_similarities_list) - 1
    j = lis_len - 1
    matched_score = 0
    while i >= 0: #从后往前找，所以先入栈的位置序号大
        if lis_pos[i] == j:
            if j > 0:
                stack.append(pos_similarities_list[i][0])
                matched_score += pos_similarities_list[i][1]
            else:
                if len(stack) > 0:
                    # 找到左边最靠右的位置
                    while i > 0 and pos_similarities_list[i-1][0] < stack[-1]:
                        i -= 1
                stack.append(pos_similarities_list[i][0]) #使用
                matched_score += pos_similarities_list[i][1]
                break
            j -= 1
        if j == -1:
            break
        i -= 1
    min_pos = stack[-1] #从后往前找，所以后入栈的位置序号小
    max_pos = stack[0] #从后往前找，所以先入栈的位置序号大
    matched_score = matched_score / max(len(words_pinyin), max_pos - min_pos + 1)
    if matched_score < threshold:
        return None
    matched_info = {'maxMatchedLen': max_pos - min_pos + 1, 'matchedScore': matched_score, 'range': [min_pos, max_pos]}
    return matched_info

def get_all_unify_pinyins(han):
    pinyins = pinyin(han, style=Style.NORMAL, heteronym=True)[0] #这里han仅为一个汉字，所以只取第一个只所有拼音（考虑多音字）
    uni_pinyins = []
    for p in pinyins:
        py = p
        if len(p) >= 2:
            prefix = py[:2]
            if prefix == "zh" or prefix == 'ch' or prefix == 'sh':
                py = prefix[:1] + p[2:]
        if py[0] == 'n' and len(py) > 1:
            py = 'l' + py[1:]
        if len(py) >= 3:
            postfix = py[-3:]
            if postfix == "ang" or postfix == 'eng' or postfix == 'ing':
                py = py[:-3] + postfix[:2]
        uni_pinyins.append(py)
    return uni_pinyins

def get_unify_pinyins(hans):
    pinyins = lazy_pinyin(hans) # 不考虑多音字情况（该方法会根据词/句子将多音字转为正确的拼音---单华奇人名不行）
    uni_pinyins = []
    for p in pinyins:
        py = p
        if len(p) >= 2:
            prefix = py[:2]
            if prefix == "zh" or prefix == 'ch' or prefix == 'sh':
                py = prefix[:1] + p[2:]
        if py[0] == 'n' and len(py) > 1:
            py = 'l' + py[1:]
        if len(py) >= 3:
            postfix = py[-3:]
            if postfix == "ang" or postfix == 'eng' or postfix == 'ing':
                py = py[:-3] + postfix[:2]
        uni_pinyins.append(py)
    return uni_pinyins

get_unify_pinyins('单华奇')

if __name__ == "__main__":
    # a = 'nihao'
    # print(a, is_alphabet_string(a))
    # # test Q2B and B2Q
    # for i in range(0x0020, 0x007F):
    #     print(Q2B(B2Q(chr(i))), B2Q(chr(i)))
    # # test uniform
    # ustring = '中国 人名ａ高频Ａ  扇'
    # ustring = uniform(ustring)
    # print(ustring)
    # print(is_other(','))
    # print(uniform('你干么！ｄ７＆８８８学英 语ＡＢＣ？ｎｚ'))
    # print(is_chinese('喜'))
    # print(is_chinese_string('喜,'))
    # print(is_chinese_string('丽，'))
    #
    # traditional_sentence = '憂郁的臺灣烏龜'
    # simplified_sentence = traditional2simplified(traditional_sentence)
    # print(traditional_sentence, simplified_sentence)
    # print(is_alphabet_string('Teacher'))
    # print(is_alphabet_string('Teacher '))

    # lcs(sentence_pinyin_map, words_pinyin, threshold=0.7):
    # lst = [2]
    # pos = bin_search(lst, 1, 1)
    # print(pos)
    #
    # lst = [1]
    # pos = bin_search(lst, 1, 5)
    # print(pos)
    #
    # lst = [1, 5]
    # pos = bin_search(lst, 2, 3)
    # print(pos)
    #
    # lst = [1, 3]
    # pos = bin_search(lst, 2, 6)
    # print(pos)
    #
    # lst = [1, 3, 6]
    # pos = bin_search(lst, 3, 4)
    # print(pos)
    #
    # lst = [1, 3, 4]
    # pos = bin_search(lst, 3, 8)
    # print(pos)

    # lst = [5, 6]
    # pos = bin_search(lst, 2, 3)
    # print(pos)
    #
    # lst = [3, 6]
    # pos = bin_search(lst, 2, 4)
    # print(pos)

    # sentence_pinyin_map = {}
    # sentence_pinyin_map['jia']=[[0,1]]
    # sentence_pinyin_map['jian'] = [[0, 0.75]]
    # sentence_pinyin_map['jiao'] = [[0, 0.75]]
    # sentence_pinyin_map['li'] = [[1, 1]]
    # sentence_pinyin_map['de'] = [[2, 1]]
    # sentence_pinyin_map['wai'] = [[3, 1]]
    # sentence_pinyin_map['fa'] = [[4, 1]]
    # sentence_pinyin_map['xin'] = [[5, 1]]
    # sentence_pinyin_map['xian'] = [[5, 0.75]]
    # sentence_pinyin_map['hao'] = [[8, 1],[6,1]]
    # sentence_pinyin_map['bu'] = [[7, 1]]
    # words_pinyin = ['wai','fa','wai','fa','hao']
    # lcs_info = lcs(sentence_pinyin_map, words_pinyin, threshold=0.7)

    # sentence_pinyin_map = {}
    # sentence_pinyin_map['xin'] = [[5, 1]]
    # sentence_pinyin_map['hao'] = [[6, 1]]
    # sentence_pinyin_map['bu'] = [[3, 1]]
    # words_pinyin = ['xin','hao','bu']
    # lcs_info = lcs(sentence_pinyin_map, words_pinyin, threshold=0.7)
    # print(lcs_info)

    # sentence_pinyin_map = {}
    # sentence_pinyin_map['xin'] = [[5, 0.5]]
    # sentence_pinyin_map['hao'] = [[1, 1]]
    # sentence_pinyin_map['bu'] = [[4, 0.4]]
    # words_pinyin = ['xin','hao','bu']
    # lcs_info = lcs(sentence_pinyin_map, words_pinyin, threshold=0.1)
    # print(lcs_info)

    sentence_pinyin_map = {}
    sentence_pinyin_map['suan'] = [[6, 1]]
    sentence_pinyin_map['cen'] = [[3, 0.6],[1,0.6]]
    words_pinyin = ['suan','cen']
    lcs_info = lcs(sentence_pinyin_map, words_pinyin, threshold=0.1)
    print(lcs_info)