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

# get LCS(longest common subsquence),DP
def lcs(pinyin_similarity_map, sentence_pinyin, words_pinyin, threshold=0.7):
    # 得到一个二维的数组，类似用dp[lena+1][lenb+1],并且初始化为0
    similarities = [[0 for j in range(len(words_pinyin) + 1)] for i in range(len(sentence_pinyin) + 1)]
    direction = [[0 for j in range(len(words_pinyin) + 1)] for i in range(len(sentence_pinyin) + 1)]

    # enumerate(a)函数： 得到下标i和a[i]
    for i, x in enumerate(sentence_pinyin):
        for j, y in enumerate(words_pinyin):
            similarity = _get_update_pinyin_similarity(pinyin_similarity_map, x, y)
            if similarity > threshold:
                similarities[i + 1][j + 1] = similarities[i][j] + similarity
                direction[i + 1][j + 1] = '↖'
            else:
                left_similarity = similarities[i + 1][j]
                top_similarity = similarities[i][j + 1]
                if left_similarity > top_similarity:
                    similarities[i + 1][j + 1] = left_similarity
                    direction[i + 1][j + 1] = '←'
                else:
                    similarities[i + 1][j + 1] = top_similarity
                    direction[i + 1][j + 1] = '↑'

    similarities_sum = similarities[len(sentence_pinyin)][len(words_pinyin)]
    rough_score = similarities_sum / len(words_pinyin)
    if rough_score < threshold:
        return []

    # 到这里已经得到最长的子序列的长度，下面从这个矩阵中就是得到最长子序列
    result = []
    x = len(sentence_pinyin)
    while x != 0:
        y = len(words_pinyin)
        if similarities[x][y] < similarities_sum:
            break
        while direction[x - 1][y] == '↖' and similarities[x - 1][y] == similarities_sum:
            x -= 1 #找右侧最靠近左侧的最贴近的匹配位置
        minX, maxX = len(sentence_pinyin), -1
        matched_pinyins = ''
        while x != 0 and y != 0:
            if direction[x][y] == '↖':
                matched_pinyins = sentence_pinyin[x - 1] + matched_pinyins
                x -= 1
                y -= 1
                if maxX == -1:
                    maxX = x
                minX = x
            elif direction[x][y] == '←':
                y -= 1
            else:  # direction[x][y]='↑'
                x -= 1
        if maxX == -1:
            matched_info = {'matchedScore': 0}
        else:
            matched_score = similarities_sum / max((maxX - minX + 1), len(words_pinyin))
            matched_info = {'maxMatchedLen': similarities_sum, 'matchedScore': matched_score, 'range': [minX, maxX]}
        result.append(matched_info)
    return result

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
    a = 'nihao'
    print(a, is_alphabet_string(a))
    # test Q2B and B2Q
    for i in range(0x0020, 0x007F):
        print(Q2B(B2Q(chr(i))), B2Q(chr(i)))
    # test uniform
    ustring = '中国 人名ａ高频Ａ  扇'
    ustring = uniform(ustring)
    print(ustring)
    print(is_other(','))
    print(uniform('你干么！ｄ７＆８８８学英 语ＡＢＣ？ｎｚ'))
    print(is_chinese('喜'))
    print(is_chinese_string('喜,'))
    print(is_chinese_string('丽，'))

    traditional_sentence = '憂郁的臺灣烏龜'
    simplified_sentence = traditional2simplified(traditional_sentence)
    print(traditional_sentence, simplified_sentence)
    print(is_alphabet_string('Teacher'))
    print(is_alphabet_string('Teacher '))

    result = lcs(["qing",'xiao', 'feng'], ['qiao','feng'])
    print(result)

