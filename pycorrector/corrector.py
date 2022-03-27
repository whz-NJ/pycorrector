# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: corrector with pinyin and stroke
"""
import operator
import os
from codecs import open
import sys
from pypinyin import lazy_pinyin
import numpy as np
from . import config
from .detector import Detector, ErrorType
from .utils.logger import logger
from .utils.math_utils import edit_distance_word
from .utils.text_utils import is_chinese_string, convert_to_unicode
from .utils.tokenizer import segment, split_2_short_text


class Corrector(Detector):
    def __init__(self,
                 common_char_path=config.common_char_path,
                 similar_hans_path=config.similar_hans_path,
                 # same_stroke_path=config.same_stroke_path,
                 language_model_path=config.language_model_path,
                 word_freq_path=config.word_freq_path,
                 # custom_word_freq_path='',
                 custom_word_freq_path=config.custom_word_freq_path,
                 custom_confusion_path=config.custom_confusion_path,
                 person_name_path=config.person_name_path,
                 place_name_path=config.place_name_path,
                 stopwords_path=config.stopwords_path,
                 en_ch_alias_path=config.en_ch_alias_path,
                 similar_pinyins_path = config.similar_pinyins_path
                 ):
        super(Corrector, self).__init__(language_model_path=language_model_path,
                                        word_freq_path=word_freq_path,
                                        custom_word_freq_path=custom_word_freq_path,
                                        custom_confusion_path=custom_confusion_path,
                                        person_name_path=person_name_path,
                                        place_name_path=place_name_path,
                                        stopwords_path=stopwords_path,
                                        similar_hans_path=similar_hans_path,
                                        en_ch_alias_path=en_ch_alias_path,
                                        similar_pinyins_path=similar_pinyins_path
                                        )
        self.name = 'corrector'
        self.common_char_path = common_char_path
        #self.same_stroke_text_path = same_stroke_path
        self.initialized_corrector = False
        self.cn_char_set = None
        self.same_stroke = None

    @staticmethod
    def load_set_file(path):
        words = set()
        with open(path, 'r', encoding='utf-8') as f:
            for w in f:
                w = w.strip()
                if w.startswith('#'):
                    continue
                if w:
                    words.add(w)
        return words

    @staticmethod
    def load_same_stroke(path, sep='\t'):
        """
        加载形似字
        :param path:
        :param sep:
        :return:
        """
        result = dict()
        if not os.path.exists(path):
            logger.warn("file not exists:" + path)
            return result
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#'):
                    continue
                parts = line.split(sep)
                if parts and len(parts) > 1:
                    for i, c in enumerate(parts):
                        exist = result.get(c, set())
                        current = set(list(parts[:i] + parts[i + 1:]))
                        result[c] = exist.union(current)
        return result

    def _initialize_corrector(self):
        # chinese common char
        self.cn_char_set = self.load_set_file(self.common_char_path)
        # same stroke
        #self.same_stroke = self.load_same_stroke(self.same_stroke_text_path)
        self.initialized_corrector = True

    def check_corrector_initialized(self):
        if not self.initialized_corrector:
            self._initialize_corrector()

    def get_same_stroke(self, char):
        """
        取形似字
        :param char:
        :return:
        """
        self.check_corrector_initialized()
        return self.same_stroke.get(char, set())

    def known(self, words):
        """
        取得词序列中属于常用词部分
        :param words:
        :return:
        """
        self.check_detector_initialized()
        return set(word for word in words if word in self.word_freq)

    def _confusion_char_set(self, c):
        return self.get_same_pinyin(c).union(self.get_same_stroke(c))

    def _confusion_word_set(self, word):
        confusion_word_set = set()
        ## WHZ 找到所有编辑距离为1的常用词（长度相同）
        candidate_words = list(self.known(edit_distance_word(word, self.cn_char_set)))
        for candidate_word in candidate_words:
            #如果两个词的拼音一样，则替换
            if lazy_pinyin(candidate_word) == lazy_pinyin(word):
                # same pinyin
                confusion_word_set.add(candidate_word)
        return confusion_word_set

    def _confusion_custom_set(self, word):
        confusion_word_set = set()
        if word in self.custom_confusion:
            confusion_word_set = {self.custom_confusion[word]}
        return confusion_word_set

    def generate_items(self, word, fragment=1):
        """
        生成纠错候选集
        :param word:
        :param fragment: 分段
        :return:
        """
        self.check_corrector_initialized()
        # 1字
        candidates_1 = []
        # 2字
        candidates_2 = []
        # 多于2字
        candidates_3 = []

        # same pinyin word
        candidates_1.extend(self._confusion_word_set(word))
        # custom confusion word
        candidates_1.extend(self._confusion_custom_set(word))
        # same pinyin char
        if len(word) == 1:
            # same one char pinyin
            confusion = [i for i in self._confusion_char_set(word[0]) if i]
            candidates_1.extend(confusion)
        if len(word) >= 2:
            # same first char pinyin
            confusion = [i + word[1:] for i in self._confusion_char_set(word[0]) if i]
            candidates_2.extend(confusion)
            # same last char pinyin
            confusion = [word[:-1] + i for i in self._confusion_char_set(word[-1]) if i]
            candidates_2.extend(confusion)
        if len(word) > 2:
            # same mid char pinyin
            confusion = [word[0] + i + word[2:] for i in self._confusion_char_set(word[1])]
            candidates_3.extend(confusion)

            # same first word pinyin
            confusion_word = [i + word[-1] for i in self._confusion_word_set(word[:-1])]
            candidates_3.extend(confusion_word)

            # same last word pinyin
            confusion_word = [word[0] + i for i in self._confusion_word_set(word[1:])]
            candidates_3.extend(confusion_word)

        # add all confusion word list
        confusion_word_set = set(candidates_1 + candidates_2 + candidates_3)
        confusion_word_list = [item for item in confusion_word_set if is_chinese_string(item)]
        confusion_sorted = sorted(confusion_word_list, key=lambda k: self.word_frequency(k), reverse=True)
        return confusion_sorted[:len(confusion_word_list) // fragment + 1]

    def get_lm_correct_item(self, sentence, modified_sentence_old_start_idx, crossed_begin_end_idx, maybe_errors_map, details, cut_type='char', threshold=1.5):
        """
        通过语言模型纠正字词错误
        :param sentence: 目前为止纠错得到的最新的句子内容
        :param modified_sentence_old_start_idx: 当前句子在前面部分已修改text中的位置
        :param crossed_begin_end_idx: 重叠纠错项合并起始/结束位置(词在 text 原始输入的起始/结束位置)
        :param maybe_errors_map: 目前为止待处理的纠错项
        :parm details: 纠错历史信息
        :param cut_type: 切词方式, 字粒度
        :param threshold: ppl系数, 原始字词替换后大于原词ppl值*系数，则认为是错误
        :return: 更新后的 sentence（ maybe_errors_map 内容也会更新）
        """
        # 得到待纠错词在 sentence 中的位置
        sentence = sentence.lower() #将可能存在的英文单词全部转换为小写
        pos_index_pair = crossed_begin_end_idx.split("_")
        crossed_begin_idx = int(pos_index_pair[0])
        crossed_end_idx = int(pos_index_pair[1])
        #找纠错项
        top_token = None
        crossed_maybe_errors = maybe_errors_map[crossed_begin_end_idx]
        for token, begin_idx, end_idx, matched_score, error_type in crossed_maybe_errors:
            if error_type == ErrorType.confusion: #强制纠错
                top_token = token
                old_begin_idx = begin_idx
                old_end_idx = end_idx
                # 转换为目前为止已部分修改的句子的位置
                new_begin_idx = self.get_current_pose_idx(old_begin_idx) - modified_sentence_old_start_idx
                # 因为要取替换前的后续文字，所以这里值不是 new_begin_idx + len(top_token)
                new_end_idx = new_begin_idx + (old_end_idx - old_begin_idx)
                cur_item = sentence[new_begin_idx:new_end_idx]
                break #重叠位置的纠错项仅随机取一个
        if top_token is None: #未找到强制纠错项
            # new_begin_idx = self.get_current_pose_idx(crossed_begin_idx) - modified_sentence_old_start_idx
            # # crossed_end_idx 在 old_new_pose_idx_list 表没有记录
            # new_end_idx = new_begin_idx + (crossed_end_idx - crossed_begin_idx)
            # # 取得待纠错的词
            # cur_item = sentence[new_begin_idx:new_end_idx]
            #correction_ppl_score_map = {}
            cur_score = self.ppl_score(segment(sentence, cut_type))
            # cur_candidate = (cur_item, crossed_begin_idx, crossed_end_idx, ErrorType.word)
            #correction_ppl_score_map[candidate] = score
            top_scores = []
            top_candidates = []
            top_similarities = []
            top_candidate = None
            for candidate in crossed_maybe_errors:
                token, begin_idx, end_idx = candidate[0],candidate[1],candidate[2]
                new_begin_idx = self.get_current_pose_idx(begin_idx) - modified_sentence_old_start_idx
                new_end_idx = new_begin_idx + (end_idx - begin_idx)
                score = self.ppl_score(segment(sentence[:new_begin_idx] + token + sentence[new_end_idx:], cut_type))
                if (score * threshold) < cur_score: #得分越小越通顺
                    top_scores.append(score)
                    top_candidates.append(candidate)
                    top_similarities.append(candidate[3])
            if len(top_scores) > 0:
                mean_score = np.mean(top_scores) #均值
                std_score = np.std(top_scores) #标准差
                mean_similarity = np.mean(top_similarities)
                std_similarity = np.std(top_similarities)
                max_composed_score = -sys.maxsize
                for idx in range(len(top_scores)):
                    if std_score > 0.00001:
                        score = (top_scores[idx] - mean_score) / std_score #参数标准化
                    else:
                        score = 0
                    if std_similarity > 0.00001:
                        similarity = (top_similarities[idx] - mean_similarity) / std_similarity
                    else:
                        similarity = 0
                    composed_score = similarity - score
                    if max_composed_score < composed_score:
                        top_candidate = top_candidates[idx]
                        max_composed_score = composed_score
            if top_candidate is not None:
                top_token = top_candidate[0]
                old_begin_idx = top_candidate[1]
                old_end_idx = top_candidate[2]
                new_begin_idx = self.get_current_pose_idx(old_begin_idx) - modified_sentence_old_start_idx
                new_end_idx = new_begin_idx + (old_end_idx - old_begin_idx)
                cur_item = sentence[new_begin_idx:new_end_idx]

        del maybe_errors_map[crossed_begin_end_idx]
        if top_token is not None: #发现调整项
            # 修正句子
            sentence = sentence[:new_begin_idx] + top_token + sentence[new_end_idx:]
            # 记录修正信息
            detail_word = (cur_item, top_token, old_begin_idx, old_end_idx)
            details.append(detail_word)
            # 寻找和 top_item 不重叠的 item
            for candidate in crossed_maybe_errors:
                # token, begin_idx, end_idx = candidate
                token, begin_idx, end_idx = candidate[0], candidate[1], candidate[2]
                if begin_idx >= old_end_idx or end_idx <= old_begin_idx:
                    self._add_maybe_error_item(candidate, maybe_errors_map)
            self.update_pose_idx(old_begin_idx, old_end_idx - old_begin_idx, top_token)
        return sentence

    def getRange(self, key):
        min_max = key.split("_")
        min = int(min_max[0])
        max = int(min_max[1])
        return max - min

    def correct(self, text, include_symbol=True, num_fragment=1, threshold=57, **kwargs):
        """
        句子改错
        :param text: str, query 文本
        :param include_symbol: bool, 是否包含标点符号
        :param num_fragment: 纠错候选集分段数, 1 / (num_fragment + 1)
        :param threshold: 语言模型纠错ppl阈值
        :param kwargs: ...
        :return: text (str)改正后的句子, list(wrong, right, begin_idx, end_idx)
        """
        text_new = ''
        details = []
        self.check_corrector_initialized()
        # 编码统一，utf-8 to unicode
        text = convert_to_unicode(text)
        # 长句切分为短句
        blocks = split_2_short_text(text, include_symbol=include_symbol)
        # self.old_new_pos_list = []
        self.old_new_pose_idx_list = []
        total_delta = 0
        for blk, idx in blocks:
            if len(blk) <= 1:
                text_new += blk
                continue
            # maybe_errors_map 的 key 为原词在 text 中的 开始位置_结束位置
            # value 为 [候选词,原词起始位置,原词结束位置,错误类型]
            blk2 = blk
            maybe_errors_map = self.detect_short(blk2, idx, total_delta)
            # 按照每个key归类的 maybe_errors 中取最匹配的，再把与该类与最匹配的 maybe_error 不重叠的再归成子类，
            # 求各子类中最匹配的 maybe_errors，依次下去，直到集合为空，优先处理范围宽的
            while len(maybe_errors_map) > 0:
                for crossed_begin_end_idx in sorted(maybe_errors_map.keys(), key = lambda k: self.getRange(k), reverse=True):
                    blk2 = self.get_lm_correct_item(blk2, idx + total_delta, crossed_begin_end_idx, maybe_errors_map, details)
            text_new += blk2
            total_delta += (len(blk2) - len(blk))
        details = sorted(details, key=operator.itemgetter(2))
        return text_new, details
