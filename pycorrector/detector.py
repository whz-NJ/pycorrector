# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: error word detector
"""
import os
import time
from codecs import open
import re
import numpy as np

from . import config
from .utils.get_file import get_file
from .utils.logger import logger
from .utils.text_utils import uniform, is_alphabet_string, convert_to_unicode, is_chinese_string,get_unify_pinyins,get_all_unify_pinyins, lcs
from .utils.tokenizer import Tokenizer, split_2_short_text
from .utils.langconv import Converter
from .utils.bktree import levenshtein,BKTree

class ErrorType(object):
    # error_type = {"confusion": 1, "word": 2, "char": 3}
    confusion = 'confusion'
    word = 'word'


class Detector(object):
    pre_trained_language_models = {
        # 语言模型 2.95GB
        'zh_giga.no_cna_cmn.prune01244.klm': 'https://deepspeech.bj.bcebos.com/zh_lm/zh_giga.no_cna_cmn.prune01244.klm',
        # 人民日报训练语言模型 20MB
        'people_chars_lm.klm': 'https://www.borntowin.cn/mm/emb_models/people_chars_lm.klm'
    }

    def __init__(self,
                 language_model_path=config.language_model_path,
                 word_freq_path=config.word_freq_path,
                 custom_word_freq_path='',
                 custom_confusion_path='',
                 person_name_path=config.person_name_path,
                 place_name_path=config.place_name_path,
                 stopwords_path=config.stopwords_path,
                 same_pinyin_path=config.same_pinyin_path,
                 en_ch_alias_path=config.en_ch_alias_path,
                 ):
        self.name = 'detector'
        self.language_model_path = language_model_path
        self.word_freq_path = word_freq_path
        self.custom_word_freq_path = custom_word_freq_path
        self.custom_confusion_path = custom_confusion_path
        self.person_name_path = person_name_path
        self.place_name_path = place_name_path
        self.stopwords_path = stopwords_path
        self.same_pinyin_text_path = same_pinyin_path
        self.en_ch_alias_path = en_ch_alias_path
        self.is_char_error_detect = True
        self.is_word_error_detect = True
        self.initialized_detector = False
        self.lm = None
        self.word_freq = None
        self.custom_confusion = None
        self.custom_word_freq = None
        self.person_names = None
        self.place_names = None
        self.stopwords = None
        self.tokenizer = None

    def _initialize_detector(self):
        t1 = time.time()
        try:
            import kenlm
        except ImportError:
            raise ImportError('pycorrector dependencies are not fully installed, '
                              'they are required for statistical language model.'
                              'Please use "pip install kenlm" to install it.'
                              'if you are Win, Please install kenlm in cgwin.')
        if not os.path.exists(self.language_model_path):
            filename = self.pre_trained_language_models.get(self.language_model_path,
                                                            'zh_giga.no_cna_cmn.prune01244.klm')
            url = self.pre_trained_language_models.get(filename)
            get_file(
                filename, url, extract=True,
                cache_dir='~',
                cache_subdir=config.USER_DATA_DIR,
                verbose=1
            )
        self.lm = kenlm.Model(self.language_model_path)
        t2 = time.time()
        logger.debug('Loaded language model: %s, spend: %.3f s.' % (self.language_model_path, t2 - t1))

        # 词、频数dict
        self.word_freq = self.load_word_freq_dict(self.word_freq_path)
        # 自定义混淆集
        self.custom_confusion = self._get_custom_confusion_dict(self.custom_confusion_path)
        # 自定义切词词典
        self.custom_word_freq = self.load_word_freq_dict(self.custom_word_freq_path)
        self.person_names = self.load_word_freq_dict(self.person_name_path)
        self.place_names = self.load_word_freq_dict(self.place_name_path)
        self.stopwords = self.load_word_freq_dict(self.stopwords_path)
        self.same_pinyin = self.load_same_pinyin(self.same_pinyin_text_path)
        self.en_ch_alias = self.load_en_ch_alias(self.en_ch_alias_path)
        # 合并切词词典及自定义词典 append:加单个元素，extend加list多个元素
        self.custom_word_freq.update(self.person_names)
        self.custom_word_freq.update(self.place_names)
        self.custom_word_freq.update(self.stopwords)
        self.word_freq.update(self.custom_word_freq)
        self.tokenizer = Tokenizer(dict_path=self.word_freq_path, custom_word_freq_dict=self.custom_word_freq,
                                   custom_confusion_dict=self.custom_confusion)
        self._build_freq_words_pinyin_map_bk_tree()

        self.old_new_pose_idx_list = []
        t3 = time.time()
        logger.debug('Loaded dict file, spend: %.3f s.' % (t3 - t2))
        self.initialized_detector = True

    def check_detector_initialized(self):
        if not self.initialized_detector:
            self._initialize_detector()

    def _build_freq_words_pinyin_map_bk_tree(self):
        pinyin_freq_words_map = dict()
        for freq_word in self.word_freq:
            pys = get_unify_pinyins(freq_word)
            freq_words = pinyin_freq_words_map.get("".join(pys), None)
            if freq_words is not None:
                freq_words.append(freq_word)
            else:
                pinyin_freq_words_map["".join(pys)] = [freq_word]
        self.pinyin_freq_words_map = pinyin_freq_words_map

        pinyin_english_map = dict()
        for en in self.en_ch_alias:
            ch_aliases = self.en_ch_alias[en]
            for ch_alias in ch_aliases:
                pys = get_unify_pinyins(ch_alias)
                english = pinyin_english_map.get("".join(pys), None)
                if english is None:
                    pinyin_english_map["".join(pys)] = english # 保存谐音汉字拼音和英文单词对应关系
                else: # 相同的谐音汉字拼音只能对应一个英文单词
                    continue
        self.pinyin_english_map = pinyin_english_map

        pinyin_set1 = set(pinyin_freq_words_map.keys())
        pinyin_set2 = set(pinyin_english_map.keys())
        pinyin_set = pinyin_set1.union(pinyin_set2)
        self.freq_words_pinyin_bk_tree = BKTree(levenshtein, pinyin_set)

    def get_same_pinyin(self, char):
        """
        取同音字
        :param char:
        :return:
        """
        self.check_corrector_initialized()
        return self.same_pinyin.get(char, set())

    @staticmethod
    def load_same_pinyin(path, sep='\t'):
        """
        加载同音字
        :param path:
        :param sep:
        :return:
        """
        same_pinyin_map = dict()
        if not os.path.exists(path):
            logger.warn("file not exists:" + path)
            return same_pinyin_map
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#'):
                    continue
                parts = line.split(sep)
                if not parts or len(parts) ==0:
                    continue

                key_char = parts[0]
                if len(parts) == 1:
                    value = set()
                elif len(parts) == 2:
                    value = set(parts[1])
                elif parts and len(parts) >= 3:
                    value1 = set(parts[1])
                    value2 = set(parts[2])
                    value = value1.union(value2)
                same_pinyin_map[key_char] = value
        return same_pinyin_map

    @staticmethod
    def load_en_ch_alias(path, sep='\t'):
        """
        加载英文单词的中文谐音字表
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
                if parts and len(parts) == 2:
                    words = parts[1].split(',')
                    chineses = set()
                    for han in words:
                        chineses.add(han.lower())
                    english = parts[0]
                    result[english] = chineses
        return result

    @staticmethod
    def load_word_freq_dict(path):
        """
        加载切词词典
        :param path:
        :return:
        """
        word_freq = {}
        if path:
            if not os.path.exists(path):
                logger.warning('file not found.%s' % path)
                return word_freq
            else:
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith('#'):
                            continue
                        info = line.split()
                        if len(info) < 1:
                            continue
                        word = info[0]
                        # 取词频，默认1
                        freq = int(info[1]) if len(info) > 1 else 1
                        word = (Converter('zh-hans').convert(word).lower())
                        old_freq = word_freq.get(word, None)
                        if old_freq is None:
                            word_freq[word] = freq
                        else:
                            word_freq[word] = freq + old_freq
        return word_freq

    def _get_custom_confusion_dict(self, path):
        """
        取自定义困惑集
        :param path:
        :return: dict, {variant: origin}, eg: {"交通先行": "交通限行"}
        """
        confusion = {}
        if path:
            if not os.path.exists(path):
                logger.warning('file not found.%s' % path)
                return confusion
            else:
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith('#'):
                            continue
                        info = line.split()
                        if len(info) < 2:
                            continue
                        variant = info[0]
                        origin = info[1]
                        freq = int(info[2]) if len(info) > 2 else 1
                        self.word_freq[origin] = freq
                        confusion[variant] = origin
        return confusion

    def set_language_model_path(self, path):
        self.check_detector_initialized()
        import kenlm
        self.lm = kenlm.Model(path)
        logger.debug('Loaded language model: %s' % path)

    def set_custom_confusion_dict(self, path):
        self.check_detector_initialized()
        self.custom_confusion = self._get_custom_confusion_dict(path)
        logger.debug('Loaded confusion path: %s, size: %d' % (path, len(self.custom_confusion)))

    def set_custom_word_freq(self, path):
        self.check_detector_initialized()
        word_freqs = self.load_word_freq_dict(path)
        # 合并字典
        self.custom_word_freq.update(word_freqs)
        # 合并切词词典及自定义词典
        self.word_freq = self.word_freq.update(self.custom_word_freq)
        self.tokenizer = Tokenizer(dict_path=self.word_freq_path, custom_word_freq_dict=self.custom_word_freq,
                                   custom_confusion_dict=self.custom_confusion)
        for k, v in word_freqs.items():
            self.set_word_frequency(k, v)
        logger.debug('Loaded custom word path: %s, size: %d' % (path, len(word_freqs)))

    def enable_char_error(self, enable=True):
        """
        is open char error detect
        :param enable:
        :return:
        """
        self.is_char_error_detect = enable

    def enable_word_error(self, enable=True):
        """
        is open word error detect
        :param enable:
        :return:
        """
        self.is_word_error_detect = enable

    def ngram_score(self, chars):
        """
        取n元文法得分
        :param chars: list, 以词或字切分
        :return:
        """
        self.check_detector_initialized()
        return self.lm.score(' '.join(chars), bos=False, eos=False)

    def ppl_score(self, words):
        """
        取语言模型困惑度得分，越小句子越通顺
        :param words: list, 以词或字切分
        :return:
        """
        self.check_detector_initialized()
        return self.lm.perplexity(' '.join(words))

    def word_frequency(self, word):
        """
        取词在样本中的词频
        :param word:
        :return: dict
        """
        self.check_detector_initialized()
        return self.word_freq.get(word, 0)

    @staticmethod
    def _check_contain_error(maybe_err, maybe_errors):
        """
        检测错误集合(maybe_errors)是否已经包含该错误位置（maybe_err)
        :param maybe_err: [error_word, begin_pos, end_pos, error_type]
        :param maybe_errors:list
        :return: bool
        """
        error_word_idx = 0
        begin_idx = 1
        end_idx = 2
        for err in maybe_errors:
            # 待选词是已选词集一部分，并且长度小于等于已选词
            if maybe_err[error_word_idx] in err[error_word_idx] and maybe_err[begin_idx] >= err[begin_idx] and \
                    maybe_err[end_idx] <= err[end_idx]:
                return True
        return False

    def bin_search_pose_idx(self, old_pose_idx):
        left = 0
        right = len(self.old_new_pose_idx_list) - 1
        while left <= right:
            mid = (left + right) //2
            if old_pose_idx > self.old_new_pose_idx_list[mid][0]:
                left = mid + 1
            elif old_pose_idx < self.old_new_pose_idx_list[mid][0]:
                right = mid -1
            else:
                return mid
        raise Exception("未找到位置")

    def update_pose_idx(self, old_begin_pos_idx, old_word_len, correct_word):
        delta = len(correct_word) - old_word_len
        if delta != 0:
            begin_pos_idx = self.bin_search_pose_idx(old_begin_pos_idx)
            # self.old_new_pose_idx_list[begin_pos_idx][1] += delta
            idx = begin_pos_idx + 1
            while idx < len(self.old_new_pose_idx_list):
                self.old_new_pose_idx_list[idx][1] += delta
                idx += 1

    def get_current_pose_idx(self, old_begin_pos_idx):
        pose = self.bin_search_pose_idx(old_begin_pos_idx)
        return self.old_new_pose_idx_list[pose][1]

    def add_pose_idx(self, pose_idx, new_pos_idx):
        left = 0
        right = len(self.old_new_pose_idx_list) - 1
        mid = None
        while left <= right:
            mid = (left + right) // 2
            if pose_idx > self.old_new_pose_idx_list[mid][0]:
                left = mid + 1
            elif pose_idx < self.old_new_pose_idx_list[mid][0]:
                right = mid - 1
            else:
                return  # 重复的位置
        if mid is None: # 不能用 if not mid，因为mid为0是也将为 False
            self.old_new_pose_idx_list.append([pose_idx, new_pos_idx])
        elif pose_idx > self.old_new_pose_idx_list[mid][0]:
            self.old_new_pose_idx_list.insert(mid + 1, [pose_idx, new_pos_idx])
        else:
            self.old_new_pose_idx_list.insert(mid, [pose_idx, new_pos_idx])

    def _merge_maybe_error_item(self, maybe_err, key, maybe_errors_map):
        """
        将 maybe_err 项合并到 maybe_errors_map[key] 中
        :param maybe_err: maybe_err 纠错项
        :param key: maybe_errors_map 纠错项map key
        :param key: maybe_errors_map 纠错项map
        :return: maybe_err 和 key 纠错范围是否有重叠
        """
        begin_idx = 1
        end_idx = 2
        begin_pos1 = maybe_err[begin_idx]
        end_pos1 = maybe_err[end_idx]

        poses = key.split("_")
        begin_pos0 = int(poses[0])
        end_pos0 = int(poses[1])
        if (begin_pos1 >= end_pos0) or (end_pos1 <= begin_pos0):
            new_key = str(maybe_err[begin_idx]) + "_" + str(maybe_err[end_idx])
            maybe_errors_map[new_key] = [maybe_err]
            return False  # 纠错范围没有重叠
        if (begin_pos0 == begin_pos1) and (end_pos0 == end_pos1): #纠错范围一致
            maybe_errors = maybe_errors_map[key]
            if maybe_err in maybe_errors:
                return True # 纠错范围有重叠（重复添加）
        # 有重叠，将重叠的项合并到list中
        min_begin = min(begin_pos0, begin_pos1)
        max_end = max(end_pos0, end_pos1)
        new_key = str(min_begin) + "_" + str(max_end)
        maybe_errors = maybe_errors_map[key]
        maybe_errors.append(maybe_err)
        maybe_errors_map[new_key] = maybe_errors
        if new_key == key:
            return True # 纠错范围有重叠
        del maybe_errors_map[key]
        # 更新范围，将和 maybe_err 有重叠的所有 maybe_errors 合并为一个整体
        # 然后看看之前判断和 maybe_errr 没有重叠的项，和合并后的项是否有重叠
        begin_pos1 = min_begin
        end_pos1 = max_end

    def _add_maybe_error_item(self, maybe_err, maybe_errors_map):
        """
        新增错误
        :param maybe_err:
        :param maybe_errors_map:
        :return:
        """
        begin_idx = 1
        end_idx = 2
        begin_pos1 = maybe_err[begin_idx]
        end_pos1 = maybe_err[end_idx]
        changed = True
        merged = False
        while changed:
            changed = False
            # 查找有重叠的纠错项
            for key in list(maybe_errors_map):
                poses = key.split("_")
                begin_pos0 = int(poses[0])
                end_pos0 = int(poses[1])
                if (begin_pos1 >= end_pos0) or (end_pos1 <= begin_pos0):
                    continue # 位置没有交错
                if (begin_pos0 == begin_pos1) and (end_pos0 == end_pos1):
                    maybe_errors = maybe_errors_map[key]
                    if maybe_err in maybe_errors:
                        continue #跳过已合并项
                #有重叠，将重叠的项合并到list中
                min_begin = min(begin_pos0, begin_pos1)
                max_end = max(end_pos0, end_pos1)
                new_key = str(min_begin) + "_" + str(max_end)
                maybe_errors = maybe_errors_map[key]
                maybe_errors.append(maybe_err)
                maybe_errors_map[new_key] = maybe_errors
                if new_key == key:
                    return #新加入的纠错项,key用已经存在的,这个key之前已经考虑过合并，不需要再while循环检查了
                del maybe_errors_map[key]
                # 更新范围，将和 maybe_err 有重叠的所有 maybe_errors 合并为一个整体
                # 然后看看之前判断和 maybe_errr 没有重叠的项，和合并后的项是否有重叠
                begin_pos1 = min_begin
                end_pos1 = max_end
                merged = True
                changed = True
        # 没有找到交错位置的纠错项
        if not merged:
            new_key = str(maybe_err[begin_idx]) + "_" + str(maybe_err[end_idx])
            maybe_errors_map[new_key] = [maybe_err]

    @staticmethod
    def _get_maybe_error_index(scores, ratio=0.6745, threshold=2):
        """
        取疑似错字的位置，通过平均绝对离差（MAD）
        :param scores: np.array
        :param ratio: 正态分布表参数
        :param threshold: 阈值越小，得到疑似错别字越多
        :return: 全部疑似错误字的index: list
        """
        result = []
        scores = np.array(scores)
        if len(scores.shape) == 1:
            scores = scores[:, None]
        median = np.median(scores, axis=0)  # get median of all scores
        margin_median = np.abs(scores - median).flatten()  # deviation from the median
        # 平均绝对离差值
        med_abs_deviation = np.median(margin_median)
        if med_abs_deviation == 0:
            return result
        y_score = ratio * margin_median / med_abs_deviation
        # 打平
        scores = scores.flatten()
        maybe_error_indices = np.where((y_score > threshold) & (scores < median))
        # 取全部疑似错误字的index
        result = [int(i) for i in maybe_error_indices[0]]
        return result

    @staticmethod
    def _get_maybe_error_index_by_stddev(scores, n=2):
        """
        取疑似错字的位置，通过平均值上下n倍标准差之间属于正常点
        :param scores: list, float
        :param n: n倍
        :return: 全部疑似错误字的index: list
        """
        std = np.std(scores, ddof=1)
        mean = np.mean(scores)
        down_limit = mean - n * std
        upper_limit = mean + n * std
        maybe_error_indices = np.where((scores > upper_limit) | (scores < down_limit))
        # 取全部疑似错误字的index
        result = list(maybe_error_indices[0])
        return result

    @staticmethod
    def is_filter_token(token):
        """
        是否为需过滤字词
        :param token: 字词
        :return: bool
        """
        result = False
        # pass blank
        if not token.strip():
            result = True
        # pass num
        if token.isdigit():
            result = True
        # pass alpha
        if is_alphabet_string(token.lower()):
            result = True
        # pass not chinese
        if not is_chinese_string(token):
            result = True
        return result

    def detect(self, text):
        """
        文本错误检测
        :param text: 长文本
        :return: 错误index
        """
        maybe_errors = []
        if not text.strip():
            return maybe_errors
        # 初始化
        self.check_detector_initialized()
        # 编码统一，utf-8 to unicode
        text = convert_to_unicode(text)
        # 文本归一化
        text = uniform(text)
        # 长句切分为短句
        blocks = split_2_short_text(text)
        for blk, idx in blocks:
            maybe_errors += self.detect_short(blk, idx)
        return maybe_errors

    def detect_short(self, sentence, sentence_old_start_idx=0, former_sentences_size_changed=0):
        """
        检测句子中的疑似错误信息，包括 [候选词,原词起始位置,原词结束位置,错误类型]
        :param sentence:
        :param sentence_old_start_idx: 当前句子首字在原始text中的位置
        :param former_sentences_size_changed: 前面已修正句子总长度相对原始text的长度变化量
        :return: map{原词起始位置_原词结束位置: [correction_words, begin_pos, end_pos, error_type]}
        """
        # 初始化
        self.check_detector_initialized()
        maybe_errors_map = {}
        # 自定义混淆集加入疑似错误词典
        for confuse in self.custom_confusion: #遍历key
            idx = sentence.find(confuse)
            if idx > -1:
                correction = self.custom_confusion[confuse]
                old_start_idx = sentence_old_start_idx + idx
                new_start_idx = old_start_idx + former_sentences_size_changed
                self.add_pose_idx(old_start_idx, new_start_idx)
                # maybe_err 的 start_idx/end_idx 为修改前原词在 text(原始用户输入，包含若干句) 中的下标位置
                maybe_err = (correction, old_start_idx, old_start_idx + len(confuse), ErrorType.confusion)
                self._add_maybe_error_item(maybe_err, maybe_errors_map)

        # 切词
        min_match_threshold = 0.85
        tokens = self.tokenizer.tokenize(sentence)
        for token, begin_idx, end_idx in tokens:
            pys = get_unify_pinyins(token)
            pys_str = "".join(pys)
            max_distance = int(len(pys_str)*(1-min_match_threshold))
            if max_distance == 0:
                similar_pinyins = [pys_str]
            else:
                # 在BK树里找与当前词最接近的词（可能是热词或英文单词的中文谐音）
                similar_pinyins = self.freq_words_pinyin_bk_tree.query(pys_str, max_distance)
            if similar_pinyins and len(similar_pinyins) > 0:
                for similar_pinyin in similar_pinyins:
                    freq_words = self.pinyin_freq_words_map.get(similar_pinyin, None)
                    if freq_words is not None and len(freq_words) > 0: #有热词读音和当前词相近
                        for freq_word in freq_words:
                            if freq_word != token:
                                old_start_idx = begin_idx + sentence_old_start_idx
                                old_end_idx = (end_idx-1) + sentence_old_start_idx + 1  # 包含这个字
                                new_start_idx = old_start_idx + former_sentences_size_changed
                                self.add_pose_idx(old_start_idx, new_start_idx)
                                maybe_err = (freq_word, old_start_idx, old_end_idx, ErrorType.word)
                                self._add_maybe_error_item(maybe_err, maybe_errors_map)
                    else:
                        english = self.pinyin_english_map.get(similar_pinyin, None)
                        if english and english != token: #有英文单词的中文谐音和当前词相近
                            old_start_idx = begin_idx + sentence_old_start_idx
                            old_end_idx = (end_idx-1) + sentence_old_start_idx + 1  # 包含这个字
                            new_start_idx = old_start_idx + former_sentences_size_changed
                            self.add_pose_idx(old_start_idx, new_start_idx)
                            maybe_err = (english, old_start_idx, old_end_idx, ErrorType.word)
                            self._add_maybe_error_item(maybe_err, maybe_errors_map)

        if self.is_char_error_detect:
            # 语言模型检测疑似错误字
            try:
                ngram_avg_scores = []
                for n in [2, 3]:
                    scores = []
                    for i in range(len(sentence) - n + 1):
                        word = sentence[i:i + n]
                        score = self.ngram_score(list(word))
                        scores.append(score)
                    if not scores:
                        continue
                    # 移动窗口补全得分
                    for _ in range(n - 1):
                        scores.insert(0, scores[0])
                        scores.append(scores[-1])
                    avg_scores = [sum(scores[i:i + n]) / len(scores[i:i + n]) for i in range(len(sentence))]
                    ngram_avg_scores.append(avg_scores)

                if ngram_avg_scores:
                    # 取拼接后的n-gram平均得分
                    sent_scores = list(np.average(np.array(ngram_avg_scores), axis=0))
                    # 取疑似错字信息
                    for i in self._get_maybe_error_index(sent_scores):
                        token = sentence[i]
                        # pass filter word
                        if self.is_filter_token(token):
                            continue
                        # pass in stop word dict
                        if token in self.stopwords:
                            continue
                        # 取拼音接近的单字
                        for char in self.same_pinyin.get(token):
                            old_start_idx = i + sentence_old_start_idx
                            new_start_idx = old_start_idx + former_sentences_size_changed
                            self.add_pose_idx(old_start_idx, new_start_idx)
                            maybe_err = (char, old_start_idx, old_start_idx + 1, ErrorType.word)
                            self._add_maybe_error_item(maybe_err, maybe_errors_map)
            except IndexError as ie:
                logger.warn("index error, sentence:" + sentence + str(ie))
            except Exception as e:
                logger.warn("detect error, sentence:" + sentence + str(e))
        return maybe_errors_map
