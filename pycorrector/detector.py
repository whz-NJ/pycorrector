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
import operator

from . import config
from .utils.get_file import get_file
from .utils.logger import logger
from .utils.text_utils import uniform, is_alphabet_string, convert_to_unicode, is_chinese_string,get_unify_pinyins,get_all_unify_pinyins, lcs
from .utils.tokenizer import Tokenizer, split_2_short_text
import Levenshtein
from .utils.langconv import Converter

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
                 similar_hans_path=config.similar_hans_path,
                 en_ch_alias_path=config.en_ch_alias_path,
                 similar_pinyins_path = config.similar_pinyins_path
                 ):
        self.name = 'detector'
        self.language_model_path = language_model_path
        self.word_freq_path = word_freq_path
        self.custom_word_freq_path = custom_word_freq_path
        self.custom_confusion_path = custom_confusion_path
        self.person_name_path = person_name_path
        self.place_name_path = place_name_path
        self.stopwords_path = stopwords_path
        self.similar_hans_path = similar_hans_path
        self.en_ch_alias_path = en_ch_alias_path
        self.similar_pinyins_path = similar_pinyins_path
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

    def update_words_freq_pinyin(self):
        for word_freq in self.word_freq:
            if word_freq not in self.word_freq_pinyin:
                self.word_freq_pinyin[word_freq] = get_unify_pinyins(word_freq.lower())

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

        self.word_freq_pinyin = {}
        # 词、频数dict
        self.word_freq = self.load_word_freq_dict(self.word_freq_path)
        # 自定义混淆集
        self.custom_confusion = self._get_custom_confusion_dict(self.custom_confusion_path)
        # 自定义切词词典
        self.custom_word_freq = self.load_word_freq_dict(self.custom_word_freq_path)
        self.person_names = self.load_word_freq_dict(self.person_name_path)
        self.place_names = self.load_word_freq_dict(self.place_name_path)
        self.stopwords = self.load_word_freq_dict(self.stopwords_path)
        self.hans_similarity_map = self.load_similar_hans(self.similar_hans_path) ## ??
        self.en_ch_alias, self.en_ch_alias_pinyin = self.load_en_ch_alias(self.en_ch_alias_path)
        # 合并切词词典及自定义词典 append:加单个元素，extend加list多个元素
        self.custom_word_freq = self.custom_word_freq.union(self.person_names)
        self.custom_word_freq = self.custom_word_freq.union(self.place_names)
        # self.custom_word_freq.update(self.stopwords)
        self.word_freq = self.word_freq.union(self.custom_word_freq)
        self.update_words_freq_pinyin()
        self.pinyin_similarity_map = self.load_similar_pinyins(self.similar_pinyins_path) ## ??
        # self.tokenizer = Tokenizer(dict_path=self.word_freq_path, custom_word_freq_dict=self.custom_word_freq,
        #                            custom_confusion_dict=self.custom_confusion)
        t3 = time.time()
        logger.debug('Loaded dict file, spend: %.3f s.' % (t3 - t2))
        self.initialized_detector = True

    def check_detector_initialized(self):
        if not self.initialized_detector:
            self._initialize_detector()

    def load_similar_pinyins(self, path):
        """
         加载拼音相似度文件
         :param path:
         :return:
         """
        pinyin_similarity_map = {}
        if path:
            if not os.path.exists(path):
                logger.warning('file not found.%s' % path)
                return pinyin_similarity_map
            else:
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith('#'):
                            continue
                        info = line.split('\t')
                        if len(info) < 1:
                            continue
                        py = info[0]
                        similar_pys = []
                        if len(info) > 1:
                            for py_similarity in info[1].split(','):
                                similar_py = py_similarity.split(':')[0]
                                similarity = float(py_similarity.split(':')[1])
                                similar_pys.append([similar_py, similarity])
                        pinyin_similarity_map[py] = similar_pys
        return pinyin_similarity_map

    def build_sentence_pinyin_map(self, sentence):
        sentence_pinyin_map = {}
        pinyins = get_unify_pinyins(sentence.lower())
        for pos,py in enumerate(pinyins):
            pos_similarities = sentence_pinyin_map.get(py, None)
            if pos_similarities is None:
                 sentence_pinyin_map[py] = [[pos, 1]]
            else:
                pos_similarities.append([pos, 1])
            similar_pys = self.pinyin_similarity_map.get(py, [])
            for py_similarity in similar_pys:
                similar_py = py_similarity[0]
                similarity = py_similarity[1]
                pos_similarities = sentence_pinyin_map.get(similar_py, None)
                if pos_similarities is None:
                    sentence_pinyin_map[similar_py] = [[pos, similarity]]
                else:
                    pos_similarities.append([pos, similarity])
        for py in sentence_pinyin_map:
            pos_similarities = sentence_pinyin_map[py]
            sentence_pinyin_map[py] = sorted(pos_similarities, key=lambda x: x[0], reverse=True)
        return sentence_pinyin_map

    @staticmethod
    def _build_pinyin_similarity_map(pinyin_set):
        pinyin_similarity_map = {}
        for pinyin1 in pinyin_set:
            pinyin_similarity_map[pinyin1] = dict()
            for pinyin2 in pinyin_set:
                if pinyin2 == pinyin1:
                    pinyin_similarity_map[pinyin1][pinyin2] = 1
                    continue
                edit_distance = Levenshtein.distance(pinyin1, pinyin2)
                pinyin_similarity_map[pinyin1][pinyin2] = 1 - edit_distance / (max(len(pinyin1), len(pinyin2)))
        return pinyin_similarity_map

    def get_same_pinyin(self, char):
        """
        取同音字
        :param char:
        :return:
        """
        self.check_corrector_initialized()
        return self.hans_similarity_map.get(char, set())

    @staticmethod
    def load_similar_hans(path):
        """
        加载同音字
        :param path:
        :param sep:
        :return:
        """
        similar_hans_map = dict()
        if not os.path.exists(path):
            logger.warn("file not exists:" + path)
            return similar_hans_map
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#'):
                    continue
                parts = line.split('\t')
                if not parts or len(parts) ==0:
                    continue

                key_char = parts[0]
                value = []
                if len(parts) == 2:
                    for han_similarity in parts[1].split(','):
                        similar_han = han_similarity.split(':')[0]
                        similarity = float(han_similarity.split(':')[1])
                        value.append([similar_han, similarity])
                similar_hans_map[key_char] = value
        #pinyin_similarity_map = Detector._build_pinyin_similarity_map(pinyin_set)
        return  similar_hans_map

    @staticmethod
    def load_en_ch_alias(path, sep='\t'):
        """
        加载英文单词的中文谐音字表
        :param path:
        :param sep:
        :return:
        """
        en_ch_alias_map = dict()
        en_ch_alias_pinyin_map = dict()
        if not os.path.exists(path):
            logger.warn("file not exists:" + path)
            return [en_ch_alias_map, en_ch_alias_pinyin_map]
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#'):
                    continue
                parts = line.split(sep)
                if parts and len(parts) == 2:
                    ch_aliases = parts[1].split(',')
                    chinese_alias = set()
                    pinyins = []
                    for ch_alias in ch_aliases:
                        alias = ch_alias.lower()
                        chinese_alias.add(alias)
                        pinyins.append(get_unify_pinyins(alias))
                    english = parts[0]
                    en_ch_alias_map[english] = chinese_alias
                    en_ch_alias_pinyin_map[english] = pinyins
        return [en_ch_alias_map, en_ch_alias_pinyin_map]

    def load_word_freq_dict(self, path):
        """
        加载切词词典
        :param path:
        :return:
        """
        word_freq = set()
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
                        word = Converter('zh-hans').convert(word).lower()
                        word_freq.add(word)
                        if len(info) >= 4:
                            self.word_freq_pinyin[word]=info[3].split(',')

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
                        self.word_freq.add(origin)
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
        self.custom_word_freq = self.custom_word_freq.union(word_freqs)
        # 合并切词词典及自定义词典
        self.word_freq = self.word_freq.union(self.custom_word_freq)
        # self.tokenizer = Tokenizer(dict_path=self.word_freq_path, custom_word_freq_dict=self.custom_word_freq,
        #                            custom_confusion_dict=self.custom_confusion)
        # for k, v in word_freqs.items():
        #     self.set_word_frequency(k, v)
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
        #old_new_pose_idx_list 是升序排序的
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

    # def _get_map_size(self, maybe_errors_map):
    #     count = 0
    #     for key in maybe_errors_map:
    #         count += len(maybe_errors_map[key])
    #     return count

    def _add_maybe_error_item(self, maybe_err, maybe_errors_map):
        # old_count = self._get_map_size(maybe_errors_map)
        """
        新增错误
        :param maybe_err:
        :param maybe_errors_map:
        :return:
        """
        begin_idx = 1
        end_idx = 2
        merged_begin_pos = maybe_err[begin_idx]
        merged_end_pos = maybe_err[end_idx]
        former_merged_errs = list()
        former_merged_errs.append(maybe_err)
        changed = True
        merged = False
        while changed:
            changed = False
            keys = []
            keys.extend(maybe_errors_map.keys())
            if keys is None:
                break
            # 查找有重叠的纠错项
            for key in keys:
                poses = key.split("_")
                begin_pos = int(poses[0])
                end_pos = int(poses[1])
                if (merged_begin_pos >= end_pos) or (merged_end_pos <= begin_pos):
                    continue # 位置没有交错
                elif (merged_begin_pos == begin_pos and merged_end_pos == end_pos): #位置一样
                    if not merged: # 当前元素没有合并过，则合并
                        maybe_errors = maybe_errors_map[key]
                        maybe_errors.extend(former_merged_errs)
                        former_merged_errs = maybe_errors
                        merged = True
                        changed = True
                    continue # 已经合并了
                maybe_errors = maybe_errors_map[key]
                #有重叠，将重叠的项合并到list中
                min_begin = min(begin_pos, merged_begin_pos)
                max_end = max(end_pos, merged_end_pos)
                merged_key = str(min_begin) + "_" + str(max_end)
                former_merged_key = str(merged_begin_pos) + "_" + str(merged_end_pos)
                maybe_errors.extend(former_merged_errs)
                if merged_key in maybe_errors_map:
                    if former_merged_key == merged_key: # 之前已合并项和新的已合并项相同
                        maybe_errors_map[merged_key] = maybe_errors
                    elif merged_key != key:
                        maybe_errors_map[merged_key].extend(maybe_errors)
                    else: # former_merged_key != merged_key 并且 merged_key = key
                        # print("err=" + str(maybe_err))
                        # for key in maybe_errors_map:
                        #     print(key + ":" + str(maybe_errors_map[key]))
                        # print('\n')
                        # new_count = self._get_map_size(maybe_errors_map)
                        # if new_count != (old_count +1):
                        #     print("error!!!!!")
                        return  # 新加入的纠错项,key用已经存在的,这个key之前已经考虑过合并，不需要再while循环检查了
                else:
                    # 此时 key != new_key, 因为 new_key 不在 maybe_errors_map，而key在
                    maybe_errors_map[merged_key] = maybe_errors
                if merged_key != key:
                    del maybe_errors_map[key]
                # 更新范围，将和 maybe_err 有重叠的所有 maybe_errors 合并为一个整体
                # 然后看看之前判断和 maybe_errr 没有重叠的项，和合并后的项是否有重叠
                merged_begin_pos = min_begin
                merged_end_pos = max_end
                former_merged_errs = maybe_errors_map[merged_key]
                merged = True
                changed = True
        # 没有找到交错位置的纠错项
        if not merged:
            merged_key = str(maybe_err[begin_idx]) + "_" + str(maybe_err[end_idx])
            maybe_errors_map[merged_key] = [maybe_err]
            # print("err=" + str(maybe_err))
            # for key in maybe_errors_map:
            #     print(key + ":" + str(maybe_errors_map[key]))
            #
            # new_count = self._get_map_size(maybe_errors_map)
            # if new_count != (old_count + 1):
            #     print("error!!!!!")
            # print('\n')

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
        检测句子中的疑似错误信息，包括 [候选词,原词起始位置,原词结束位置,匹配得分,错误类型]
        :param sentence:
        :param sentence_old_start_idx: 当前句子首字在原始text中的位置
        :param former_sentences_size_changed: 前面已修正句子总长度相对原始text的长度变化量
        :return: map{原词起始位置_原词结束位置: [correction_words, begin_pos, end_pos, matched_score, error_type]}
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
                maybe_err = (correction, old_start_idx, old_start_idx + len(confuse), len(confuse), ErrorType.confusion)
                self._add_maybe_error_item(maybe_err, maybe_errors_map)

        sentence_pinyin_map = self.build_sentence_pinyin_map(sentence)
        lcs_match_threshold = 0.7
        ## 根据英文单词的中文谐音，找到匹配的英文单词
        for en_word in self.en_ch_alias:
            for chinese_pinyin in self.en_ch_alias_pinyin[en_word]:
                lcs_info = lcs(sentence_pinyin_map, chinese_pinyin, lcs_match_threshold)
                if lcs_info is not None:
                    begin_idx = lcs_info.get("range")[0]
                    end_idx = lcs_info.get("range")[1]  # 包含这个字
                    matched_score = lcs_info.get('maxMatchedLen') * lcs_info.get('matchedScore')
                    old_start_idx = begin_idx + sentence_old_start_idx
                    old_end_idx = end_idx + sentence_old_start_idx + 1  # +1变成不包含这个字
                    new_start_idx = old_start_idx + former_sentences_size_changed
                    self.add_pose_idx(old_start_idx, new_start_idx)
                    maybe_err = (en_word, old_start_idx, old_end_idx, matched_score, ErrorType.word)
                    self._add_maybe_error_item(maybe_err, maybe_errors_map)
        ## 根据最长公共子序列匹配，找可能出错的热词
        if self.is_word_error_detect:
            for word in self.word_freq:
                word_pinyin = self.word_freq_pinyin[word]
                lcs_info = lcs(sentence_pinyin_map, word_pinyin, lcs_match_threshold)
                if lcs_info is not None:
                    begin_idx = lcs_info.get("range")[0]
                    end_idx = lcs_info.get("range")[1] # 包含这个字
                    matched_score = lcs_info.get('maxMatchedLen') * lcs_info.get('matchedScore')
                    old_start_idx = begin_idx + sentence_old_start_idx
                    old_end_idx = end_idx + sentence_old_start_idx + 1 # +1变成不包含这个字
                    new_start_idx = old_start_idx + former_sentences_size_changed
                    self.add_pose_idx(old_start_idx, new_start_idx)
                    maybe_err = (word, old_start_idx, old_end_idx, matched_score, ErrorType.word)
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
                        for char_similarity in self.hans_similarity_map.get(token):
                            old_start_idx = i + sentence_old_start_idx
                            new_start_idx = old_start_idx + former_sentences_size_changed
                            self.add_pose_idx(old_start_idx, new_start_idx)
                            maybe_err = (char_similarity[0], old_start_idx, old_start_idx + 1, char_similarity[1], ErrorType.word)
                            self._add_maybe_error_item(maybe_err, maybe_errors_map)
            except IndexError as ie:
                logger.warn("index error, sentence:" + sentence + str(ie))
            except Exception as e:
                logger.warn("detect error, sentence:" + sentence + str(e))
        return maybe_errors_map

# detector = Detector()
# maybe_errors_map = {}
# detector.add_maybe_error_item(('wifi', 3, 7, 'word'), maybe_errors_map)
# detector.add_maybe_error_item(('玁', 5, 6, 'word'), maybe_errors_map)
# detector.add_maybe_error_item(('攦', 1, 2, 'word'), maybe_errors_map)
# detector.add_maybe_error_item(('峤', 0, 1, 'word'), maybe_errors_map)
# detector.add_maybe_error_item(('隷', 1, 2, 'word'), maybe_errors_map)
# print('ok')