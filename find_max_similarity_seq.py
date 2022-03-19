import os
import Levenshtein
from pycorrector.utils.text_utils import get_unify_pinyins

pwd_path = os.path.abspath(os.path.dirname(__file__))
same_pinyin_file_path = pwd_path + "/pycorrector/data/same_pinyin_2.txt"
pinyin_set = set()
with open(same_pinyin_file_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line.startswith('#'):
            continue
        parts = line.split('\t')
        if parts and len(parts) >= 1:
            key_char = parts[0]
            pinyin = get_unify_pinyins(key_char)
            pinyin_set.add(pinyin[0])
pinyin_similarity_map = {}
for pinyin1 in pinyin_set:
    pinyin_similarity_map[pinyin1] = dict()
    for pinyin2 in pinyin_set:
        if pinyin2 == pinyin1:
            pinyin_similarity_map[pinyin1][pinyin2] = 1
            continue
        edit_distance = Levenshtein.distance(pinyin1,pinyin2)
        pinyin_similarity_map[pinyin1][pinyin2] = 1 - edit_distance/(max(len(pinyin1),len(pinyin2)))

# get LCS(longest common subsquence),DP
def lcs(sentence_pinyin, words_pinyin, threshold=0.7):
    # 得到一个二维的数组，类似用dp[lena+1][lenb+1],并且初始化为0
    similarities = [[0 for j in range(len(words_pinyin) + 1)] for i in range(len(sentence_pinyin) + 1)]
    direction = [[0 for j in range(len(words_pinyin) + 1)] for i in range(len(sentence_pinyin) + 1)]

    # enumerate(a)函数： 得到下标i和a[i]
    for i, x in enumerate(sentence_pinyin):
        for j, y in enumerate(words_pinyin):
            similarity = pinyin_similarity_map[x][y]
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
    rough_score = similarities_sum/len(words_pinyin)
    if rough_score < threshold:
        return {'matchedScore': rough_score}

    # 到这里已经得到最长的子序列的长度，下面从这个矩阵中就是得到最长子序列
    result = []
    x = len(sentence_pinyin)
    while x != 0:
        y = len(words_pinyin)
        if similarities[x][y] < similarities_sum:
            break
        while direction[x-1][y] == '↖' and similarities[x-1][y] == similarities_sum:
            x -= 1
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
            else: # direction[x][y]='↑'
                x -= 1
        if maxX == -1:
            matched_info = {'matchedScore': 0}
        else:
            matched_score = similarities_sum / max((maxX - minX + 1), len(words_pinyin))
            matched_info = {'maxMatchedLen': similarities_sum, 'matchedScore': matched_score, 'range': [minX, maxX]}
        result.append(matched_info)
    return result


# sentence = "晓峰你好啊啊啊啊小凤"
# sentence = "悄悄峰峰"
# sentence = '小阿峰'
# word = "乔峰"
sentence = '歪发信号'
word = "歪发一"
sentence_pinyin = get_unify_pinyins(sentence)
word_pinyin = get_unify_pinyins(word)
result = lcs(sentence_pinyin, word_pinyin)
print(result)
