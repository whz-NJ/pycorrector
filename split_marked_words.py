import sys
import os
import re
import jieba

sentence_ending_marks_str = "。，？！；：,?!;" #这些符号标识句子结束
sentence_ending_marks = set(m for m in sentence_ending_marks_str)
ignore_marks_str = "、—《》‘’\"“”（）{} []\t" # 这些符号不用显示在字幕里，讯飞也不会转写出这些字符
ignore_marks = set(m for m in ignore_marks_str)

mark_pattern_str = r'/[A-Za-z]+[0-9]*' #词性标注模式
mark_pattern = re.compile(mark_pattern_str)
if mark_pattern.match('abc'):
    print("错误的正则表达式")
    exit(1)
if not mark_pattern.match('/abc'):
    print("错误的正则表达式")
    exit(1)

def split_words(sentence):
    non_chinese = ''
    words = []
    sentence = sentence.lower() #将可能存在的英文单词转换为小写
    for uchar in sentence:
        if uchar in ignore_marks:
            if len(non_chinese) > 0:
                non_chinese = re.sub(mark_pattern_str, '',non_chinese) #删除词性标注
                if len(non_chinese) > 0:
                    non_chinese = non_chinese.strip('.') #数字后的.不需要，只有数字中间的.才需要
                    non_chinese = non_chinese.replace('/', ''); # 字符 / 在讯飞转写结果中不会出现
                    words.append(' ') #防止前后2个英文单词连一起了
                    words.append(non_chinese)
                non_chinese = ''
            continue  # 忽略这些字符
        if ('\u4e00' <= uchar <= '\u9fa5') or uchar in sentence_ending_marks:  #汉字或句子结束
            if len(non_chinese) > 0:
                non_chinese = re.sub(mark_pattern_str, '',non_chinese)
                if len(non_chinese) > 0:
                    non_chinese = non_chinese.strip('.') #数字后的.不需要，只有数字中间的.才需要
                    non_chinese = non_chinese.replace('/', '')  # 字符 / 在讯飞转写结果中不会出现
                    words.append(' ')  # 防止前后2个英文单词连一起了
                    words.append(non_chinese)
                non_chinese = ''
            words.append(uchar)
        elif ('0' <= uchar <= '9') or (uchar == '.' or ('a' <= uchar <= 'z') or ('A' <= uchar <= 'Z')) \
              or '+' == uchar or '/' == uchar or '-' == uchar or '%' == uchar:
            non_chinese += uchar
    if len(non_chinese) > 0:
        non_chinese = re.sub(mark_pattern_str, '',non_chinese)
        if len(non_chinese) > 0:
            non_chinese = non_chinese.strip('.')  # 数字后的.不需要，只有数字中间的.才需要
            non_chinese = non_chinese.replace('/', '')  # 字符 / 在讯飞转写结果中不会出现
            words.append(' ')  # 防止前后2个英文单词连一起了
            words.append(non_chinese)
    if len(words) == 0:
        return []
    last_word = words[-1]
    if len(last_word) == 0:
        return []
    last_char = last_word[-1]
    if last_char not in sentence_ending_marks: #添加句结束标识
        words.append('。')
    return jieba.lcut("".join(words))
print(split_words('接/v “/w I/x -/w station/x ”/w Wifi/x 热'))

in_path = sys.argv[1]
out_path = in_path[0:in_path.rindex('.')] + '_split.txt'

result = []
if not os.path.exists(in_path):
    print("file not exists:" + in_path)
    exit(1)
with open(in_path, 'r', encoding='utf-8') as f:
    lines = 0
    for line in f:
        line2 = line.strip()
        line2 = split_words(line2)
        result.append(' '.join(line2))
        lines += 1
        if(lines % 10000) == 0:
            print(str(lines) + " lines processed.")

with open(out_path, 'w', encoding='utf-8') as f:
    for line in result:
        f.write(line + '\n')

