import sys
import os

def non_digit_char_needed(uchar):
    if ('\u4e00' <= char <= '\u9fa5'):
        return True
    if char == '。' or char == '，' or char == '？' or char == '！' or char == '；' \
            or char == '、' or char == '—' or char == '《' or char == '》' or char == '‘' \
            or char == '’' or char == '“' or char == '”' or char == '+' or char == '-' \
            or char == ';' or char == "'" or char == '"' or char == '?' \
            or char == ',' or char == '：' or char == ':' or char == '（' or char == '）':
        return True
    return False

def split_chinese(string):
    result = ""
    digits = ""
    pre_char = '0'
    for char in string:
        if non_digit_char_needed(char):
            if(len(digits) > 0):
                result += (digits + ' ' + char + ' ')
                digits = ''
            else:
                result += (char + ' ')
        elif ('0' <= char <= '9') or (char == '.'):
            # 前一个字符不是英文字幕
            if '\u0041' > pre_char or '\u005a' < pre_char < '\u0061' or pre_char > '\u007a':
                digits += char
        pre_char = char
    return result

in_path = sys.argv[1]
out_path = in_path[0:in_path.rindex('.')] + '_split.txt'

result = []
if not os.path.exists(in_path):
    logger.warn("file not exists:" + in_path)
    exit(1)
with open(in_path, 'r', encoding='utf-8') as f:
    lines = 0
    for line in f:
        line2 = line.strip()
        line2 = split_chinese(line2)
        result.append(line2)
        lines += 1
        if(lines % 10000) == 0:
            print(str(lines) + " lines processed.")

with open(out_path, 'w', encoding='utf-8') as f:
    for line in result:
        f.write(line + '\n')
