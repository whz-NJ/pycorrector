import os
from pycorrector.utils.text_utils import get_unify_pinyins
pwd_path = os.path.abspath(os.path.dirname(__file__))
# same_pinyin_path0 = os.path.join(pwd_path, 'same_pinyin.txt')
input_file_path = os.path.join(pwd_path, 'pycorrector/data/word_freq.txt')
output_file_path = os.path.join(pwd_path, 'pycorrector/data/word_freq_pinyin.txt')

with open(input_file_path, 'r', encoding='utf-8') as f:
    lines = []
    for line in f:
        line2 = line.strip()
        if len(line2) == 0:
            continue
        fields = line2.split()
        if len(fields) > 0:
            chinese = fields[0]
            py = get_unify_pinyins(chinese.lower())
            fields.append(','.join(py))
            lines.append(fields)

with open(output_file_path, 'w', encoding='utf-8') as f:
    for line in lines:
        f.write(' '.join(line)+'\n')
