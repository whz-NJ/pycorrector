import sys
import os

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
        sentence = ""
        for ch in line2:
            if ch == '，' or ch == '。':
                if len(sentence) > 0:
                    result.append(sentence)
                sentence=''
            elif ch == ' ' or ch == '\t':
                continue
            else:
                sentence += ch

with open(out_path, 'w', encoding='utf-8') as f:
    count = 0
    for line in result:
        f.write("'" + line + "',\n")
