import json

# fp = open('relation2id.txt', 'r')
# fs = 0
# lines = []
# for line in fp.readlines():
#     a, b = line.strip().split('\t')
#     st, ed = a.find('<'), a.rfind('>')
#     a = a[st + 1 : ed]
#     lines.append("{}\t{}".format(a, b))
# p = open('new_relation2id.txt', 'w')
# p.writelines('\n'.join(lines))

files = ['train.txt', 'valid.txt', 'test.txt']
for file in files:
    fp = open(file, 'r')
    save_file = file
    strs = []
    for line in fp.readlines():
        a, b, c, d, e = line.strip().split('\t')
        strs.append('{}\t{}\t{}\t{}'.format(a, b, c, d))
    p = open(save_file, 'w')
    p.writelines('\n'.join(strs))
    p.close()