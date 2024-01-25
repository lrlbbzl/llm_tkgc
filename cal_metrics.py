import json
import os
dir_path = "./outputs/ICEWS05-15"
path = os.path.join(dir_path, 'llama_1b_len20_bi_aug', 'results_right.json')
sentence = "The missing entity of query quadruplet is "

y = json.load(open(path, 'r'))
leng = len(y)

def eq(a, b):
    index_a = 0
    index_b = 0
    while index_b < len(b):
        if a[index_a] == b[index_b]:
            index_a += 1
            if index_a == len(a):
                return True
        
        index_b += 1
    
    return index_a == len(a)

acc, gap = 0, 5000
for id in range(0, leng, gap):
    x = y[id : id + gap]
    temp_acc = 0
    for idx, a in enumerate(x):
        prediction, answer = a['predict'], a['answer']
        st = prediction.find(sentence) + len(sentence)
        prediction = prediction[st : ]
        if prediction.find('.\n') != -1:
            prediction = prediction[:prediction.find('.\n')]
        elif prediction.find('.</s>') != -1:
            prediction = prediction[:prediction.find('.</s>')]
        else:
            prediction = prediction[:prediction.find('.')]
        if answer == prediction:
            acc += 1
            temp_acc += 1
    print("{}-{}: {}".format(id, min(id + gap, leng), temp_acc / (min(id + gap, leng) - id)))

print(acc * 1. / len(y))
print(len(y))

# mp = dict()
# res = []
# for a, b in zip(x['pred'], x['answer']):
#     res.append({'pred' : a, 'answer' : b})
# json.dump(res, open('res.json', 'w'))