import json
import os
dir_path = "./outputs/ICEWS14"
path = os.path.join(dir_path, 'llama_prefix_len10_paged', 'results.json')

x = json.load(open(path, 'r'))

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

acc = 0
for idx, a in enumerate(x):
    prediction, answer = a['predict'], a['answer']
    prediction = prediction[: prediction.find('.')]
    if answer in prediction:
        acc += 1
    # else:
    #     sentence = "The missing entity of query quadruplet is "
    #     substr = prediction[prediction.find(sentence) :]
    #     prediction = substr[len(sentence) : ]
    #     if prediction in answer:
    #         acc += 1
                

print(acc * 1. / len(x))

# mp = dict()
# res = []
# for a, b in zip(x['pred'], x['answer']):
#     res.append({'pred' : a, 'answer' : b})
# json.dump(res, open('res.json', 'w'))