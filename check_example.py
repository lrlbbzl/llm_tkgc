import json
import os

output_path = './outputs/YAGO'
data_path = './prompts/YAGO'
prediction = json.load(open(os.path.join(output_path, 'llama_float16_paged_nokbit/results.json'), 'r'))
data = json.load(open(os.path.join(data_path, 'Llama-2-7b-ms_test.json'), 'r'))

while True:
    x = int(input())
    ans, pred = prediction[x]['answer'], prediction[x]['predict']
    single_data = data[x]['query']
    print('*' * 20)
    print(single_data)
    print('*' * 20)
    print(ans)
    print('*' * 20)
    print(pred)
    print()