# Author: Antimage
# Date: 23-11-12

import json
import os
from typing import Tuple, List, Any, Optional, Dict
import argparse
import random


class TripletData(object):
    def __init__(self, path, dataset):
        self.data_path = os.path.join(path, dataset)
        self.ent_path = os.path.join(path, dataset, 'entity2id.txt')
        self.rel_path = os.path.join(path, dataset, 'relation2id.txt')
        self.num_nodes = len(open(self.ent_path, 'r').readlines())
        self.num_rels = len(open(self.rel_path, 'r').readlines())

    def load(self, type='train'):
        fp = open(os.path.join(self.data_path, type + '.txt'), 'r')
        res = []

        for line in fp.readlines():
            h, r, t, tim = line.strip().split('\t')
            res.append((int(h), int(r), int(t)))
        self.triples = res
        return None



class DataLoader(object):
    def __init__(self, args, dataset_path, history_data_paths, inference_data_path):
        self.history_data_paths = history_data_paths
        self.inference_data_path = inference_data_path
        self.dataset = dataset_path
        self.args = args

    def load_id_dic(self, path):
        dic = dict()
        fp = open(path, 'r', encoding='utf-8')
        for line in fp.readlines():
            a, b = line.strip().split('\t')
            b = int(b)
            dic.update({b : a})
        return dic
    
    def load_quadruples(self, files, direction='right'):
        assert direction in ['left', 'right']
        search_dic = dict()
        sm = 0
        for file in files:
            fp = open(os.path.join(self.dataset, file), 'r', encoding='utf-8')
            for line in fp.readlines():
                h, r, t, tim = list(map(lambda x : int(x), line.strip().split('\t')))
                if h not in self.entity_dic or t not in self.entity_dic or r not in self.relation_dic:
                    continue
                head, rel, tail = self.entity_dic[h], self.relation_dic[r], self.entity_dic[t]

                if direction == 'right':
                    ## tail batch
                    if head not in search_dic:
                        search_dic.update({head : dict()})            
                    if tim not in search_dic[head]:
                        search_dic[head].update({tim : dict()})
                    if rel not in search_dic[head][tim]:
                        search_dic[head][tim].update({rel : []})
                    search_dic[head][tim][rel].append(tail)
                elif direction == 'left':
                    ## head batch
                    if tail not in search_dic:
                        search_dic.update({tail : dict()})
                    if tim not in search_dic[tail]:
                        search_dic[tail].update({tim : dict()})
                    if rel not in search_dic[tail][tim]:
                        search_dic[tail][tim].update({rel : []})
                    search_dic[tail][tim][rel].append(head)
        return search_dic
    
    def load_test_quadruples(self, ):
        test_samples = list()

        fp = open(os.path.join(self.dataset, self.inference_data_path), 'r', encoding='utf-8')
        for line in fp.readlines():
            h, r, t, tim = list(map(lambda x : int(x), line.strip().split('\t')))
            head, rel, tail = self.entity_dic[h], self.relation_dic[r], self.entity_dic[t]
            test_samples.append((head, rel, tail, tim))
        return test_samples


    def generate_history(self, ):
        self.entity_dic = self.load_id_dic(os.path.join(self.dataset, 'entity2id.txt'))
        self.relation_dic = self.load_id_dic(os.path.join(self.dataset, 'relation2id.txt'))

        self.head_search = self.load_quadruples(self.history_data_paths, 'right')
        self.tail_search = self.load_quadruples(self.history_data_paths, 'left')

        # self.test_data = self.load_test_quadruples('test.txt')
        
        return None

    def update_history(self, timestamp_history):
        ## Maybe delete tail search... it seems not to be used
        for data in timestamp_history:
            h, r, t, ts = data
            if h not in self.head_search:
                self.head_search.update({h : dict()})
            if ts not in self.head_search[h]:
                self.head_search[h].update({ts : dict()})
            if r not in self.head_search[h][ts]:
                self.head_search[h][ts].update({r : []})
            self.head_search[h][ts][r].append(t)
        return None
    
    def search_history(self, ent, rel, history_length, direction='right'):
        ## search_dict[ent][time][rel]: list[ent]
        assert direction in ['right', 'left']
        search_dict = self.head_search if direction == 'right' else self.tail_search
        if ent not in search_dict:
            return []
        ## priority selection of the same schema
        schema_search_history = { k : dict({rel : v[rel]}) for k, v in search_dict[ent].items() if rel in v} # {time : {rel : list_of_tail}}
        # schema_search_history = sorted(schema_search_history.items(), key=lambda x : x[0], reverse=True)
        tot = []
        for k, v in schema_search_history.items():
            for tail in v[rel]:
                tot.append((ent, rel, tail, k))
        tot = sorted(tot, key=lambda x : x[3])
        if len(tot) >= history_length:
            return tot[ -history_length : ]
        leng = len(tot)

        ## padding history by entity itself
        for timestamp in list(sorted(search_dict[ent].keys(), reverse=True)):
            for another_rel in search_dict[ent][timestamp]:
                if another_rel == rel:
                    continue
                for pair_ent in search_dict[ent][timestamp][another_rel]:
                    tot.append((ent, another_rel, pair_ent, timestamp))
                    leng += 1
                    if leng == history_length:
                        break
                if leng == history_length:
                    break
            if leng == history_length:
                break

        ## padding history by relation
        if leng != history_length:
            for another_ent, dic in search_dict.items():
                for timestamp in list(sorted(dic.keys(), reverse=True)):
                    if rel in dic[timestamp]:
                        tot.append((another_ent, rel, dic[timestamp][rel][0], timestamp))
                        leng += 1
                    if leng == history_length:
                        break
                if leng == history_length:
                    break
        
        return tot

    def generate_input(self, x, search_dict, ):
        # generate history for test_sample
        ent, rel, tim = x[0], x[1], x[3]
        history_list = self.search_history(search_dict, ent, rel, self.args.history_length)

        return history_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LLM for TKGC')

    parser.add_argument('--data-path', type=str, default='./data', help='data path')
    parser.add_argument("--dataset", type=str, default='ICEWS14', help='select dataset')
    args = parser.parse_args()

    loader = DataLoader(args, os.path.join(args.data_path, args.dataset), ['train.txt'], 'valid.txt',)
    loader.generate_history()
    import pickle
    pickle.dump(loader.head_search, open('head_search.pkl', 'wb'))
