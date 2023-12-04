import json
import os.path as osp
from typing import Union
import re


class Prompter(object):
    def __init__(self, template, id2ent, id2rel):
        super(Prompter, self).__init__()
        template = json.load(open(template, 'r'))
        self.query_template = template['query']
        self.response_template = template['response']
        self.id2ent = id2ent
        self.id2rel = id2rel
        self.ent2id = {v : k for k, v in id2ent.items()}
        self.rel2id = {v : k for k, v in id2rel.items()}

    def prepare_prompt(self, query, history_list, response=None):
        """
        args:
            query: query tuple (ent, rel, timestamp)
            history_list: list of tuple (h, r, t, timestamp)
            response: used during fine-tuning, target ent
        """
        history_list = sorted(history_list, key=lambda x : x[3])
        given_history = "History:\n" 

        for history in history_list:
            h, r, t = history[0], history[1], history[2]
            single = "{}: [{}, {}, {}]\n".format(history[3], h, r, t)
            given_history += single

        q = "\nQuery:\n{}: [{}, {}, ]\n".format(query[2], query[0], query[1])
        response = response if response is not None else None

        return {
            'query' : given_history + q,
            'response' : response,
        }
    
    def full_prompt(self, query, response):
        pattern1, pattern2 = r'\{query\}', r'\{response\}'
        res1 = re.sub(pattern1, query, self.query_template)
        res2 = re.sub(pattern2, response, self.response_template)
        return res1, res2

    def test_prompt(self, query):
        pattern1 = r'\{query\}'
        res = re.sub(pattern1, query, self.query_template)
        return res
    
    def concat_prompt(self, query, response):
        pattern1, pattern2 = r'\{query\}', r'\{response\}'
        res1 = re.sub(pattern1, query, self.query_template)
        res2 = re.sub(pattern2, response, self.response_template)
        return res1 + res2
    
    def generate_prompt(self, input, label=None):
        # schema1 = r"\{query\}"
        # res = re.sub(schema1, input, self.query_template)
        # if label is not None:
        #     schema2 = r"\{response\}"
        #     label = re.sub(schema2, label, self.response_template)
        #     res = f"{res}{label}"
        res = self.query_template.replace("{query}", input)
        if label is not None:
            label = self.response_template.replace("{response}", label)
            res = f"{res}{label}"
        return res