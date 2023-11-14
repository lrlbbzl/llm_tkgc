import json
import os.path as osp
from typing import Union


# class Prompter(object):
#     __slots__ = ("template", "_verbose")

#     def __init__(self, template_name: str = "", verbose: bool = False):
#         self._verbose = verbose
#         if not template_name:
#             # Enforce the default here, so the constructor can be called with '' and will not break.
#             template_name = "alpaca"
#         file_name = osp.join("templates", f"{template_name}.json")
#         if not osp.exists(file_name):
#             raise ValueError(f"Can't read {file_name}")
#         with open(file_name) as fp:
#             self.template = json.load(fp)
#         if self._verbose:
#             print(
#                 f"Using prompt template {template_name}: {self.template['description']}"
#             )

#     def generate_prompt(
#         self,
#         instruction: str,
#         input: Union[None, str] = None,
#         label: Union[None, str] = None,
#     ) -> str:
#         # returns the full prompt from instruction and optional input
#         # if a label (=response, =output) is provided, it's also appended.
#         if input:
#             res = self.template["prompt_input"].format(
#                 instruction=instruction, input=input
#             )
#         else:
#             res = self.template["prompt_no_input"].format(
#                 instruction=instruction
#             )
#         if label:
#             res = f"{res}{label}"
#         if self._verbose:
#             print(res)
#         return res

#     def get_response(self, output: str) -> str:
#         return output.split(self.template["response_split"])[1].strip()


class Prompter(object):
    def __init__(self, template, id2ent, id2rel):
        super(Prompter, self).__init__()
        self.template = json.load(open(template, 'r'))['template']
        print(template)
        self.id2ent = id2ent
        self.id2rel = id2rel
        self.ent2id = {v : k for k, v in id2ent.items()}
        self.rel2id = {v : k for k, v in id2rel.items()}

    def prepare_prompt(self, query, history_list, answer=None):
        """
        args:
            query: query tuple (ent, rel, timestamp)
            history_list: list of tuple (h, r, t, timestamp)
            answer: used during fine-tuning, target ent
        """
        instruction = "You must correctly predict the next {object} from a given contexts consisting of multiple quadruplets in the form of \
{time}: [{subject}, {relation}, {object}] and a query in the form of {time}: [{subject}, {relation}, ]. Please directly giving the answer."  

        history_list = sorted(history_list, key=lambda x : x[3])
        given_history = "" 
        embedding_ids = [self.ent2id[query[0]], self.rel2id[query[1]], ]

        for history in history_list:
            h, r, t = history[0], history[1], history[2]
            single = "{}: [{}, {}, {}]\n".format(history[3], h, r, t)
            given_history += single
            embedding_ids.append(self.ent2id[history[2]])

        q = "{}: [{}, {}, ]".format(query[2], query[0], query[1])
        answer = answer if answer is not None else None

        

        return {
            'instruction' : instruction,
            'input' : given_history + q,
            'output' : answer,
            'embedding_ids' : embedding_ids
        }
    
    def full_prompt(self, instruction, input, output):
        return self.template.format(instruction=instruction, input=input, output=output)
