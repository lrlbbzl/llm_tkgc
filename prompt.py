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
        self.template = json.load(open(template, 'r'))
        self.id2ent = id2ent
        self.id2rel = id2rel

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

        for history in history_list:
            h, r, t = self.id2ent[history[0]], self.id2rel[history[1]], self.id2ent[history[2]]
            single = "{}: [{}, {}, {}]\n".format(history[3], h, r, t)
            given_history += single

        q = "{}: [{}, {}, ]".format(query[2], self.id2ent(query[0]), self.id2rel(query[1]))
        answer = self.id2ent(answer) if answer is not None else None

        return {
            'instruction' : instruction,
            'input' : given_history + q,
            'output' : answer
        }
    
    def full_prompt(self, instruction, input, output):
        return self.template.format(instruction=instruction, input=input, output=output)
