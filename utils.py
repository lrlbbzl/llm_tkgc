import torch
from torch import Tensor
import numpy as np
import dgl
from collections import defaultdict
import sys

def r2e(triplets, num_rels):
    src, rel, dst = triplets.transpose(0, 1)
    # get all relations
    uniq_r = torch.unique(rel)
    uniq_r = torch.cat((uniq_r, uniq_r+num_rels))
    # generate r2e
    r_to_e = defaultdict(set)
    for j, (src, rel, dst) in enumerate(triplets):
        r_to_e[rel].add(src)
        r_to_e[rel+num_rels].add(src)
    r_len = []
    e_idx = []
    idx = 0
    for r in uniq_r:
        r_len.append((idx,idx+len(r_to_e[r])))
        e_idx.extend(list(r_to_e[r]))
        idx += len(r_to_e[r])
    return uniq_r, r_len, e_idx

def build_graph(num_nodes, num_rels, triples, use_cuda, gpu):
    def compute_deg_norm(g):
        # right norm
        in_degree = g.in_degrees().float()
        in_degree[torch.nonzero(in_degree == 0.).view(-1)] = 1
        norm = 1. / in_degree
        return norm
    triples = torch.LongTensor(triples)
    src, rel, dst = triples.T
    # add reciprocal relation
    src, dst = torch.cat((src, dst)), torch.cat((dst, src))
    rel = torch.cat((rel, rel + num_rels))

    new_data = torch.cat((src.unsqueeze(0), rel.unsqueeze(0), dst.unsqueeze(0)), dim=0).transpose(0, 1)

    # build graph
    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)
    g.add_edges(src, dst)

    norm = compute_deg_norm(g)
    node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    g.ndata.update({'id': node_id, 'norm': norm.view(-1, 1)})
    g.apply_edges(lambda edges: {'norm': edges.dst['norm'] * edges.src['norm']})
    g.edata['type'] = torch.LongTensor(rel)

    uniq_r, r_len, r_to_e = r2e(triples, num_rels)
    g.uniq_r = uniq_r
    g.r_to_e = r_to_e
    g.r_len = r_len
    if use_cuda:
        g.to('cuda')
        g.r_to_e = torch.from_numpy(np.array(r_to_e)).long()
    return g, new_data


def givens_rotation(r, x, transpose=False):
    """Givens rotations.

    Args:
        r: torch.Tensor of shape (N x d), rotation parameters
        x: torch.Tensor of shape (N x d), points to rotate
        transpose: whether to transpose the rotation matrix

    Returns:
        torch.Tensor os shape (N x d) representing rotation of x by r
    """
    givens = r.view((r.shape[0], -1, 2))
    givens = givens / torch.norm(givens, p=2, dim=-1, keepdim=True).clamp_min(1e-15)
    x = x.view((r.shape[0], -1, 2))
    if transpose:
        x_rot = givens[:, :, 0:1] * x - givens[:, :, 1:] * torch.cat((-x[:, :, 1:], x[:, :, 0:1]), dim=-1)
    else:
        x_rot = givens[:, :, 0:1] * x + givens[:, :, 1:] * torch.cat((-x[:, :, 1:], x[:, :, 0:1]), dim=-1)
    return x_rot.view((r.shape[0], -1))


def euc_distance(x: Tensor, y: Tensor, eval_mode=False) -> Tensor:
    """calculate eucidean distance

    Args:
        x (Tensor): shape:(N1, d), the x tensor 
        y (Tensor): shape (N2, d) if eval_mode else (N1, d), the y tensor
        eval_mode (bool, optional): whether or not use eval model. Defaults to False.

    Returns:
        if eval mode: (N1, N2)
        else: (N1, 1)
    """
    x2 = torch.sum(x * x, dim=-1, keepdim=True)
    y2 = torch.sum(y * y, dim=-1, keepdim=True)

    if eval_mode:
        y2 = y2.t()
        xy = x @ y.t()
    else:
        assert x.shape[0] == y.shape[0], 'The shape of x and y do not match.'
        xy = torch.sum(x * y, dim=-1, keepdim=True)

    return x2 + y2 - 2 * xy

def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    print(f"all params num: {all_model_params}, trainable param num: {trainable_model_params}")
    return trainable_model_params


def tokenize(prompt, tokenizer, length_limit, add_eos_token=False):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=length_limit,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < length_limit
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result

def generate_and_tokenize_prompt(data_point, prompter, tokenizer, length_limit, if_test: bool):
    full_prompt = prompter.generate_prompt(
        data_point["query"],
        data_point["response"],
    )

    full_tokenized = tokenize(full_prompt, tokenizer, length_limit, add_eos_token=True)
    user_prompt = prompter.generate_prompt(
        data_point["query"]
    )
    user_tokenized = tokenize(user_prompt, tokenizer, length_limit)
    user_length = len(user_tokenized["input_ids"])
    mask_token = [-100] * user_length
    full_tokenized["labels"] = mask_token + full_tokenized["labels"][user_length : ]
    return full_tokenized
        