#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : 23-11-09
# @Author : Antimage

import torch
import argparse
import warnings
warnings.filterwarnings("ignore")
import logging
from copy import deepcopy
from tqdm import tqdm
logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s',
                    level=logging.DEBUG)
import pickle
import os 
from typing import List, Tuple

from load_data import TripletData, DataLoader
from utils import build_graph
from pretrain_nn import gnn_kge
from prompt import Prompter
from model import KGEAdapterLLM

from torch.optim import AdamW
from torch import nn
from transformers import get_linear_schedule_with_warmup
from transformers import LlamaForCausalLM, LlamaTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)


def run(args):
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_data_loader = TripletData(args.data_path, args.dataset)
    train_data_loader.load('train')
    train_data = train_data_loader.triples
    num_nodes, num_rels = train_data_loader.num_nodes, train_data_loader.num_rels
    g, train_data = build_graph(num_nodes, num_rels, train_data, use_cuda, args.gpu)
    g = g.to(device)
    train_data.to(device)
    
    if args.do_pretrain:
        logging.info('*' * 20 + 'Start pretraining' + '*' * 20)
        global_model = gnn_kge(g, num_nodes, num_rels, args.hidden_size, args.score_func, args.global_layers,
                                args.global_heads, args.global_gnn).to(device)
        kge_optimizer = AdamW(global_model.parameters(), lr=args.kge_lr, weight_decay=args.weight_decay)
        loss_fn = nn.CrossEntropyLoss(reduction='mean', weight=None)
        global_step = (len(train_data) // args.kge_batch_size) * args.n_global_epochs if len(train_data) % args.kge_batch_size == 0 \
                            else (len(train_data) // args.kge_batch_size + 1) * args.n_global_epoch
        kge_scheduler = get_linear_schedule_with_warmup(optimizer=kge_optimizer, num_warmup_steps=100, num_training_steps=global_step)
        logging.info('Data size: {}, batch size: {}, training epoch: {}.'.format(len(train_data), args.kge_batch_size, args.n_global_epoch))
        for epoch in range(args.n_global_epoch):
            samples, length = deepcopy(train_data).to(device), len(train_data)
            losses = []
            samples = samples[torch.randperm(samples.shape[0]), :]
            iters = int(length // args.kge_batch_size) + 1 if length % args.kge_batch_size != 0 else length // args.kge_batch_size
            for step in tqdm(range(iters)):
                new_feature = global_model.gnn_forward()
                batch_data = samples[args.kge_batch_size * step : min(length, args.kge_batch_size * (step + 1))]
                score = global_model(batch_data, new_feature)
                loss = loss_fn(score, batch_data[:, 2])

                losses.append(loss)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(global_model.parameters(), args.grad_norm)  # clip gradients
                kge_optimizer.step()
                kge_optimizer.zero_grad()
                kge_scheduler.step()

            logging.info("Epoch {:04d} in static KGE | Ave Loss: {:.4f} ".format(epoch, sum(losses) / len(losses)))

        pickle.dump(global_model.ent_embedding, open(os.path.join(args.save_path, 'entity_emb_gnn_{}.pkl'.format(args.score_func)), 'wb'))
        pickle.dump(global_model.rel_embedding, open(os.path.join(args.save_path, 'relation_emb_gnn_{}.pkl'.format(args.score_func)), 'wb'))

    else:
        #TODO
        # Check the file and load
        pass

    if args.do_finetune:
        logging.info('*' * 20 + 'Start fine-tuning' + '*' * 20)

        gradient_accumulation_steps = args.batch_size // args.sm_batch_size

        # training setting
        device_map = "auto"
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        ddp = world_size != 1
        if ddp:
            device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
            gradient_accumulation_steps = gradient_accumulation_steps // world_size

        model = LlamaForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.float32,
            device_map=device_map
        )
        
        tokenizer = LlamaTokenizer.from_pretrained(args.base_model)

        tokenizer.pad_token_id = (0)
        tokenizer.padding_side = 'left'

        prompter = Prompter(args.prompt_template)

        def tokenize(prompt, add_eos_token=True):
            # there's probably a way to do this with the tokenizer settings
            # but again, gotta move fast
            result = tokenizer(
                prompt,
                truncation=True,
                max_length=args.truncation_length,
                padding=False,
                return_tensors=None,
            )
            if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < args.truncation_length
                and add_eos_token
            ):
                result["input_ids"].append(tokenizer.eos_token_id)
                result["attention_mask"].append(1)

            result["labels"] = result["input_ids"].copy()

            return result

        def generate_and_tokenize_prompt(data_point):
            full_prompt = prompter.generate_prompt(
                data_point["instruction"],
                data_point["input"],
                data_point["output"],
            )
            tokenized_full_prompt = tokenize(full_prompt)
            if not args.train_on_inputs:
                user_prompt = prompter.generate_prompt(
                    data_point["instruction"], data_point["input"]
                )
                tokenized_user_prompt = tokenize(
                    user_prompt, add_eos_token=args.add_eos_token
                )
                user_prompt_len = len(tokenized_user_prompt["input_ids"])

                if args.add_eos_token:
                    user_prompt_len -= 1

                tokenized_full_prompt["labels"] = [
                    -100
                ] * user_prompt_len + tokenized_full_prompt["labels"][
                    user_prompt_len:
                ]  # could be sped up, probably
            return tokenized_full_prompt
        
    
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules,
            lora_dropout=args.dropout,
            bias='none',
            task='CAUSAL_LM'
        )

        model = get_peft_model(model, lora_config)
        load_ent_path = os.path.join(args.save_path, 'entity_emb_{}.pkl'.format(args.score_func))
        load_rel_path = os.path.join(args.save_path, 'relation_emb_{}.pkl'.format(args.score_func))
        prefix_added_lora_model = KGEAdapterLLM(model, args.num_prefix, (load_ent_path, load_rel_path))

        if not ddp and torch.cuda.device_count() > 1:
            model.is_parallelizable = True
            model.model_parallel = True


        ## Prepare data
        data_loader = DataLoader(args, os.path.join(args.data_path, args.dataset), ['train.txt'], 'valid.txt', )
        data_loader.generate_history()
        
        # valid data in fine-tune and test data in inference
        test_samples = data_loader.load_test_quadruples()

        val_set_size = args.val_size
        trainer = Trainer(
            model=prefix_added_lora_model,
            train_dataset=train_data,
            args=TrainingArguments(
                per_device_train_batch_size=args.sm_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps=100,
                num_train_epochs=args.n_ft_epoch,
                learning_rate=args.lr,
                fp16=True,
                logging_steps=10,
                optim="adamw_hf",
                evaluation_strategy="steps" if val_set_size > 0 else "no",
                save_strategy="steps",
                eval_steps=None,
                save_steps=5000,
                output_dir=args.output_dir,
                save_total_limit=2,
                load_best_model_at_end=True if val_set_size > 0 else False,
                ddp_find_unused_parameters=False if ddp else None,
                group_by_length=False,
                report_to=None,
                run_name=None,
            ),
            data_collator=DataCollatorForSeq2Seq(
                tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
        )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LLM for TKGC')

    parser.add_argument('--data_path', type=str, default='./data', help='data path')
    parser.add_argument("--dataset", type=str, default='ICEWS14', help='select dataset')
    parser.add_argument("--save_path", type=str, default='./pretrained_emb', help='embedding save path')
    parser.add_argument("--gpu", type=int, default=1, help='gpu id')

    # Configure for global KGE
    parser.add_argument("--hidden-size", type=int, default=200, help='hidden size for KGE')
    parser.add_argument("--global-gnn", type=str, default='rgat', help='type of gnn in global graph')
    parser.add_argument("--global-heads", type=int, default=4, help='heads of attention during RGAT')
    parser.add_argument("--global-layers", type=int, default=2, help='numbers of propagation')
    parser.add_argument("--n-global-epoch", type=int, default=200, help='KGE epochs')
    parser.add_argument("--score-func", type=str, default='RotatE', help='KGE model for optimization')
    parser.add_argument("--kge-lr", type=str, default=1e-4, help='learning rate in KGE phase')
    parser.add_argument("--weight-decay", type=float, default=1e-6, help='weight decay for optimizer')
    parser.add_argument("--kge-batch-size", type=int, default=500, help='batch size in KGE')
    parser.add_argument("--grad-norm", type=float, default=1., help='grad norm during training')
    # Configure for phase
    parser.add_argument("--do-pretrain", action='store_true', help='whether pretrain KGE')
    parser.add_argument("--do-finetune", action='store_true', help='whether fine-tuning')

    # Configure for LLM fine-tune
    parser.add_argument("--base-model", type=str, default='Llama-7b', help='base llm')
    parser.add_argument("--batch-size", type=int, default=8, help='fine-tuning batch size')
    parser.add_argument("--sm-batch-size", type=int, default=8, help='small batch size')
    parser.add_argument("--n-ft-epoch", type=int, default=2, help='fine-tuning epoch')
    parser.add_argument("--lr", type=float, default=1e-4, help='learning rate during fine-tuning')
    parser.add_argument("--truncation-length", type=int, default=256, help='truncation length limit')
    parser.add_argument("--train-on-inputs", type=bool, default=True, help='whether training on inputs data')
    parser.add_argument("--add-eos-tokens", type=bool, default=False, help='whether adding eos')
    parser.add_argument("--prompt-template", type=str, default='llama', help='prompt template')
    # Configure for lora
    parser.add_argument("--lora-rank", type=int, default=16, help='lora rank')
    parser.add_argument("--lora-alpha", type=int, default=16, help='lora alpha')
    parser.add_argument("--lora-dropout", type=float, default=0.1, help='dropout rate during ft')
    parser.add_argument("--lora-target-modules", type=List[str], default=['q_proj', 'v_proj'], help='lora target modules')

    # Configure
    parser.add_argument("--history-length", type=int, default=10, help='history references')
    parser.add_argument("--val-size", type=int, default=0, help='vaild dataset length')
    parser.add_argument("--output-dir", type=str, default='./outputs', help='output dirs')
    args = parser.parse_args()
    

    # start
    run(args)



