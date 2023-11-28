#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : 23-11-09
# @Author : Antimage

import torch
import argparse
import warnings
from functools import partial
warnings.filterwarnings("ignore")
import logging
from copy import deepcopy
from tqdm import tqdm
logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s',
                    level=logging.DEBUG)
import pickle
import os 
import json
import sys
from typing import List, Tuple
from datasets import load_dataset

from load_data import TripletData, DataLoader
from utils import build_graph, generate_and_tokenize_prompt, print_number_of_trainable_model_parameters
from pretrain_nn import gnn_kge
from prompt import Prompter
from model import KGEAdapterLLM

from torch.optim import AdamW
from torch import nn
from transformers import get_linear_schedule_with_warmup
from transformers import LlamaForCausalLM, LlamaTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from transformers import AutoModelForCausalLM, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
    prepare_model_for_kbit_training
)

import warnings
warnings.filterwarnings("ignore")

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

    use_gnn = 'gcn'
    if args.gcn == False:
        use_gnn = 'none'
    logging.info("Use gcn: {}".format(use_gnn))
    kge_dir = os.path.join(args.save_path, args.dataset)
    if not os.path.exists(kge_dir):
        os.makedirs(kge_dir)
    kge_ent_embs_path = os.path.join(kge_dir, 'entity_emb_{}_{}.pkl'.format(args.score_func, use_gnn))
    kge_rel_embs_path = os.path.join(kge_dir, 'relation_emb_{}_{}.pkl'.format(args.score_func, use_gnn))
    
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
                new_feature = global_model.gnn_forward(args.gcn)
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

        pickle.dump(global_model.ent_embedding, open(kge_ent_embs_path, 'wb'))
        pickle.dump(global_model.rel_embedding, open(kge_rel_embs_path, 'wb'))

    else:
        if os.path.exists(kge_ent_embs_path) and os.path.exists(kge_rel_embs_path):
            pass
        else:
            raise Exception("KGE files {} do not exist!".format(kge_ent_embs_path))



    if args.do_finetune:
        logging.info('*' * 20 + 'Start fine-tuning' + '*' * 20)
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        llm_path = os.path.join(args.base_model_path, args.base_model)
        gradient_accumulation_steps = args.batch_size // args.sm_batch_size

        ## training setting
        device_map = "auto"
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        ddp = world_size != 1
        if ddp:
            device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
            gradient_accumulation_steps = gradient_accumulation_steps // world_size


        ## Prepare data
        data_loader = DataLoader(args, os.path.join(args.data_path, args.dataset), ['train.txt'], 'valid.txt', )
        data_loader.generate_history()
        id2ent, id2rel = data_loader.entity_dic, data_loader.relation_dic
        ### valid data in fine-tune and test data in inference
        test_samples = data_loader.load_test_quadruples()
        val_set_size = args.val_size
        ### dump prompt 
        prompt_save_dir = os.path.join(args.prompt_path, args.dataset)
        if not os.path.exists(prompt_save_dir):
            os.makedirs(prompt_save_dir)
        prompt_save_file = os.path.join(prompt_save_dir, '{}_{}.json'.format(args.base_model, args.history_length))
        template_path = os.path.join(args.template_path, args.base_model + '.json')
        prompter = Prompter(template_path, id2ent, id2rel)
        if os.path.exists(prompt_save_file):
            prompts = json.load(open(prompt_save_file, 'r'))
        else:
            prompts = []
            for sample in tqdm(test_samples):
                h, r, t, ts = sample
                history_list = data_loader.search_history(h, r, args.history_length, 'right')
                if len(history_list) != args.history_length:
                    continue
                prompt = prompter.prepare_prompt((h, r, ts), history_list, response=t)
                prompts.append(prompt)

                if args.add_reciprocal:
                    # TODO
                    pass
            json.dump(prompts, open(prompt_save_file, 'w'))


        ## Tokenize
        tokenizer = AutoTokenizer.from_pretrained(llm_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

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
            # if (
            #     result["input_ids"][-1] != tokenizer.eos_token_id
            #     and len(result["input_ids"]) < length_limit
            #     and add_eos_token
            # ):
            #     result["input_ids"].append(tokenizer.eos_token_id)
            #     result["attention_mask"].append(1)

            result["labels"] = result["input_ids"].copy()

            return result

        def generate_and_tokenize_prompt(data_point):
            full_prompt = prompter.generate_prompt(
                data_point["query"],
                data_point["response"],
            )

            full_tokenized = tokenize(full_prompt, tokenizer, args.truncation_length, add_eos_token=True)
            user_prompt = prompter.generate_prompt(
                data_point["query"]
            )
            user_tokenized = tokenize(user_prompt, tokenizer, args.truncation_length)
            user_length = len(user_tokenized["input_ids"])
            mask_token = [-100] * user_length
            full_tokenized["labels"] = mask_token + full_tokenized["labels"][user_length : ]
            return full_tokenized

        data = load_dataset('json', data_files=prompt_save_file)
        # partial_func = partial(generate_and_tokenize_prompt, prompter=prompter, tokenizer=tokenizer, length_limit=args.truncation_length, if_test=False)
        # train_data = data["train"].shuffle().map(partial_func)
        train_data = data["train"].map(generate_and_tokenize_prompt)
        val_data = None

        
        ## create peft model and trainer
        # if not ddp and torch.cuda.device_count() > 1:
        #     # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        #     model.is_parallelizable = True
        #     model.model_parallel = True

        ## Prepare model
        model = AutoModelForCausalLM.from_pretrained(
            llm_path,
            torch_dtype=torch.float16,
            device_map=device_map,
        )
        ori_p = print_number_of_trainable_model_parameters(model)
        if args.prepare_kbit:
            model = prepare_model_for_kbit_training(model)
        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules,
            lora_dropout=args.lora_dropout,
            bias='none',
            task_type='CAUSAL_LM'
        )
        model = get_peft_model(model, peft_config)
        peft_p = print_number_of_trainable_model_parameters(model)
        logging.info(f'# Trainable parameter \nBefore: {ori_p}\nAfter: {peft_p} \nPercentage: {round(peft_p/ori_p * 100, 2)}')

        if args.add_prefix:
            logging.info("****Add prefix embedding****")
            prefix_model = KGEAdapterLLM(model, args.history_length + 2, (kge_ent_embs_path, kge_rel_embs_path))

        training_args = TrainingArguments(
            per_device_train_batch_size=args.sm_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=500,
            num_train_epochs=args.n_ft_epoch,
            learning_rate=args.lr,
            fp16=True,
            logging_steps=100,
            optim="paged_adamw_32bit",
            save_strategy="steps",
            eval_steps=None,
            save_steps=5000,
            output_dir=args.output_dir,
            save_total_limit=2,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=False,
            report_to='wandb',
            run_name=args.run_name,
        )

        if args.add_prefix:
            trainer = Trainer(
                model=prefix_model,
                train_dataset=train_data,
                eval_dataset=val_data,
                args=training_args,
                data_collator=DataCollatorForSeq2Seq(
                    tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
                ),
            )
        else:
            trainer = Trainer(
                model=model,
                train_dataset=train_data,
                eval_dataset=val_data,
                args=training_args,
                data_collator=DataCollatorForSeq2Seq(
                    tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
                ),
            )
        model.config.use_cache = False

        # old_state_dict = model.state_dict
        # model.state_dict = (
        #     lambda self, *_, **__: get_peft_model_state_dict(
        #         self, old_state_dict()
        #     )
        # ).__get__(model, type(model))

        # import sys
        # if torch.__version__ >= "2" and sys.platform != "win32":
        #     model = torch.compile(model)

        trainer.train()

        model.save_pretrained(args.output_dir)
        if args.add_prefix:
            torch.save(prefix_model.embeddings, os.path.join(args.output_dir, "embeddings.pth"))

def check_args(args):
    if args.dataset not in ['ICEWS14', 'ICEWS18', 'YAGO', 'WIKI']:
        raise Exception("Invalid dataset name.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LLM for TKGC')

    parser.add_argument('--data-path', type=str, default='./data', help='data path')
    parser.add_argument("--dataset", type=str, default='ICEWS14', help='select dataset')
    parser.add_argument("--save-path", type=str, default='./pretrained_emb', help='embedding save path')
    parser.add_argument("--template-path", type=str, default='./templates', help='prompt template path')
    parser.add_argument("--prompt-path", type=str, default='./prompts', help='prompt save path')
    parser.add_argument("--base-model-path", type=str, default='./models', help='base llm')
    parser.add_argument("--base-model", type=str, default='Llama-2-7b-ms', help='base llm')
    parser.add_argument("--gpu", type=int, default=1, help='gpu id')

    # Configure for global KGE
    parser.add_argument("--hidden-size", type=int, default=200, help='hidden size for KGE')
    parser.add_argument("--global-gnn", type=str, default='rgat', help='type of gnn in global graph')
    parser.add_argument("--global-heads", type=int, default=4, help='heads of attention during RGAT')
    parser.add_argument("--global-layers", type=int, default=1, help='numbers of propagation')
    parser.add_argument("--n-global-epoch", type=int, default=200, help='KGE epochs')
    parser.add_argument("--gcn", action='store_true', help='whether use rgcn or some other gnn models during pretraining')
    parser.add_argument("--score-func", type=str, default='RotatE', help='KGE model for optimization')
    parser.add_argument("--kge-lr", type=str, default=1e-4, help='learning rate in KGE phase')
    parser.add_argument("--weight-decay", type=float, default=1e-6, help='weight decay for optimizer')
    parser.add_argument("--kge-batch-size", type=int, default=500, help='batch size in KGE')
    parser.add_argument("--grad-norm", type=float, default=1., help='grad norm during training')
    parser.add_argument("--add-prefix", action='store_true', help='whether add prefix embedding')
    # Configure for phase
    parser.add_argument("--do-pretrain", action='store_true', help='whether pretrain KGE')
    parser.add_argument("--do-finetune", action='store_true', help='whether fine-tuning')

    # Configure for LLM fine-tune
    parser.add_argument("--batch-size", type=int, default=8, help='fine-tuning batch size')
    parser.add_argument("--sm-batch-size", type=int, default=2, help='small batch size')
    parser.add_argument("--n-ft-epoch", type=int, default=2, help='fine-tuning epoch')
    parser.add_argument("--prepare-kbit", action='store_true', help='whether prepare for kbit training')
    parser.add_argument("--lr", type=float, default=2e-5, help='learning rate during fine-tuning')
    parser.add_argument("--truncation-length", type=int, default=1024, help='truncation length limit')
    parser.add_argument("--train-on-inputs", type=bool, default=True, help='whether training on inputs data')
    parser.add_argument("--add-eos-tokens", type=bool, default=False, help='whether adding eos')
    parser.add_argument("--prompt-template", type=str, default='llama', help='prompt template')
    # Configure for lora
    parser.add_argument("--lora-rank", type=int, default=32, help='lora rank')
    parser.add_argument("--lora-alpha", type=int, default=16, help='lora alpha')
    parser.add_argument("--lora-dropout", type=float, default=0.05, help='dropout rate during ft')
    parser.add_argument("--lora-target-modules", type=List[str], default=['q_proj', 'k_proj', 'v_proj', 'o_proj'], help='lora target modules')

    # Configure for other places
    parser.add_argument("--history-length", type=int, default=8, help='history references')
    parser.add_argument("--val-size", type=int, default=0, help='vaild dataset length')
    parser.add_argument("--output-dir", type=str, default='./outputs', help='output dirs')
    parser.add_argument("--logging-dir", type=str, default='./logs', help='logs save dir')
    parser.add_argument("--add-reciprocal", type=bool, default=False, help='whether do reverse reasoning')
    parser.add_argument("--run-name", type=str, default='llama-2-7b', help='tag for checking in wandb')
    args = parser.parse_args()

    check_args(args)

    # start
    run(args)



