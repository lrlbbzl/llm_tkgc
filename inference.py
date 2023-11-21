import os 
import argparse
from functools import partial
from tqdm import tqdm
from datasets import load_dataset
import json
import torch
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s',
                    level=logging.DEBUG)

from torch import nn
from transformers import get_linear_schedule_with_warmup
from transformers import LlamaForCausalLM, LlamaTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

from utils import generate_and_tokenize_prompt
from load_data import DataLoader 
from prompt import Prompter
import warnings
warnings.filterwarnings('ignore')

from peft import (
    PeftModel,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)


def inference(args):
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_map = "auto"
    

    base_model_path = os.path.join(args.base_model_path, args.base_model)
    # kg_embedding = torch.load(args.kge_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = AutoModelForCausalLM.from_pretrained(base_model_path,
                                             torch_dtype=torch.float16, device_map=device_map).to(device)
    model = PeftModel.from_pretrained(model, args.lora_weights_path, torch_dtype=torch.float16).to(device)
    # model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    # model.config.bos_token_id = 1
    # model.config.eos_token_id = 2
    model.eval()


    # Prepare things such as prompts list for test data;
    ## During fine-tuning, we select valid dataset as test set and create history in train dataset
    ## During testing here, we evaluate on test dataset and create history in train + valid dataset
    data_loader = DataLoader(args, os.path.join(args.data_path, args.dataset), ['train.txt', 'valid.txt'], 'test.txt', )
    data_loader.generate_history()
    id2ent, id2rel = data_loader.entity_dic, data_loader.relation_dic
    ### valid data in fine-tune and test data in inference
    test_samples = data_loader.load_test_quadruples()
    val_set_size = 0
    ### dump prompt 
    prompt_save_file = os.path.join(args.prompt_path, args.dataset, args.base_model + '_test.json')
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

    # data = load_dataset('json', data_files=prompt_save_file)
    # partial_func = partial(generate_and_tokenize_prompt, prompter=prompter, tokenizer=tokenizer, length_limit=args.truncation_length, if_test=False)
    # test_data = data["train"].shuffle().map(partial_func)
    result = []
    for test_sample in tqdm(prompts):
        query, answer, ids = test_sample['query'], test_sample['response'], test_sample['embedding_ids']
        ids = torch.LongTensor(ids).reshape(1, -1).to(device)
        # prefix = kg_embedding(ids)
        prompt = prompter.generate_prompt(query)
        model_inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = model_inputs.input_ids.to(device)
        # token_embeds = model.model.model.embed_tokens(input_ids)
        # ## float16 for Half-float inference
        # if args.half:
        #     prefix = torch.tensor(prefix.clone(), dtype=torch.float16)
        # input_embeds = torch.cat((prefix, token_embeds), dim=1)
        # attention_mask = torch.ones_like(input_embeds)
        generation_config = GenerationConfig(
            do_sample=True,
            temperature=0.9,
            # top_p=0.92,
            # top_k=100,
            num_beams=4, # beam search
            # no_repeat_ngram_size=2
        )
        generate_ids = model.generate(
            input_ids=input_ids,
            # inputs_embeds=input_embeds, 
            generation_config=generation_config,
            return_dict_in_generate=True,
            max_new_tokens=100,
        )

        inputs_text = tokenizer.decode(input_ids[0])
        output = tokenizer.decode(generate_ids.sequences[0]).replace(inputs_text, "")
        result.append(
            {
                "answer": answer,
                "predict": output
            }
        )
    json.dump(result, open(os.path.join(args.output_dir, 'results.json'), 'w'))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TKGC inference')
    
    parser.add_argument('--data-path', type=str, default='./data', help='data path')
    parser.add_argument("--dataset", type=str, default='ICEWS14', help='select dataset')
    parser.add_argument("--save-path", type=str, default='./pretrained_emb', help='embedding save path')
    parser.add_argument("--template-path", type=str, default='./templates', help='prompt template path')
    parser.add_argument("--prompt-path", type=str, default='./prompts', help='prompt save path')
    parser.add_argument("--base-model-path", type=str, default='./models/modelscope', help='base llm')
    parser.add_argument("--base-model", type=str, default='Llama-2-7b-ms', help='base llm')
    parser.add_argument("--gpu", type=int, default=1, help='gpu id')
    parser.add_argument("--lora-weights-path", type=str, default='./outputs', help='lora save path')
    parser.add_argument("--kge-path", type=str, default='./outputs/embeddings.pth', help='kge save path')
    parser.add_argument("--half", type=bool, default=False, help='half precision')

    parser.add_argument("--history-length", type=int, default=8, help='history references')
    parser.add_argument("--output-dir", type=str, default='./outputs', help='output dirs')
    parser.add_argument("--add-reciprocal", type=bool, default=False, help='whether do reverse reasoning')
    
    args = parser.parse_args()
    inference(args)
