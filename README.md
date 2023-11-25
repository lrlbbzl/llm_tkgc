# Knowledge-injection Assists LLM Instruction-tuning For Temporal Knowledge Graph Completion

## Description

Dirs

```bash
./
	--logs 日志文件夹
	--models 基座模型存放
	--outputs 微调后模型存储
	--pretrained_emb 预训练图向量存储
	--prompts 历史挖掘等数据准备的存放，用于放入prompt template
	--templates 接收query和response生成prompt
```

Files

```bash
./
	--main.py 主文件，图向量pretraining和模型fine-tune
	--inference.py 推理
	--pretrain_nn.py 类GCN与KGE模型，用于完成静态拼接大图的向量训练
	--prompt.py prompter, 对应设计在template
	--model.py KGE向量注入模型
	--utils.py 一些计算和tokenize相关函数
```

## Parameters

Fine-tune

```bash
--n-global-epoch 全局图训练轮数
--use-gnn 是否使用类GCN辅助KGE
--hidden-size 图embedding的hidden size
--do-pretrain 是否做向量预训练
--do-finetune
--batch-size fine-tune的batch_size
--sm-batch-size gradient accumulation的small batch size
--lora-rank
--lora-dropout
--n-ft-epoch 微调轮数
--base-model-path 基座模型路径
--base-model 基座模型名
--run-name wandb logging名
--output-dir 输出模型保存路径
```

Inference

```bash
--half kge半精
--output-dir 推理结果存放路径
——lora-weights-path ft的output-dir
```

## Run

Fine-tune

```bash
nohup bash run_llama.sh > ./logs/llama2_7b_prefixadded_float16_adamwhf.log &
```

Inference

```bash
nohup bash inference.sh > ./logs/llama2_7b_prefixadded_float16_adamwhf_infer.log &
```

