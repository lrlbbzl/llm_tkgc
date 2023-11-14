import torch
from torch import nn
import pickle
import torch.nn.functional as F

from transformers import LlamaForCausalLM
from typing import Tuple, Optional, List

class KGEmbedding(nn.Module):
    def __init__(self, ent_path, rel_path, dim_model, num_prefix):
        super(KGEmbedding, self).__init__()
        
        self.dim_model = dim_model
        self.emb_path = [ent_path, rel_path]
        self.num_prefix = num_prefix

        self.ent_emb = nn.Embedding.from_pretrained(pickle.load(open(self.emb_path[0], 'rb')).weight)
        self.rel_emb = nn.Embedding.from_pretrained(pickle.load(open(self.emb_path[1], 'rb')).weight)

        self.kge_dim = self.ent_emb.weight.shape[1] # kge dimension
        # kge prefix is frozen
        self.ent_emb.requires_grad_(False)
        self.rel_emb.requires_grad_(False)

        self.adapter_fc = nn.Linear(self.kge_dim, self.dim_model)

    def forward(self, ls):
        """
        args:
            ls: embedding space
        """
        ls = torch.LongTensor(ls) # (bs, num_prefix)
        ent_idx = ls[:, torch.cat([torch.LongTensor([0]), torch.arange(2, self.num_prefix)])]
        rel_idx = ls[:, torch.LongTensor([1])]
        ent_embs = self.ent_emb(ent_idx) # (bs, num_prefix - 1, kge_dim)
        rel_embs = self.rel_emb(rel_idx) # (bs, 1, kge_dim)
        
        embs = torch.cat((ent_embs[:, :1, :], rel_embs, ent_embs[:, 1:, :]), dim=1) # (bs, num_prefix, kge_dim)
        prefix_emb = self.adapter_fc(embs) # (bs, num_prefix, dim_model)
        return prefix_emb


class KGEAdapterLLM(nn.Module):
    def __init__(
        self,
        model: LlamaForCausalLM,
        num_prefix: int,
        pretrain_emb_path: Tuple
    ) -> None:
        super(KGEAdapterLLM, self).__init__()
        self.llama_model = model
        self.embeddings = KGEmbedding(
            ent_path=pretrain_emb_path[0],
            rel_path=pretrain_emb_path[1],
            dim_model=4096,
            num_prefix=num_prefix
        )
        
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        embedding_ids: torch.LongTensor = None
    ):
        kg_embeds = self.embeddings(embedding_ids) # (bs, 2 + history_length, kge_size)
        batch_size, seq_len, _ = kg_embeds.shape
        token_embeds = self.llama_model.model.model.embed_tokens(input_ids)
        input_embeds = torch.cat((kg_embeds, token_embeds), dim=1)
        prefix_mask = torch.ones((batch_size, seq_len))
        prefix_labels = torch.full((batch_size, seq_len), fill_value=-100, dtype=torch.long)
        new_attention_mask = torch.cat((prefix_mask.cuda(), attention_mask), dim=-1)
        new_labels = torch.cat((prefix_labels.cuda(), labels), dim=-1)
        return self.llama_model(
            input_ids=None,
            attention_mask=new_attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=input_embeds,
            labels=new_labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

