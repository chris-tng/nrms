import torch
import torch.nn as nn
import torch.nn.functional as F


# +
def dropout_mask(x, sz, p):
    "Return a dropout mask of the same type as `x`, size `sz`, with probability `p` to nullify an element."
    return x.new(*sz).bernoulli_(1-p).div_(1-p)


class EmbeddingDropout(nn.Module):
    "Apply dropout with probabily `embed_p` to an embedding layer `emb`."

    def __init__(self, emb: nn.Embedding, embed_p: float = 0.):
        super().__init__()
        self.emb = emb
        self.embed_p = embed_p

    def forward(self, words: torch.Tensor, scale: float = None):
        if self.training and self.embed_p != 0:
            size = (self.emb.weight.size(0),1)
            mask = dropout_mask(self.emb.weight.data, size, self.embed_p)
            masked_embed = self.emb.weight * mask
        else: 
            masked_embed = self.emb.weight
        
        if scale: 
            masked_embed.mul_(scale)
        
        return F.embedding(words, masked_embed, 
            self.emb.padding_idx or -1, self.emb.max_norm,
            self.emb.norm_type, self.emb.scale_grad_by_freq, 
            self.emb.sparse)


# -

class AdditiveAttention(nn.Module):
    
    def __init__(self, input_size: int, attention_size: int):
        super().__init__()
        self.proj = nn.Linear(input_size, attention_size, bias=True)
        self.register_parameter("query", nn.Parameter(torch.ones(attention_size, 1)))
        
    def forward(self, x):
        "x of shape (b, n, d)"
        x_proj = self.proj(x).tanh() # (b, n, a)
        attn_w = (x_proj @ self.query).squeeze(-1).softmax(-1) # (b, n)
        attn_out = (attn_w.unsqueeze(-1) * x).sum(1)
        return attn_out


class NRMS(nn.Module):
    
    def __init__(self, glove_emb, embed_p, num_heads=16, head_size=16, attention_size=200):
        super().__init__()
        emb = nn.Embedding.from_pretrained(torch.from_numpy(glove_emb), freeze=False, padding_idx=0)
        self.word_emb = EmbeddingDropout(emb, embed_p=embed_p)
        embed_size = glove_emb.shape[-1]
        
        # News encoder
        self.proj = nn.Linear(in_features=embed_size, out_features=num_heads*head_size, bias=False)
        self.word_self_attn = nn.MultiheadAttention(embed_dim=head_size*num_heads, num_heads=num_heads)
        self.word_add_attn = AdditiveAttention(num_heads*num_heads, attention_size=attention_size)
        
        # User encoder
        self.user_self_attn = nn.MultiheadAttention(embed_dim=head_size*num_heads, num_heads=num_heads)
        self.user_add_attn = AdditiveAttention(num_heads*num_heads, attention_size=attention_size)
        
        
    def forward(self, x, seq_len):
        # (b, n, e)
        seq_emb = self.word_emb(x)
        # (b, n, d)
        seq_emb = self.proj(seq_emb)
        seq_emb = seq_emb.transpose(0, 1)
        
        attn_out, attn_w = self.word_self_attn(
            seq_emb, seq_emb, seq_emb, 
            key_padding_mask=(x == 0)
        )
        
        attn_out = attn_out.transpose(0, 1)
        x_pooled = self.word_add_attn(attn_out)
        
        seq_pooled = x_pooled[:seq_len]
        label_pooled = x_pooled[seq_len:]
        
        seq_pooled = seq_pooled.unsqueeze(0).transpose(0, 1)
        seq_encoded, _ = self.user_self_attn(seq_pooled, seq_pooled, seq_pooled)
        seq_encoded = seq_encoded.transpose(0, 1)
        
        user_encoded = self.user_add_attn(seq_encoded)
        
        return user_encoded, label_pooled
