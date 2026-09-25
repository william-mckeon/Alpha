"""Factorized tiktoken embeddings and projections on the single shared transformer."""
import torch
from torch import nn
import torch.nn.functional as F

CHOICES=('silence','listen','pause','resume','replay','express')

class LanguageAdapter(nn.Module):
    def __init__(self,dim,vocab,text_dim=128):
        super().__init__()
        self.embedding=nn.Embedding(vocab,text_dim)
        nn.init.normal_(self.embedding.weight,std=.02)
        self.input=nn.Linear(text_dim,dim,bias=False)
        self.output=nn.Linear(dim,text_dim,bias=False)
        self.choice=nn.Linear(4,len(CHOICES))
        nn.init.zeros_(self.choice.weight);nn.init.zeros_(self.choice.bias)

    def forward(self,core,tokens,last_only=False):
        hidden=core.trunk_embedded(self.input(self.embedding(tokens)))
        if last_only: hidden=hidden[:, -1:]
        return F.linear(self.output(hidden),self.embedding.weight)

    def loss(self,core,tokens,labels,residual,chunk_size=32):
        from torch.utils.checkpoint import checkpoint
        hidden=core.trunk_embedded(self.input(self.embedding(tokens)))
        projected=self.output(hidden)+residual[:,None]
        flat=projected.reshape(-1,projected.shape[-1]); targets=labels.reshape(-1)
        count=(targets != -100).sum()
        if not bool(count): raise ValueError('No language targets')
        def chunk_loss(values,weight,target):
            return F.cross_entropy(F.linear(values,weight),target,ignore_index=-100,reduction='sum')
        total=flat.new_zeros(())
        for start in range(0,len(targets),chunk_size):
            # Recompute vocabulary logits in backward instead of retaining every chunk.
            total=total+checkpoint(chunk_loss,flat[start:start+chunk_size],self.embedding.weight,
                                   targets[start:start+chunk_size],use_reentrant=False)
        return total/count

    def cached_logits(self,core,tokens,cache):
        hidden=core.trunk_cached(self.input(self.embedding(tokens)),cache)
        return F.linear(self.output(hidden[:,-1]),self.embedding.weight)

    def decide(self,features,allowed):
        device=self.embedding.weight.device
        logits=self.choice(torch.tensor(features,dtype=torch.float32,device=device))
        logits=logits.masked_fill(~torch.tensor(allowed,device=device),-1e9)
        dist=torch.distributions.Categorical(logits=logits)
        index=dist.sample()
        return int(index),dist.log_prob(index)

def generate(adapter,core,tokenizer,context,allowed_ids,limit=12):
    device=next(adapter.parameters()).device;ids=list(context[-116:]) or tokenizer.encode('Arcus:')
    output=[];allowed=torch.tensor(allowed_ids,device=device)
    with torch.no_grad():
        for _ in range(limit):
            logits=adapter(core,torch.tensor([ids[-128:]],device=device),last_only=True)[0,-1]
            probs=(logits[allowed]/.8).softmax(-1)
            token=int(allowed[torch.multinomial(probs,1)])
            if token==tokenizer.eot_token:break
            output.append(token);ids.append(token)
    raw=b''.join(tokenizer.enc.decode_single_token_bytes(i) for i in output)
    # Incomplete UTF-8 suffixes are buffered/discarded, never shown as replacement symbols.
    import codecs
    decoder=codecs.getincrementaldecoder('utf-8')('strict')
    try:text=decoder.decode(raw,final=False)
    except UnicodeDecodeError:text=''
    return ''.join(c for c in text if c.isprintable() or c in '\n\t').strip(),output
