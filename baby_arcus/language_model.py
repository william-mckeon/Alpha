"""Factorized tiktoken embeddings on the frozen motor transformer; independent text weights."""
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

    def forward(self,core,tokens):
        hidden=core.trunk_embedded(self.input(self.embedding(tokens)))
        return F.linear(self.output(hidden),self.embedding.weight)

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
            logits=adapter(core,torch.tensor([ids[-128:]],device=device))[0,-1]
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
