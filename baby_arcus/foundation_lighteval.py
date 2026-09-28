"""LightEval 0.6.2 request adapter, greedy generation and exact token likelihoods."""
import torch
from lighteval.models.abstract_model import LightevalModel, ModelInfo
from lighteval.models.model_output import GenerativeResponse, LoglikelihoodResponse, LoglikelihoodSingleTokenResponse
from baby_arcus.foundation_evaluation import completion, read_only, score_ids


class ArcusLightEval(LightevalModel):
    def __init__(self, model, tokenizer, checkpoint_sha):
        self.model, self._tokenizer = model, tokenizer
        self.model_info=ModelInfo(model_name='Arcus-foundation',model_sha=checkpoint_sha,model_dtype='float32/bfloat16',model_size=str(sum(p.numel() for p in model.parameters())))

    @property
    def tokenizer(self):return self._tokenizer
    @property
    def add_special_tokens(self):return False
    @property
    def max_length(self):return self.model.body.cfg.max_seq_len
    @property
    def disable_tqdm(self):return True
    def tok_encode(self,text,add_special_tokens=None):return self.tokenizer.encode(text)
    def tok_decode(self,tokens):return [self.tokenizer.decode(list(row)) for row in tokens]

    def pair(self,context,continuation):
        # Joined tokenization, longest stable prefix handles BPE boundary merges.
        whole=self.tokenizer.encode(context+continuation);prefix=self.tokenizer.encode(context)
        start=0
        while start<min(len(prefix),len(whole)) and prefix[start]==whole[start]:start+=1
        if start==0:whole=[self.tokenizer.eot_token]+whole;start=1
        if start==len(whole):raise ValueError('Empty continuation')
        if len(whole)-1>self.max_length:raise ValueError('Request exceeds context; no silent truncation')
        return whole,start

    def likelihood(self,ids,start):
        x=torch.tensor([ids[:-1]],device=next(self.model.parameters()).device)
        with read_only(self.model),torch.autocast('cuda',dtype=torch.bfloat16):
            hidden=self.model.core.trunk_embedded(self.model.language.input(self.model.language.embedding(x)))
            projected=self.model.language.output(hidden)
            total=0.;greedy=True
            # Match the native loss's chunk boundaries, including masked prefix.
            # Changing GEMM shapes can change BF16 rounding and apparent parity.
            for at in range(0,len(ids)-1,32):
                logits=torch.nn.functional.linear(projected[:,at:at+32],self.model.language.embedding.weight).float()
                labels=torch.tensor(ids[at+1:at+1+logits.shape[1]],device=x.device)[None]
                skip=max(0,start-1-at)
                logits,labels=logits[:,skip:],labels[:,skip:]
                if not labels.numel():continue
                total+=float(logits.log_softmax(-1).gather(-1,labels[:,:,None]).sum())
                greedy=greedy and bool((logits.argmax(-1)==labels).all())
        # Use exactly the native chunked cross-entropy implementation for the
        # probability score; the extra logits above establish greedy correctness.
        with read_only(self.model):
            scored=score_ids(self.model,ids,start)
        return -scored['nll']*scored['target_tokens'],greedy

    def loglikelihood(self,requests,override_bs=None):
        results=[]
        for request in requests:
            ids,start=self.pair(request.context,request.choice)
            results.append(LoglikelihoodResponse(result=self.likelihood(ids,start),input_tokens=ids[:start],generated_tokens=ids[start:]))
        return results

    def loglikelihood_single_token(self,requests,override_bs=None):
        results=[]
        for request in requests:
            values=[]
            for choice in request.choices:
                ids,start=self.pair(request.context,choice)
                if len(ids)-start!=1:raise ValueError('Single-token request tokenized into multiple tokens')
                values.append(self.likelihood(ids,start)[0])
            results.append(LoglikelihoodSingleTokenResponse(result=values))
        return results

    def loglikelihood_rolling(self,requests,override_bs=None):
        result=[]
        for request in requests:
            tokens=[self.tokenizer.eot_token]+self.tokenizer.encode(request.context);total=0.
            for at in range(0,len(tokens)-1,self.max_length):
                total+=self.likelihood(tokens[at:at+self.max_length+1],1)[0]
            result.append(LoglikelihoodResponse(result=(total,False),input_tokens=tokens))
        return result

    def greedy_until(self,requests,override_bs=None):
        results=[]
        with read_only(self.model):
            for request in requests:
                if request.generation_size and request.generation_size>128:raise ValueError('Generation bound is 128')
                row=completion(self.model,self.tokenizer,request.context,request.generation_size or 128)
                stops=request.stop_sequence or []
                if isinstance(stops,str):stops=[stops]
                text=row['response']
                end=min([text.find(s) for s in stops if s and s in text]+[len(text)])
                results.append(GenerativeResponse(result=[text[:end]],input_tokens=self.tokenizer.encode(request.context),generated_tokens=row['token_ids']))
        return results
