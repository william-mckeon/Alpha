"""One trainable MoDE core; shared sensory context and specialized outputs."""
from io import BytesIO
import torch
from torch import nn
from baby_arcus.body_vocabulary import encode,ACTIONS
from baby_arcus.shared_experience import validate
from baby_arcus.shared_pooling import AdaptivePool,adaptive_pool

class SharedModel(nn.Module):
    def __init__(self,body,language,version=2):
        super().__init__();self.body=body;self.language=language;dim=body.cfg.dim;self.version=version
        self.rgb=nn.Sequential(nn.Conv2d(3,16,3,padding=1),nn.GELU(),AdaptivePool((4,4)))
        self.visual_input=nn.Linear(16,dim);self.internal_input=nn.Linear(6,dim)
        self.query=nn.Parameter(torch.zeros(1,1,dim))
        self.rest=nn.Linear(dim,5);self.curiosity=nn.Linear(dim,1)
        self.activity=nn.Linear(dim,6);self.language_choice=nn.Linear(dim,6)
        self.body_context=nn.Linear(dim,len(ACTIONS));self.prediction=nn.Linear(dim,5)
        self.object_input=nn.Linear(4,dim)
        self.body_sensation_input=nn.Linear(23,dim)
        self.posture_choice=nn.Linear(dim,3)
        if version>=2:
            self.gaze_input=nn.Linear(4,dim);self.gaze_choice=nn.Linear(dim,7)
            self.activity=nn.Linear(dim,7)
        if version>=3:
            self.activity=nn.Linear(dim,8)
            from baby_arcus.visual_model import MultiscaleObjectPerceptionAdapter
            self.perception=MultiscaleObjectPerceptionAdapter(dim)
            self.perceptual_input=nn.Linear(7,dim)
            self.text_context=nn.Linear(dim,language.embedding.embedding_dim)
            nn.init.zeros_(self.text_context.weight);nn.init.zeros_(self.text_context.bias)
        if version>=4:
            self.hearing_adapter=nn.Linear(language.embedding.embedding_dim,dim,bias=False)
            nn.init.zeros_(self.hearing_adapter.weight)
        if version>=6:
            # Preserve spatial order when combining pixel patches with a heard word.
            self.sensory_fusion=nn.Sequential(nn.Linear(112+language.embedding.embedding_dim+33,256),nn.GELU(),nn.Linear(256,dim))
            nn.init.zeros_(self.sensory_fusion[-1].weight);nn.init.zeros_(self.sensory_fusion[-1].bias)
        if version>=7:
            self.hearing_pool=nn.Linear(language.embedding.embedding_dim,1,bias=False)
            nn.init.zeros_(self.hearing_pool.weight)
        if version>=8:
            self.future_body=nn.Linear(dim,20);self.future_rgb=nn.Linear(dim,48)
            self.action_quality=nn.Linear(dim,2)
        if version>=9:
            self.memory_input=nn.Linear(72,dim,bias=False)
            nn.init.zeros_(self.memory_input.weight)
            self.causal_predictors=nn.ModuleList([nn.Sequential(nn.Linear(dim+164,128),nn.GELU(),nn.Linear(128,68)) for _ in range(3)])
            for head in self.causal_predictors:
                nn.init.zeros_(head[-1].weight);nn.init.zeros_(head[-1].bias)
        nn.init.zeros_(self.body_context.weight);nn.init.zeros_(self.body_context.bias)
        if body.cfg.capacity==.25:
            from baby_arcus.shared_depth import verify_depth
            verify_depth(self)

    @property
    def core(self):return self.body.core

    def forward(self,rows,tokenizer,requested=None):
        if self.version>=5 and requested and set(requested)<= {'body','lying','sitting'}:
            for row in rows:validate(row)
            goals={'body':'standing','lying':'lying','sitting':'sitting'}
            return {key:self.body([row['senses'] for row in rows],goals[key]) for key in requested}
        # Independent sequence lengths avoid padding-dependent expert routing.
        outputs=[];device=next(self.parameters()).device
        for row in rows:
            raw=validate(row)
            spatial_features=torch.zeros((1,112),device=device)
            hearing_features=torch.zeros((1,self.language.embedding.embedding_dim),device=device)
            tokens=torch.tensor([encode(row['senses'])],device=device)
            parts=[self.core.token_embed(tokens)]
            if self.version>=2:
                for prior in row.get('history',[]):
                    parts.insert(-1,self.core.token_embed(torch.tensor([encode(prior['senses'])],device=device)))
                parts.append(self.gaze_input(torch.tensor([row.get('gaze',[0,0,0,0])],device=device,dtype=torch.float32))[:,None])
            from baby_arcus.body_dynamics import JOINTS
            s=row['senses'];tilt=s['tilt']
            sensation=[s['height'],float(s['fallen']),float(s['stable']),float(s['supported']),tilt['pitch'],tilt['roll']]
            sensation += [s['joint_velocities'][key] for key in JOINTS]
            sensation += [float(s['contacts'][key]) for key in ('front_left','front_right','rear_left','rear_right')]
            sensation += [s['eyelid_openness']]
            parts.append(self.body_sensation_input(torch.tensor([sensation],device=device,dtype=torch.float32))[:,None])
            if raw:
                import numpy as np
                from PIL import Image
                with Image.open(BytesIO(raw)) as image:
                    if image.width>800 or image.height>560:raise ValueError('Oversized sensory frame')
                    pixels=torch.tensor(np.asarray(image.convert('RGB').resize((96,96))).copy(),device=device,dtype=torch.float32).permute(2,0,1)[None]/255
                parts.append(self.visual_input(self.rgb(pixels).flatten(2).transpose(1,2)))
                if self.version>=3:
                    perception=self.perception(self.core,pixels)
                    probabilities=perception.softmax(1)
                    mass=adaptive_pool(probabilities[:,3:4],(4,4))
                    colors=adaptive_pool(pixels*probabilities[:,3:4],(4,4))/mass.clamp_min(.001)
                    features=torch.cat((colors,adaptive_pool(probabilities,(4,4))),1)
                    spatial_features=features.flatten(1)
                    parts.append(self.perceptual_input(features.flatten(2).transpose(1,2)))
            messages=row['hearing'][-1:] if self.version>=4 else row['hearing']
            text='\n'.join(str(m.get('text','')) for m in messages)[-2048:]
            ids=row.get('language_prefix_ids',tokenizer.encode(text)[-64:])
            if len(ids)>64 or any(type(i) is not int or not 0<=i<self.language.embedding.num_embeddings for i in ids):raise ValueError('Invalid language prefix')
            if ids:
                embedding=self.language.embedding(torch.tensor([ids],device=device))
                hearing_features=embedding.mean(1)
                if self.version>=7:hearing_features=(embedding*self.hearing_pool(embedding).softmax(1)).sum(1)
                hearing=self.language.input(embedding)
                if self.version>=4:hearing=hearing+self.hearing_adapter(embedding)
                parts.append(hearing)
            if row.get('executed_action') is not None:
                # Counterfactual/outcome training conditions on the verified action,
                # without pretending that an action record was heard speech.
                import json
                action_ids=tokenizer.encode('action:'+json.dumps(row['executed_action'],sort_keys=True))[-64:]
                parts.append(self.language.input(self.language.embedding(torch.tensor([action_ids],device=device))))
            internal=torch.tensor([row['internal']+[float(bool(raw))]],device=device,dtype=torch.float32)
            parts.extend([self.internal_input(internal)[:,None],self.query])
            if self.version>=6:
                physical=torch.tensor([sensation+row.get('gaze',[0,0,0,0])],device=device,dtype=torch.float32)
                fused=torch.cat((spatial_features,hearing_features,physical,internal),dim=-1)
                parts[-1]=self.query+self.sensory_fusion(fused)[:,None]
            object_rows=row.get('objects',[])[:32]
            if object_rows:
                object_tokens=self.object_input(torch.tensor([[o['features'] for o in object_rows]],device=device,dtype=torch.float32))
                parts.insert(-1,object_tokens)
            sequence=torch.cat(parts,dim=1)
            if self.version>=9 and row.get('executed_action') and row.get('memory'):
                memories=row['memory'][-8:]
                memory=torch.tensor([m['features'] for m in memories],device=device,dtype=torch.float32).mean(0,keepdim=True)
                sequence=sequence.clone();sequence[:,-1]+=self.memory_input(memory)
            if sequence.shape[1]>self.body.cfg.max_seq_len:raise ValueError('Shared context exceeds core capacity')
            hidden=self.core.trunk_embedded(sequence)[:,-1]
            text_features=self.language.output(hidden)
            if self.version>=3 and (requested is None or 'text' in requested):
                prefix=ids or tokenizer.encode('Arcus:')
                language_hidden=self.core.trunk_embedded(self.language.input(self.language.embedding(torch.tensor([prefix],device=device))))[:,-1]
                text_features=self.language.output(language_hidden)+self.text_context(hidden)
            if self.version>=3:
                if not raw:perception=torch.zeros((1,4,96,96),device=device)
            # Existing motor pathway uses the very same core; zero residual starts with retained standing logits.
            from baby_arcus.body_vocabulary import mask
            body_hidden=self.core.trunk(tokens)[:,-1]
            allowed=torch.tensor([mask(row['senses'])],device=device)
            # Sensory context selects the intention; retained motor heads execute it.
            # An unconstrained context residual overturned working joint sequences.
            def motor(head):
                logits=head(body_hidden)
                if self.version<5:logits=logits+self.body_context(hidden)
                return logits.masked_fill(~allowed,-torch.inf)
            object_values=torch.full((1,32),-1e9,device=device)
            if object_rows:object_values[:,:len(object_rows)]=self.curiosity(object_tokens+hidden[:,None]).squeeze(-1)
            result={'body':motor(self.body.actor),
                'lying':motor(self.body.lying_actor),
                'sitting':motor(self.body.sitting_actor),
                'posture_choice':self.posture_choice(hidden),'objects':object_values,
                'rest':self.rest(hidden),'curiosity':self.curiosity(hidden),'activity':self.activity(hidden),
                'language_choice':self.language_choice(hidden),'prediction':self.prediction(hidden),
                'text':nn.functional.linear(text_features,self.language.embedding.weight) if requested is None or 'text' in requested else torch.empty((1,0),device=device),
                'hidden':hidden,'aux':self.core.last_aux_loss}
            if self.version>=2:result['gaze_choice']=self.gaze_choice(hidden)
            if self.version>=3:result['perception']=perception
            if self.version>=8:
                result.update(future_body=self.future_body(hidden),future_rgb=self.future_rgb(hidden),action_quality=self.action_quality(hidden))
            if self.version>=9:
                from baby_arcus.shared_causal import causal_features
                features=torch.tensor([causal_features(row)],device=device,dtype=torch.float32)
                context=torch.cat((nn.functional.normalize(hidden,dim=-1),features),-1)
                predictions=torch.stack([head(context) for head in self.causal_predictors])
                baseline=torch.cat((features[:,:20],features[:,24:72]),-1)
                predicted=baseline+predictions.mean(0)
                result.update(future_body=predicted[:,:20],future_rgb=predicted[:,20:],
                    future_ensemble=baseline[None]+predictions,uncertainty=predictions.var(0,unbiased=False).mean(-1,keepdim=True))
                # Batch first, then ensemble index.
                result['future_ensemble']=result['future_ensemble'].transpose(0,1)
                result['curiosity']=self.curiosity(nn.functional.normalize(hidden,dim=-1)).sigmoid()
            outputs.append(result)
        return {key:torch.cat([r[key] for r in outputs],dim=0) if key!='aux' else torch.stack([r[key] for r in outputs]).mean() for key in outputs[0] if requested is None or key in requested}
