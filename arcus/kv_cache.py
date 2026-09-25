"""Conversation-local, bounded compact KV cache. Never stored in model checkpoints."""
class KVCache:
    def __init__(self, max_tokens):
        if type(max_tokens) is not int or max_tokens < 1: raise ValueError('Invalid cache bound')
        self.max_tokens=max_tokens
        self.layers={}
        self.positions={}
        self.length=0

    def reset(self):
        self.layers.clear(); self.positions.clear(); self.length=0

    def append(self,layer,k,v):
        import torch
        if (k.ndim!=4 or k.shape!=v.shape or k.dtype!=v.dtype or k.device!=v.device
                or k.shape[-2]+self.length>self.max_tokens):
            raise ValueError('KV cache shape or capacity exceeded')
        end=self.length+k.shape[-2]
        if layer in self.layers:
            old_k,old_v=self.layers[layer]
            if self.positions[layer]!=self.length: raise ValueError('Cache layer position mismatch')
            if (old_k.shape[:2]!=k.shape[:2] or old_k.shape[-1]!=k.shape[-1]
                    or old_k.device!=k.device or old_k.dtype!=k.dtype):
                raise ValueError('Cache tensor contract changed')
        elif self.length:
            raise ValueError('Missing cache layer')
        else:
            shape=(*k.shape[:2],self.max_tokens,k.shape[-1])
            old_k=torch.empty(shape,device=k.device,dtype=k.dtype)
            old_v=torch.empty(shape,device=v.device,dtype=v.dtype)
            self.layers[layer]=(old_k,old_v)
        old_k[:,:,self.length:end].copy_(k)
        old_v[:,:,self.length:end].copy_(v)
        self.positions[layer]=end
        return old_k[:,:,:end],old_v[:,:,:end]
