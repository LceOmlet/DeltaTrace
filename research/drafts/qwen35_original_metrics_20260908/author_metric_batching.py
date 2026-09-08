"""UNEXECUTED DRAFT, deferred after the shared NI0 adaptation failure.

Intended to batch author evaluator requests without changing its metric function.

The two original metric calls run in CPU threads and request real token logprobs.
Only the main thread executes the actual model, with a right-padded B2. No proxy
model or fabricated logprob is used. Cache identity is the entire paired input.
"""
import concurrent.futures
import hashlib
import queue
import threading
import torch


class MetricBatchRequests:
    def __init__(self):
        self.queue=queue.Queue();self.pending=[];self.lock=threading.Lock();self.aborted=False

    def request(self,sample,prompt,response):
        promise=concurrent.futures.Future()
        item={'sample':sample,'prompt':prompt.detach().cpu().clone(),'response':response.detach().cpu().clone(),'promise':promise}
        with self.lock:
            if self.aborted:raise RuntimeError('Metric batching was aborted.')
            self.pending.append(promise)
        self.queue.put(item)
        return promise.result(timeout=180)

    def pair(self):
        items=[self.queue.get(timeout=30),self.queue.get(timeout=30)]
        items.sort(key=lambda x:x['sample'])
        if [x['sample'] for x in items]!=[0,1]:raise RuntimeError('Expected one real request from each sample.')
        return items

    def abort(self):
        with self.lock:
            self.aborted=True
            for future in self.pending:
                if not future.done():future.set_exception(RuntimeError('Native metric batch execution stopped.'))


class BatchedAuthorEvaluator:
    """Delegate formatting to the real author evaluator; batch only model queries."""
    def __init__(self,original,sample,requests):
        self.original=original;self.sample=sample;self.requests=requests
        self.tokenizer=original.tokenizer;self.device=original.device

    def format_prompt(self,prompt):return self.original.format_prompt(prompt)
    def _ensure_pad_token_id(self):return self.original._ensure_pad_token_id()
    def _find_subsequence_start(self,*args):return self.original._find_subsequence_start(*args)
    def compute_logprob_response_given_prompt(self,prompt_ids,response_ids):
        return self.requests.request(self.sample,prompt_ids,response_ids)


def pack_native_metric_requests(items,pad_token_id):
    sequences=[torch.cat([x['prompt'],x['response']],dim=1)[0] for x in items]
    lengths=[len(x) for x in sequences];width=max(lengths)
    ids=torch.full((len(items),width),pad_token_id,dtype=torch.long);mask=torch.zeros_like(ids)
    for b,sequence in enumerate(sequences):ids[b,:len(sequence)]=sequence;mask[b,:len(sequence)]=1
    digest=hashlib.sha256(ids.numpy().tobytes()+mask.numpy().tobytes()).hexdigest()
    return ids,mask,lengths,digest


def native_batch_response_logprobs(model,items,ids,mask):
    """Author log_softmax/gather semantics in native logits dtype, full vocabulary."""
    with torch.no_grad():
        outputs=model(input_ids=ids.to(model.device),attention_mask=mask.to(model.device),use_cache=False)
        logits=outputs.logits
        if logits.dtype!=torch.bfloat16 or logits.shape[-1]!=248320:raise ValueError('Unexpected original Qwen3.5 logits.')
        # Exactly the author's dtype: do not silently cast BF16 logits to FP32.
        log_probs=torch.nn.functional.log_softmax(logits,dim=-1)
        results=[]
        for b,item in enumerate(items):
            prompt_length=item['prompt'].shape[1];target=item['response'].to(model.device);length=target.shape[1]
            selected=log_probs[b:b+1,prompt_length-1:prompt_length+length-1]
            result=selected.gather(2,target.unsqueeze(-1)).squeeze(-1).detach().clone()
            if not torch.isfinite(result).all():raise ValueError('Nonfinite actual model target logprob.')
            results.append(result)
    return results
