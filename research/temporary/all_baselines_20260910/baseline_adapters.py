"""Explicit output-scope adapters around unchanged author attribution primitives."""
import numpy as np

def aggregate_matrix(matrix,weights,prompt_tokens):
    raw=np.asarray(matrix,dtype=np.float32);w=np.asarray(weights,dtype=np.float64)
    if raw.ndim!=2 or w.shape!=(raw.shape[0],) or not 0<prompt_tokens<=raw.shape[1]:
        raise ValueError('Invalid matrix/target shape')
    if not set(w)<=set((0.,1.)) or not w.any():raise ValueError('Invalid target weights')
    if not np.isfinite(raw[w>0,:prompt_tokens]).all():raise ValueError('A selected output is missing prompt attribution')
    if np.isinf(raw).any():raise ValueError('Infinite raw attribution')
    finite=np.nan_to_num(raw,nan=0.)
    signed=np.sum(finite*w[:,None],axis=0,dtype=np.float64).astype(np.float32)
    positive=np.maximum(finite,0)
    normalized=positive/(positive.sum(axis=1,keepdims=True,dtype=np.float32)+np.float32(1e-8))
    native=np.sum(normalized*w[:,None],axis=0,dtype=np.float64).astype(np.float32)
    return signed,native

def validate_target(weights,generation_tokens):
    w=list(map(float,weights))
    expected=[float(i<len(generation_tokens)-1 and str(token).strip() not in ('',',','.')) for i,token in enumerate(generation_tokens)]
    if w!=expected or not any(w):raise ValueError('Target scope differs from the frozen DT/FT case')
    return w

def native_attributor(method,model,tokenizer,*,mlm_path=None):
    import llm_attr
    if method in ('Perturbation','REAGENT','CLP'):
        import perturbation_fast
        class FixedMLMAttribution(perturbation_fast.LLMPerturbationFastAttribution):
            def _ensure_mlm(self):
                if self._mlm_tokenizer is not None and self._mlm_model is not None:return
                if mlm_path is None:raise ValueError('A verified local MLM checkpoint is required')
                from transformers import LongformerForMaskedLM,LongformerTokenizer
                self._mlm_tokenizer=LongformerTokenizer.from_pretrained(str(mlm_path),local_files_only=True)
                self._mlm_model=LongformerForMaskedLM.from_pretrained(str(mlm_path),local_files_only=True).to(self.device)
                self._mlm_model.eval().requires_grad_(False)
        return FixedMLMAttribution(model,tokenizer)
    if method=='IFR':return llm_attr.LLMIFRAttribution(model,tokenizer,chunk_tokens=128,sink_chunk_tokens=32)
    if method=='AttnLRP':return llm_attr.LLMLRPAttribution(model,tokenizer)
    raise ValueError(method)

def calculate(method,tracer,item):
    """No gold or implicit `ex`/`example` answer-span stack variable is supplied."""
    import torch
    prompt,target=item['prompt'],item['target'];weights=item['target_weights']
    if method=='AttnLRP':
        result=tracer.calculate_attnlrp_span_aggregate(prompt,target=target,sink_start=0,
            sink_end=item['target_length']-1,sink_weights=torch.tensor(weights),normalize_weights=False,score_mode='generated')
        raw=result.token_importance_total.detach().float().cpu().numpy()
        if raw.ndim!=1 or not np.isfinite(raw).all():raise ValueError('Invalid LRP aggregate')
        native=np.maximum(raw,0);native=native/(native.sum(dtype=np.float32)+np.float32(1e-12))
        return raw,native,dict(raw_aggregate=raw,applied_target_weights=np.asarray(weights,dtype=np.float32))
    if method in ('Perturbation','CLP'):
        result=tracer.calculate_feature_ablation_segments(prompt,baseline=tracer.tokenizer.eos_token_id,
            measure='log_loss' if method=='Perturbation' else 'KL',target=target,source_k=20)
    elif method=='REAGENT':result=tracer.calculate_feature_ablation_segments_mlm(prompt,target=target,source_k=20)
    elif method=='IFR':result=tracer.calculate_ifr_for_all_positions(prompt,target=target)
    else:raise ValueError(method)
    raw=result.attribution_matrix.detach().float().cpu().numpy()
    signed,native=aggregate_matrix(raw,weights,len(item['user_positions']))
    return signed,native,dict(raw_matrix=raw,applied_target_weights=np.asarray(weights,dtype=np.float32))
