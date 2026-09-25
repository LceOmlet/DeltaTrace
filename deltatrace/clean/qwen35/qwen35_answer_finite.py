"""Explicit answer-target finite seed for the original Qwen3.5 output head.

Consumes actual original head logits for text targets. Categorical events use
the original head's selected weight rows and captured input at FP32 precision.
This never reconstructs a model forward. The
caller supplies target offsets explicitly: no default whole-response objective,
padding labels, tokenizer-size vocabulary crop or implicit sink convention.
"""
import copy
import torch
from compiled_logprob_seed import seed_with_checks
from qwen35_decoder_finite import _linear_transpose


class PackedAnswerTargets:
    """Pack genuine target rows across examples, endpoints interleaved per token.

    Offsets are zero-based indices in the author's fixed generation. Closed
    sink intervals must be expanded by the caller. The causal predictor is at
    prompt_length + target_offset - 1. CPU validation happens once.
    """
    def __init__(self,cases,target_offsets,padded_length,device,*,outcome_token_ids=None):
        if not cases or len(cases)!=len(target_offsets):raise ValueError('One explicit target selection per case is required.')
        samples=[];positions=[];labels=[];self.counts=[];self.offsets=[]
        for b,(case,offsets) in enumerate(zip(cases,target_offsets)):
            offsets=list(offsets);target=case['target_ids'].cpu();prompt=case['prompt_length']
            if not offsets or offsets!=sorted(set(offsets)):raise ValueError('Targets must be nonempty, unique and ordered.')
            if not all(isinstance(j,int) and 0<=j<len(target) for j in offsets):raise ValueError('Target index outside fixed generation.')
            if prompt<1 or prompt+len(target)>padded_length:raise ValueError('Invalid padded input length.')
            self.counts.append(len(offsets));self.offsets.append(offsets)
            samples.extend([b]*len(offsets));positions.extend(prompt+j-1 for j in offsets);labels.extend(int(target[j]) for j in offsets)
        self.batch=len(cases);self.length=padded_length;self.samples=torch.tensor(samples,device=device,dtype=torch.long)
        self.positions=torch.tensor(positions,device=device,dtype=torch.long);self.labels=torch.tensor(labels,device=device,dtype=torch.long)
        self.paired_samples=(2*self.samples[:,None]+torch.arange(2,device=device)[None,:]).flatten()
        self.paired_positions=self.positions.repeat_interleave(2)
        self.outcome_token_ids=None
        if outcome_token_ids is not None:
            ids=torch.as_tensor(outcome_token_ids,device=device,dtype=torch.long)
            if ids.ndim!=1 or len(ids)<2 or len(ids.unique())!=len(ids):
                raise ValueError('Outcome labels must be distinct vocabulary tokens.')
            if not bool(torch.isin(self.labels,ids).all()):
                raise ValueError('Each target must belong to the declared outcome alphabet.')
            self.outcome_token_ids=ids

    def pack_hidden(self,paired_hidden):
        assert paired_hidden.ndim==3 and paired_hidden.shape[:2]==(2*self.batch,self.length)
        return paired_hidden[self.paired_samples,self.paired_positions].contiguous()

    def scatter_hidden(self,packed):
        assert packed.ndim==2 and len(packed)==len(self.labels)
        dense=packed.new_zeros((self.batch,self.length,packed.shape[-1]))
        dense[self.samples,self.positions]=packed
        return dense

    def sample_sums(self,values):
        assert values.shape==(len(self.labels),)
        return values.new_zeros(self.batch).index_add_(0,self.samples,values)

    def suffix(self,start):
        """Rebase the same target identities onto a native cached suffix.

        Only positions and the activation extent change; samples, labels and
        endpoint ordering remain the exact artifacts selected by the caller.
        """
        if type(start) is not int or not 0<=start<self.length or not bool((self.positions>=start).all()):
            raise ValueError('Every target predictor must lie in the cached suffix.')
        selected=copy.copy(self)
        selected.length=self.length-start
        selected.positions=self.positions-start
        selected.paired_positions=self.paired_positions-start
        return selected


def _answer_seed_rule(z0,z1,target,weight):
    # Reuse the established logarithmic-mean logsoftmax finite seed. Both
    # nonlinear evaluation and reduction are FP32; vendor GEMM uses BF16.
    z0=z0.float();z1=z1.float()
    seed,checks=seed_with_checks(z0,z1,target)
    hidden=_linear_transpose(seed,weight)
    logp0=z0.log_softmax(-1).gather(-1,target[:,None]).squeeze(-1)
    logp1=z1.log_softmax(-1).gather(-1,target[:,None]).squeeze(-1)
    allocated=(seed*(z1-z0)).sum(-1)
    return hidden,checks,logp0,logp1,allocated


def _categorical_answer_seed_rule(z0,z1,target,weight):
    """Same finite log-softmax seed and linear transpose, both at FP32.

    The event readout is projected at FP32 before this call. Do not multiply
    the seed by a secant of separately rounded BF16 outputs: that attributes
    output quantization error to the input and can amplify it arbitrarily.
    """
    z0=z0.float();z1=z1.float();weight=weight.float()
    seed,checks=seed_with_checks(z0,z1,target)
    hidden=seed@weight
    logp0=z0.log_softmax(-1).gather(-1,target[:,None]).squeeze(-1)
    logp1=z1.log_softmax(-1).gather(-1,target[:,None]).squeeze(-1)
    return hidden,checks,logp0,logp1,(seed*(z1-z0)).sum(-1)


def categorical_head_logits(head,original_hidden,outcome_token_ids):
    """Project only declared event rows of the original head, without BF16 storage.

    No parameters or model activations are changed. Disabling ambient autocast
    here is necessary: a float() input alone does not keep Linear in FP32.
    """
    assert isinstance(head,torch.nn.Linear) and head.bias is None
    with torch.autocast(device_type=original_hidden.device.type,enabled=False):
        return torch.nn.functional.linear(original_hidden.float(),
            head.weight.index_select(0,outcome_token_ids).float())


class FiniteAnswerOps:
    def __init__(self,compiled=True,*,dynamic_shapes=False,compiler_options=None):
        self.dynamic_shapes=compiled and dynamic_shapes
        self.seed=torch.compile(_answer_seed_rule,fullgraph=True,dynamic=None if dynamic_shapes else False,
            options={'triton.cudagraphs':False,'max_autotune':False,**(compiler_options or {})}) if compiled else _answer_seed_rule
        self.categorical_seed=torch.compile(_categorical_answer_seed_rule,fullgraph=True,dynamic=None if dynamic_shapes else False,
            options={'triton.cudagraphs':False,'max_autotune':False,**(compiler_options or {})}) if compiled else _categorical_answer_seed_rule

    def __call__(self,original_packed_logits,head,selection,equal_endpoint=False,*,original_packed_hidden=None,outcome_logits=None):
        assert isinstance(head,torch.nn.Linear) and head.bias is None
        assert original_packed_logits.shape==(2*len(selection.labels),head.out_features)
        assert head.out_features==head.weight.shape[0]  # Full model vocabulary.
        z0=original_packed_logits[1::2] if equal_endpoint else original_packed_logits[0::2]
        z1=original_packed_logits[1::2]
        if self.dynamic_shapes and len(selection.labels)>1:
            for value in (z0,z1,selection.labels):
                torch._dynamo.mark_dynamic(value,0)
        target=selection.labels;weight=head.weight
        outcomes=getattr(selection,'outcome_token_ids',None)
        if outcomes is not None:
            # This explicitly requested categorical target uses the same head
            # rows and established finite logsoftmax rule. Default text targets
            # retain their full-vocabulary normalization unchanged.
            weight=weight.index_select(0,outcomes)
            target=(target[:,None]==outcomes[None,:]).long().argmax(-1)
            if original_packed_hidden is None or original_packed_hidden.shape!=(2*len(selection.labels),head.in_features):
                raise ValueError('Categorical finite head requires its captured native input rows.')
            if outcome_logits is None:
                outcome_logits=categorical_head_logits(head,original_packed_hidden,outcomes)
            assert outcome_logits.shape==(2*len(selection.labels),len(outcomes))
            z0=outcome_logits[1::2] if equal_endpoint else outcome_logits[0::2]
            z1=outcome_logits[1::2]
            if self.dynamic_shapes and len(selection.labels)>1:
                for value in (z0,z1):torch._dynamo.mark_dynamic(value,0)
            with torch.autocast(device_type=z0.device.type,enabled=False):
                result=self.categorical_seed(z0,z1,target,weight)
        else:
            result=self.seed(z0,z1,target,weight)
        hidden,valid,*diagnostics=result
        if not bool(valid):raise ValueError('Finite answer seed invalid.')
        return selection.scatter_hidden(hidden),{'packed_hidden':hidden,'logp0':diagnostics[0],
            'logp1':diagnostics[1],'allocated_logit_effect':diagnostics[2]}


def outcome_log_probs(original_logits,outcome_token_ids):
    """Declared event alphabet on original logits; no new head or forward."""
    return original_logits.index_select(-1,outcome_token_ids).float().log_softmax(-1)


def selected_target_log_probs(original_logits,selection,*,outcome_logits=None):
    outcomes=getattr(selection,'outcome_token_ids',None)
    labels=selection.labels
    if outcomes is None:
        logp=original_logits.float().log_softmax(-1)
    else:
        logp=(outcome_log_probs(original_logits,outcomes) if outcome_logits is None
              else outcome_logits.float().log_softmax(-1))
        labels=(labels[:,None]==outcomes[None,:]).long().argmax(-1)
    return logp.gather(-1,labels.repeat_interleave(2)[:,None]).squeeze(-1)
