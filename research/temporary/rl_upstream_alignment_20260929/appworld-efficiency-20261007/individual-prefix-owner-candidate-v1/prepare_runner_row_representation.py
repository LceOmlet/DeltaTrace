"""Prepare an isolated row-cache representation extension of the active DT owner.

This changes coordinates and owner arguments only. The existing HF/FA/FLA,
finite decoder, target seed and Q/V/A rules retain all numerical computation.
It is default-inert and must not be deployed on source inspection alone.
"""
import ast
import difflib
import hashlib
import json
from pathlib import Path


HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]
OWNER=REPO/'deltatrace/clean/qwen35/qwen35_dense_finite_runner.py'
EXPECTED='e9c7576486f742c26f895cd4078891a98d94c189e84a6e565fc8042a74aabcab'


def main():
    raw=OWNER.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==EXPECTED
    old=raw.decode('utf8'); text=old.replace('\r\n','\n')
    edits=[]
    def replace(before,after):
        nonlocal text
        assert text.count(before)==1, before[:90]
        text=text.replace(before,after)
        edits.append(dict(original=before,candidate=after))

    helpers='''
def _pack_native_cached_suffix_rows(paired_ids,selection,provider):
    """Transport original valid IDs/positions to HF's existing row mask API."""
    starts=tuple(provider.prefix_lengths);lengths=tuple(provider.context_lengths)
    if (len(starts)!=selection.batch or len(lengths)!=selection.batch
            or max(starts)!=provider.prefix_length):
        raise ValueError('Prepared row prefix metadata does not match the original targets.')
    selected=selection.suffix_rows(starts,lengths)
    width=selected.length;prefix_width=provider.prefix_length
    ids=paired_ids.new_zeros((2*selection.batch,width))
    suffix_mask=paired_ids.new_zeros(ids.shape)
    full_mask=paired_ids.new_zeros((2*selection.batch,prefix_width+width))
    for row,(start,length) in enumerate(zip(starts,lengths)):
        count=length-start
        ids[2*row:2*row+2,:count]=paired_ids[2*row:2*row+2,start:length]
        suffix_mask[2*row:2*row+2,:count]=1
        full_mask[2*row:2*row+2,:start]=1
        full_mask[2*row:2*row+2,prefix_width:prefix_width+count]=1
    positions=(torch.tensor(starts,device=paired_ids.device,dtype=torch.long)
               .repeat_interleave(2)[:,None]+torch.arange(width,device=paired_ids.device)[None,:])
    positions=positions.masked_fill(suffix_mask==0,0)
    return ids,{'full_attention':full_mask,'linear_attention':suffix_mask},positions,selected


def _compact_native_cached_kv(value,starts,lengths,prefix_width):
    """Remove only masked carrier columns, retaining original K/V values."""
    # Native attention captures [2B,H,Pmax+Smax,D]; finite FA consumes
    # logically contiguous [2B,H,max(Li),D]. Torch owns the exact copies.
    result=value.new_zeros((value.shape[0],value.shape[1],max(lengths),value.shape[3]))
    for row,(start,length) in enumerate(zip(starts,lengths)):
        result[2*row:2*row+2,:,:start]=value[2*row:2*row+2,:,:start]
        result[2*row:2*row+2,:,start:length]=value[2*row:2*row+2,:,prefix_width:prefix_width+length-start]
    return result


def _public_varlen_attention_lse(captures,mask,args):
    """Compose the installed HF unpad and public FA return, without new FA."""
    from flash_attn.bert_padding import unpad_input,pad_input
    from transformers.modeling_flash_attention_utils import _upad_input
    q,k,v=[captures[name].transpose(1,2) for name in ('query','key','value')]
    batch,width=q.shape[:2]
    q,k,v,indices,(cuq,cuk),(maxq,maxk)=_upad_input(q,k,v,mask,width,unpad_input)
    aux,lse,unused=flash_attn_varlen_func(q,k,v,cuq,cuk,maxq,maxk,
        return_attn_probs=True,**args)
    # The installed MetaX owner actually returns [H,total_q] FP32,
    # measured on the saved actual operands; do not use its stale 3-D doc.
    if lse.ndim!=2 or lse.shape!=(q.shape[1],q.shape[0]):
        raise ValueError('Installed public varlen FA LSE representation changed.')
    return (pad_input(aux,indices,batch,width),
            pad_input(lse.transpose(0,1),indices,batch,width).transpose(1,2),unused)


'''
    replace('class Qwen35DenseFiniteRunner:',helpers+'class Qwen35DenseFiniteRunner:')
    replace('        prefix_start=0;replay_cache=None;native_cache=None\n',
            '        prefix_start=0;replay_cache=None;native_cache=None\n'
            '        row_starts=None;row_lengths=None;root_positions={}\n')
    replace('        if self.reuse_native_prefix and observer is None:\n',
        '''        if (self.reuse_native_prefix and observer is None
                and getattr(prefix_cache_provider,'prefix_lengths',None) is not None):
            row_starts=tuple(prefix_cache_provider.prefix_lengths)
            row_lengths=tuple(prefix_cache_provider.context_lengths)
            # The original first-change sentinel is the common padded extent.
            # Identical rows have no source change; retain that meaning at
            # their real length rather than an invalid physical-pad index.
            coefficient_starts=[length if start==original_length else start
                                for start,length in zip(coefficient_starts,row_lengths)]
            if any(start>coefficient_starts[row] for row,start in enumerate(row_starts)):
                raise ValueError('A row cache boundary crosses a changed source token.')
            prefix_start=prefix_cache_provider.prefix_length
            with torch.no_grad():
                replay_cache=timed('shared_native_prefix_cache',lambda:prefix_cache_provider(
                    paired_ids[1::2,:prefix_start]))
            replay_cache.reorder_cache(torch.arange(selection.batch,device=paired_ids.device).repeat_interleave(2))
            native_cache=timed('fork_native_prefix_cache',lambda:copy.deepcopy(replay_cache))
            paired_ids,mask,position_ids,selection=_pack_native_cached_suffix_rows(
                paired_ids,selection,prefix_cache_provider)
            root_positions={'position_ids':position_ids}
        elif self.reuse_native_prefix and observer is None:
''')
    replace("                **({'logits_to_keep':selector.rows} if selector is not None else {})))",
            "                **root_positions,**({'logits_to_keep':selector.rows} if selector is not None else {})))")
    replace('''        layout=RightPaddedLengths([original_length]*selection.batch,original_length,paired_ids.device,
                                  coefficient_starts=coefficient_starts if self.fa_coefficient_suffix or prefix_start else None,
                                  query_start=prefix_start)
        local_starts=None if coefficient_starts is None else [start-prefix_start for start in coefficient_starts]
''','''        if row_starts is not None:
            layout=RightPaddedLengths(list(row_lengths),max(row_lengths),paired_ids.device,
                coefficient_starts=coefficient_starts,query_starts=row_starts,
                query_padded_length=selection.length)
            local_starts=[start-prefix for start,prefix in zip(coefficient_starts,row_starts)]
        else:
            layout=RightPaddedLengths([original_length]*selection.batch,original_length,paired_ids.device,
                                      coefficient_starts=coefficient_starts if self.fa_coefficient_suffix or prefix_start else None,
                                      query_start=prefix_start)
            local_starts=None if coefficient_starts is None else [start-prefix_start for start in coefficient_starts]
''')
    replace("            if mc.calls!=({'module':1,'interface':1,'native_varlen':0,'native_dense':1} if is_fa else {'module':1,'conv':1,'FLA':1,'stage':1}):raise ValueError(f'Missing actual native mixer captures at layer {i} ({layer.block_type}): {mc.calls!r}')",
        "            expected_fa_calls=({'module':1,'interface':1,'native_varlen':1,'native_dense':0}\n"
        "                               if row_starts is not None else {'module':1,'interface':1,'native_varlen':0,'native_dense':1})\n"
        "            if mc.calls!=(expected_fa_calls if is_fa else {'module':1,'conv':1,'FLA':1,'stage':1}):raise ValueError(f'Missing actual native mixer captures at layer {i} ({layer.block_type}): {mc.calls!r}')")
    replace("                args={k:v for k,v in mc.dense_arguments.items() if k!='return_attn_probs'}",
        "                native_args=mc.packed_arguments if row_starts is not None else mc.dense_arguments\n"
        "                args={k:v for k,v in native_args.items() if k in ('dropout_p','softmax_scale','causal')}")
    call="flash_attn_func(c['dense_q'],c['dense_k'],c['dense_v'],return_attn_probs=True,**args)"
    # The two sites already exist for borrowed-GPU versus offloaded captures.
    assert text.count(call)==2
    replacement="(_public_varlen_attention_lse(c,kw['attention_mask'],args) if row_starts is not None else "+call+")"
    text=text.replace(call,replacement)
    edits.append(dict(original=call,candidate=replacement,count=2))
    replace("                        for name in ('dense_q','dense_k','dense_v'):\n                            del c[name]",
        "                        if row_starts is None:\n"
        "                            for name in ('dense_q','dense_k','dense_v'):\n"
        "                                del c[name]")
    replace("                    cos,sin=kw['position_embeddings'];return attention_finite_pullback(",
        "                    if row_starts is not None:\n"
        "                        for name in ('key','value'):\n"
        "                            c[name]=_compact_native_cached_kv(c[name],row_starts,row_lengths,prefix_start)\n"
        "                    cos,sin=kw['position_embeddings'];return attention_finite_pullback(")
    replace('        if prefix_start:signed=torch.nn.functional.pad(signed,(prefix_start,0))',
        '''        if row_starts is not None:
            restored=signed.new_zeros((selection.batch,original_length))
            for row,(start,length) in enumerate(zip(row_starts,row_lengths)):
                restored[row,start:length]=signed[row,:length-start]
            signed=restored
        elif prefix_start:signed=torch.nn.functional.pad(signed,(prefix_start,0))''')
    replace("              'fa_coefficient_starts':coefficient_starts if self.fa_coefficient_suffix else None,",
        "              'native_row_prefix_lengths':row_starts,'native_row_context_lengths':row_lengths,\n"
        "              'fa_coefficient_starts':coefficient_starts if self.fa_coefficient_suffix else None,")
    replace("              'gdn_fla_coefficient_start':gdn_cut,'native_shared_prefix_length':prefix_start,",
        "              'gdn_fla_coefficient_start':gdn_cut,\n"
        "              'native_shared_prefix_length':prefix_start if row_starts is None else None,\n"
        "              'native_prefix_carrier_width':prefix_start if row_starts is not None else None,")
    candidate=text.replace('\n','\r\n') if '\r\n' in old else text
    ast.parse(candidate)
    out=HERE/'native-representation-candidate'
    for name in ('baseline','candidate'):(out/name).mkdir(parents=True,exist_ok=True)
    (out/'baseline'/OWNER.name).write_bytes(raw)
    new=out/'candidate'/OWNER.name;new.write_bytes(candidate.encode('utf8'))
    patch=out/(OWNER.name+'.patch')
    patch.write_bytes(''.join(difflib.unified_diff(old.splitlines(True),candidate.splitlines(True),
        fromfile='baseline/'+OWNER.name,tofile='candidate/'+OWNER.name)).encode('utf8'))
    receipt=dict(status='prepared_only_unaccepted_not_deployed',
        baseline_sha256=EXPECTED,candidate_sha256=hashlib.sha256(new.read_bytes()).hexdigest(),
        patch_sha256=hashlib.sha256(patch.read_bytes()).hexdigest(),
        scope='Row cache/ID/mask/position/target/FA operand representation only; original finite rules, head and PPO untouched. Varlen numerical and full runner validation still required.',
        source_edits=edits,production_changes=0)
    (out/'runner-row-representation-prepared.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
    print(json.dumps({key:value for key,value in receipt.items() if key!='source_edits'},indent=2))


if __name__=='__main__':
    main()
