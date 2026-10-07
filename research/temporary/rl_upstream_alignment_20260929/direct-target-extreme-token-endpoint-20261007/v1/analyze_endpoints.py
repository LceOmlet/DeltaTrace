"""Describe actual endpoint scores; no credit mutation or acceptance tolerance."""
import hashlib
import json
import math
from pathlib import Path

import torch

HERE=Path(__file__).resolve().parent


def main():
    summaries=[]
    for name,row,slot,decoded in [('textcraft',3,666,' Format'),('appworld',0,7260,'\n')]:
        record=json.loads((HERE/name/'rank0.json').read_bytes())
        other=json.loads((HERE/name/'rank1.json').read_bytes())
        targets=torch.load(HERE/name/'rank0-target-logp.pt',map_location='cpu',weights_only=False)
        pairs=targets['samples'].eq(row)
        future=pairs & targets['predictor_positions'].ge(slot)
        factual=targets['factual_target_logp'].double()
        deleted=targets['reference_target_logp'].double()
        d=float((factual[future]-deleted[future]).sum())
        old=record['causal_description']['saved_DT_signed']
        top=[]
        for index in pairs.nonzero().flatten():
            top.append(dict(predictor=int(targets['predictor_positions'][index]),
                target_id=int(targets['labels'][index]),factual_lp=float(factual[index]),
                single_eos_lp=float(deleted[index]),d=float(factual[index]-deleted[index])))
        top=sorted(top,key=lambda item:abs(item['d']),reverse=True)[:15]
        controls=[value for index,value in enumerate(record['native_single_delete_effect']) if index!=row]
        assert controls==[0.0]*3
        assert record['native_single_delete_effect']==other['native_single_delete_effect']
        assert record['causal_description']['earlier_delta_maxabs'] in (None,0.0)
        summaries.append(dict(task=name,source_token=decoded,
            candidate=record['prepared']['candidate'],native_d=d,saved_DT_d=old,
            native_deleted_to_factual_ratio=math.exp(-d),DT_deleted_to_factual_ratio=math.exp(-old),
            same_reward1_native_coefficient=-math.expm1(-d),saved_DT_reward1_coefficient=-math.expm1(-old),
            factual_joint_lp=record['factual_joint_logp'][row],
            single_eos_joint_lp=record['reference_joint_logp'][row],
            factual_drift_from_original_saved=record['factual_minus_saved_original'][row],
            identity_controls=controls,earlier_target_maxabs=record['causal_description']['earlier_delta_maxabs'],
            top_target_effects=top,
            owners=record['owners'],transport=dict(path=str(HERE/name/'transport.json'),
                sha256=hashlib.sha256((HERE/name/'transport.json').read_bytes()).hexdigest())))
    result=dict(cases=summaries,
        scope='Two actual candidates and actual original native HF/PEFT forward; complete other-row and earlier-target identity controls. No environment reward or QVA changed.',
        interpretation='TextCraft has actual negative target-probability potential, with DT overstating magnitude in this one-source native endpoint. AppWorld native one-source deletion has positive effect, so its large negative DT estimate is not supported by this diagnostic.',
        numerical_scope='Full native endpoint versus saved cached DT root; factual drifts are recorded. This is a counterfactual credibility diagnostic, not whole-DT FA/FLA acceptance or cumulative-deletion/RISE evaluation.',
        retained_state='TextCraft first update remains held. AppWorld formal job terminal; diagnostics perform zero optimizer steps.',
        tokenizer_owner_decode={'198':'\n','52451':' ```','12305':'python','14606':' Format','3443':' format'})
    (HERE/'endpoint-analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    assert not torch.cuda.is_initialized()
    print(json.dumps([dict(task=row['task'],token=row['source_token'],native_d=row['native_d'],DT_d=row['saved_DT_d'],native_A=row['same_reward1_native_coefficient'],DT_A=row['saved_DT_reward1_coefficient'],top=row['top_target_effects'][0]) for row in summaries]))


if __name__=='__main__':main()
