"""Diagnose native FP16 AttnLRP on the fixed first pilot without scoring gold."""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback

HERE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--environment',type=Path,required=True);a=p.parse_args()
    env=json.loads(a.environment.read_bytes())['qwen3']
    os.environ.update(MACA_PATH='/opt/maca',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    sys.path.insert(0,env['official_root'])
    import torch
    from exp.exp2.run_exp import load_model
    import lrp_rules
    from baseline_adapters import native_attributor,calculate
    torch.set_num_threads(4)
    model,tokenizer=load_model(env['checkpoint'],'cuda:0');model.requires_grad_(False)
    item=json.loads((HERE/'inputs.json').read_bytes())['cases'][0]
    tracer=native_attributor('AttnLRP',model,tokenizer)
    with torch.autograd.detect_anomaly():
        try:calculate('AttnLRP',tracer,item)
        except Exception:traceback.print_exc()
    original=lrp_rules.IdentityRuleImplicitFn.forward;counts=dict(calls=0,nonfinite_ratio=0,zero_input=0)
    def observed(ctx,fn,input,epsilon=1e-10):
        result=original(ctx,fn,input,epsilon)
        if input.requires_grad:
            ratio=fn(input)/(input+epsilon)
            counts['calls']+=1;counts['nonfinite_ratio']+=int((~torch.isfinite(ratio)).sum())
            counts['zero_input']+=int((input==0).sum())
        return result
    lrp_rules.IdentityRuleImplicitFn.forward=staticmethod(observed)
    try:calculate('AttnLRP',tracer,item)
    except Exception:traceback.print_exc()
    finally:lrp_rules.IdentityRuleImplicitFn.forward=staticmethod(original)
    print(json.dumps(counts),flush=True)

if __name__=='__main__':main()
