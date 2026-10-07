set -eu
source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'
import json,torch,sys,hashlib,inspect,time,psutil
from pathlib import Path
root=Path("/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922")
directory=root/"receipts/direct-target-prefix-runtime-20261007-v1/textcraft-first-dt"
points=[{'task': 'textcraft', 'mode': 'most_negative', 'row': 0, 'traj_uid': '3ffd8fc6-00f0-4dd4-a5dd-99c4986e92bc', 'response_slot': 809, 'packed_slot': 1124, 'token_id': 2233, 'token': ' No', 'reward': 1.0, 'DT_d': -0.8275489862675691, 'native_single_delete_d': 0.19830815475052077, 'DT_expected_A_FP32': -1.2877047061920166, 'native_endpoint_expected_A_FP32': 0.17988291382789612, 'sign_disagreement': True}, {'task': 'textcraft', 'mode': 'most_negative', 'row': 1, 'traj_uid': '82100a14-b443-4cf8-88db-03fad6e145e5', 'response_slot': 826, 'packed_slot': 1141, 'token_id': 1, 'token': '"', 'reward': 1.0, 'DT_d': -0.8157693300687925, 'native_single_delete_d': -0.04334571356002925, 'DT_expected_A_FP32': -1.2609143257141113, 'native_endpoint_expected_A_FP32': -0.04429885745048523, 'sign_disagreement': False}, {'task': 'textcraft', 'mode': 'most_negative', 'row': 2, 'traj_uid': 'f46bd5db-f29c-4445-9ea2-34c7e1c89ab0', 'response_slot': 810, 'packed_slot': 1125, 'token_id': 29281, 'token': ' readily', 'reward': 1.0, 'DT_d': -0.9158169973117463, 'native_single_delete_d': -0.03209965685800853, 'DT_expected_A_FP32': -1.4988160133361816, 'native_endpoint_expected_A_FP32': -0.03262040764093399, 'sign_disagreement': False}, {'task': 'textcraft', 'mode': 'most_negative', 'row': 3, 'traj_uid': '6e76f70e-bdeb-4726-8b35-d9e21eff2f68', 'response_slot': 351, 'packed_slot': 666, 'token_id': 14606, 'token': ' Format', 'reward': 1.0, 'DT_d': -3.1703966315267027, 'native_single_delete_d': -2.2145728506379783, 'DT_expected_A_FP32': -22.816926956176758, 'native_endpoint_expected_A_FP32': -8.15749740600586, 'sign_disagreement': False}, {'task': 'textcraft', 'mode': 'most_positive', 'row': 0, 'traj_uid': '3ffd8fc6-00f0-4dd4-a5dd-99c4986e92bc', 'response_slot': 299, 'packed_slot': 614, 'token_id': 248068, 'token': '<think>', 'reward': 1.0, 'DT_d': 7.123233280440925, 'native_single_delete_d': -0.06756158271173263, 'DT_expected_A_FP32': 0.9991938471794128, 'native_endpoint_expected_A_FP32': -0.0698961466550827, 'sign_disagreement': True}, {'task': 'textcraft', 'mode': 'most_positive', 'row': 1, 'traj_uid': '82100a14-b443-4cf8-88db-03fad6e145e5', 'response_slot': 380, 'packed_slot': 695, 'token_id': 1639, 'token': '\\n', 'reward': 1.0, 'DT_d': 10.14586344030714, 'native_single_delete_d': 13.51900724362639, 'DT_expected_A_FP32': 0.9999607801437378, 'native_endpoint_expected_A_FP32': 0.9999986290931702, 'sign_disagreement': False}, {'task': 'textcraft', 'mode': 'most_positive', 'row': 2, 'traj_uid': 'f46bd5db-f29c-4445-9ea2-34c7e1c89ab0', 'response_slot': 390, 'packed_slot': 705, 'token_id': 328, 'token': ' "', 'reward': 1.0, 'DT_d': 9.840191456445535, 'native_single_delete_d': 14.66602984987901, 'DT_expected_A_FP32': 0.9999467134475708, 'native_endpoint_expected_A_FP32': 0.9999995827674866, 'sign_disagreement': False}, {'task': 'textcraft', 'mode': 'most_positive', 'row': 3, 'traj_uid': '6e76f70e-bdeb-4726-8b35-d9e21eff2f68', 'response_slot': 362, 'packed_slot': 677, 'token_id': 1639, 'token': '\\n', 'reward': 1.0, 'DT_d': 9.549379231403675, 'native_single_delete_d': 15.118693931070538, 'DT_expected_A_FP32': 0.9999287724494934, 'native_endpoint_expected_A_FP32': 0.9999997019767761, 'sign_disagreement': False}, {'task': 'textcraft', 'mode': 'median_negative', 'row': 0, 'traj_uid': '3ffd8fc6-00f0-4dd4-a5dd-99c4986e92bc', 'response_slot': 1851, 'packed_slot': 2166, 'token_id': 1, 'token': '"', 'reward': 1.0, 'DT_d': -0.01137253627993617, 'native_single_delete_d': 0.10733730633637606, 'DT_expected_A_FP32': -0.011437449604272842, 'native_endpoint_expected_A_FP32': 0.10177735239267349, 'sign_disagreement': True}, {'task': 'textcraft', 'mode': 'median_negative', 'row': 1, 'traj_uid': '82100a14-b443-4cf8-88db-03fad6e145e5', 'response_slot': 2705, 'packed_slot': 3020, 'token_id': 3106, 'token': ' sent', 'reward': 1.0, 'DT_d': -0.006665757385835039, 'native_single_delete_d': 0.003104389454165357, 'DT_expected_A_FP32': -0.0066880229860544205, 'native_endpoint_expected_A_FP32': 0.0030995758716017008, 'sign_disagreement': True}, {'task': 'textcraft', 'mode': 'median_negative', 'row': 2, 'traj_uid': 'f46bd5db-f29c-4445-9ea2-34c7e1c89ab0', 'response_slot': 1941, 'packed_slot': 2256, 'token_id': 45714, 'token': ' Lily', 'reward': 1.0, 'DT_d': -0.01118121608687558, 'native_single_delete_d': 0.10547941595200427, 'DT_expected_A_FP32': -0.011243958957493305, 'native_endpoint_expected_A_FP32': 0.10010700672864914, 'sign_disagreement': True}, {'task': 'textcraft', 'mode': 'median_negative', 'row': 3, 'traj_uid': '6e76f70e-bdeb-4726-8b35-d9e21eff2f68', 'response_slot': 3380, 'packed_slot': 3695, 'token_id': 198, 'token': '\n', 'reward': 1.0, 'DT_d': -0.010928304912416207, 'native_single_delete_d': 0.14578915692345618, 'DT_expected_A_FP32': -0.010988237336277962, 'native_endpoint_expected_A_FP32': 0.1356600821018219, 'sign_disagreement': True}]
saved=[torch.load(directory/f"rank{rank}-pre-update.pt",map_location="cpu",weights_only=False) for rank in (0,1)]
actor_path=Path(saved[0]["provenance"]["owner"]["actor_path"])
sys.path.insert(0,str(actor_path.parents[3]))
import verl.utils.torch_functional as F
owner=Path(inspect.getsourcefile(F.masked_whiten))
raw=torch.cat([d["tensors"]["dt_token_advantages"] for d in saved])
adv=torch.cat([d["tensors"]["advantages"] for d in saved])
mask=torch.cat([d["tensors"]["response_mask"] for d in saved])
expected=F.masked_whiten(raw,mask)*mask
r={"scope":"Read-only actual pre-update actor mapping and original whole-batch whitening; not parameter gradient measurement or a production credit change","unix":time.time(),"owner":{"path":str(owner),"sha256":hashlib.sha256(owner.read_bytes()).hexdigest(),"masked_whiten_source":inspect.getsource(F.masked_whiten)},"input_paths":[str(directory/f"rank{rank}-pre-update.pt") for rank in (0,1)],"mask_matches_original_loss_mask":[torch.equal(d["tensors"]["response_mask"],d["tensors"]["loss_mask"][:,-raw.shape[1]:]) for d in saved],"actual_advantages_match_owner_whitening":torch.equal(adv,expected),"owner_whitening_maxabs_diff":(adv-expected).abs().max().item(),"points":[]}
valid=mask.bool()
for name,t in (("raw",raw),("actor",adv)):
 v=t[valid].double();neg=v[v<0]
 r[name+"_stats"]={"valid_count":v.numel(),"min":v.min().item(),"max":v.max().item(),"mean":v.mean().item(),"sumsq":v.square().sum().item(),"negative_count":neg.numel(),"negative_sumsq":neg.square().sum().item(),"nonfinite":(~torch.isfinite(v)).sum().item()}
for p in points:
 q=dict(p);q["matches"]=[]
 for rank,d in enumerate(saved):
  for row,uid in enumerate(d["non_tensors"]["traj_uid"]):
   if str(uid)!=p["traj_uid"]:continue
   artifact=d["non_tensors"]["dt_direct_target_artifact"][row]
   pos=torch.as_tensor(artifact["retained_response_positions"])
   original=artifact["response_ids"]
   slots=(pos==p["response_slot"]).nonzero(as_tuple=False).flatten().tolist()
   record={"rank":rank,"training_row":row,"full_response_length":len(original),"retained_nonpadding_count":int((pos>=0).sum()),"original_token_id":int(original[p["response_slot"]]),"slots":[]}
   assert record["original_token_id"]==p["token_id"]
   for slot in slots:
    t=d["tensors"];assert int(t["responses"][row,slot])==p["token_id"]
    record["slots"].append({"actor_response_slot":slot,"valid_policy":bool(t["response_mask"][row,slot]),"raw_A":t["dt_token_advantages"][row,slot].item(),"Q":t["dt_q_estimates"][row,slot].item(),"V":t["dt_v_estimates"][row,slot].item(),"actor_A":t["advantages"][row,slot].item()})
   q["matches"].append(record)
 q["retained_training_slots"]=sum(len(x["slots"]) for x in q["matches"])
 q["valid_training_slots"]=sum(sum(t["valid_policy"] for t in x["slots"]) for x in q["matches"])
 r["points"].append(q)
r["cpu_cuda_initialized"]=torch.cuda.is_initialized();assert not r["cpu_cuda_initialized"]
r["peak_rss_gib"]=psutil.Process().memory_info().rss/2**30
r["text_release_present"]=[(directory/f"rank{rank}-release-update").exists() for rank in (0,1)]
r["largest_actual_actor_positions"]=[]
for flat in torch.topk(torch.where(valid,adv.abs(),torch.zeros_like(adv)).flatten(),10).indices.tolist():
 globalrow,slot=divmod(flat,raw.shape[1]);rank,row=divmod(globalrow,128);d=saved[rank]
 originalpos=d["non_tensors"]["dt_direct_target_artifact"][row]["retained_response_positions"][slot]
 r["largest_actual_actor_positions"].append({"rank":rank,"training_row":row,"traj_uid":str(d["non_tensors"]["traj_uid"][row]),"actor_response_slot":slot,"original_response_slot":originalpos,"token_id":int(d["tensors"]["responses"][row,slot]),"raw_A":raw[globalrow,slot].item(),"actor_A":adv[globalrow,slot].item()})
print(json.dumps(r))

PY
