import pathlib,json,hashlib,collections,statistics,math
base=pathlib.Path('research/temporary/rl_upstream_alignment_20260929')
directory=base/'textcraft-degradation-20261005/joint-fla-20261006/v1'
packpath=base/'textcraft-degradation-20261005/native-minibatch-v4/native64-first-response-matched-layout-summary.json'
packsha='2e8390747c02801bdb763e4eac76112c0ac4caf3f513893f8218e7124e560f27'
fields=('q','k','v','beta','g')
layers=(6,8)
def provenance(p):
 p=pathlib.Path(p);b=p.read_bytes();return dict(path=str(p.resolve()),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def stats(values):
 v=[float(x) for x in values]
 assert all(math.isfinite(x) for x in v)
 if not v:return {'n':0}
 s=sorted(v)
 return dict(n=len(v),mean=statistics.mean(v),abs_mean=statistics.mean(abs(x) for x in v),rms=math.sqrt(statistics.mean(x*x for x in v)),min=min(v),max=max(v),positive=sum(x>0 for x in v),negative=sum(x<0 for x in v),zero=sum(x==0 for x in v),median=statistics.median(v))
def identity(s):
 return {k:s[k] for k in ('traj_uid','source_step','original_request_index','source_start','source_end','probe_source_positions','probe_input_positions')}
def pool(rows,key,numbers):
 members=collections.defaultdict(list)
 for row in rows:members[tuple(row[x] for x in key)].append(row)
 result=[]
 for coord,replicas in members.items():
  record={k:v for k,v in zip(key,coord)}
  record.update({k:statistics.mean(float(r[k]) for r in replicas) for k in numbers})
  record['replicas']=len(replicas)
  record['ranges']={k:dict(min=min(r[k] for r in replicas),max=max(r[k] for r in replicas),range=max(r[k] for r in replicas)-min(r[k] for r in replicas)) for k in numbers}
  record['source_coordinates']=[{k:r[k] for k in ('rank','group','slot')} for r in replicas]
  result.append(record)
 return result
pack=json.loads(packpath.read_bytes())
assert pack['original_full_pack']['sha256']==packsha
runs=[json.loads((directory/f'rank{r}-readout.json').read_bytes()) for r in (0,1)]
complete=json.loads((directory/'completed.json').read_bytes())
rawrows=[];heads=[];single_rows=[];dtypes=collections.defaultdict(set);counter=collections.Counter();calls=[];resource=[];hdelta=[];cuts=collections.Counter();owner_sources=[];capture_sources=[];completeness=[]
for rank,run in enumerate(runs):
 assert run['rank']==rank and run['phase']=='complete_native_readout' and run['input_sha256']==packsha
 assert len(run['cases'])==21 and run['optimizer_steps']==run['scheduler_steps']==run['backward_calls']==0
 calls.append(dict(rank=rank,**{k:run[k] for k in ('finite_trace_calls','finite_seed_calls','native_forward_calls','native_root_calls','native_prefix_calls','optimizer_steps','scheduler_steps','backward_calls')}))
 owner_sources.append(dict(rank=rank,sources=run['sources'],runtime_dtype={k:run[k] for k in ('model_dtype','native_fla_fp16','event_attention_backend','numerical_runner_options')}))
 groups={g['original_owner_batch_index']:g for g in pack['rank_groups'][rank]}
 full_by_group={}
 for case in run['cases']:
  number=case['original_owner_batch_index'];group=groups[number]
  assert [identity(s) for s in case['samples']]==[identity(s) for s in group['samples']]
  if case['variant']!='full_response_eos':continue
  original=case['actual_coefficient_metadata']['conditional_fla'];joint=original['joint_fla']
  full_by_group[number]=original
  slots=[i for i,s in enumerate(group['samples']) if s['source_step']==0]
  assert joint['original_slots']==original['original_slots']==slots
  assert not original['diagnostics'] and not joint['diagnostics']
  assert not joint['unconsumed_output_layer_indices']
  assert joint['retained_cpu_output_bytes_after_cleanup']==0
  assert joint['actual_public_output_returns']==joint['expected_public_output_returns']==2
  assert joint['actual_norm_input_captures']==joint['expected_norm_input_captures']==2
  assert joint['actual_original_finite_head_calls']==8
  assert joint['recorded_layer_slot_scalars']==joint['expected_layer_slot_scalars']==8*len(slots)
  assert joint['added_native_model_calls']==joint['added_native_FLA_calls']==0
  completeness.append(dict(rank=rank,group=number,slots=slots,actual_prefix_cut=case['observed_prefix_length'],counters={k:joint[k] for k in ('actual_public_output_returns','actual_norm_input_captures','actual_original_finite_head_calls','recorded_layer_slot_scalars','expected_layer_slot_scalars','added_native_model_calls','added_native_FLA_calls')}))
  for k in completeness[-1]['counters']:counter[k]+=joint[k]
  resource.append(dict(rank=rank,group=number,peak_retained_cpu_output_bytes=joint['peak_retained_cpu_output_bytes'],remaining_before_cleanup=joint['retained_cpu_output_bytes_before_cleanup'],remaining_after_cleanup=joint['retained_cpu_output_bytes_after_cleanup']))
  capture_sources.append(dict(rank=rank,group=number,owner_event=joint['original_capture_event'],owner_init=joint['original_capture_init'],owner_public=joint['original_public_FLA'],capture_backend=joint['original_capture_backend_module'],original_GDN_pullback=joint['original_GDN_pullback'],owner_token_effect=joint['owner_token_effect'],metadata=joint['capture_metadata']))
  for layer,cap in joint['capture_metadata'].items():
   assert cap['public_return_count']==cap['norm_input_count']==1
   assert cap['consumed_original_head_groups']==4
   for k,m in cap['fields'].items():dtypes[k].add(m['dtype'])
   if cap['actual_public_initial_state'] is not None:dtypes['public_initial_state'].add(cap['actual_public_initial_state']['dtype'])
  bylayerslot=collections.defaultdict(list)
  for item in joint['layers']:
   layer,slot=item['decoder_index'],item['original_slot']
   assert layer in layers and slot in slots
   sample=group['samples'][slot];assert sample['source_step']==0 and sample['observed_return']==1
   sourcegroups=original['layers'][str(layer)]['groups']
   matches=[a for a in sourcegroups if a['layout']==item['layout']]
   assert len(matches)==1
   old=matches[0]
   assert item['coordinates']==old['coordinates']
   assert item['actual_native_do']==old['incoming_do']
   assert item['actual_FLA_incoming_h']==old['incoming_state']
   assert item['incoming_first_chunk_h_pair_absmax']==old['incoming_state_pair_absmax'][str(slot)]
   assert item['native_time_start']==item['layout']['capture_start']+item['layout']['cut']==old['native_time_start']
   assert item['head_count']==old['head_count']==8 and item['head_width']==old['head_width']==128
   assert item['original_joint_input_fields']==old['actual_joint_input_projections'][str(slot)]
   assert set(item['original_joint_input_fields'])==set(fields)
   F=float(item['F_joint']);Y=float(item['Y_joint_public_output']);N=float(item['Y_joint_norm_boundary'])
   row=dict(rank=rank,group=number,slot=slot,decoder_index=layer,traj_uid=sample['traj_uid'],source_step=sample['source_step'],head_start=item['layout']['head_start'],F=F,Y=Y,N=N,F_minus_Y=F-Y,Y_minus_N=Y-N,F_minus_N=F-N,field_sum=math.fsum(item['original_joint_input_fields'].values()),F_field_closure=F-math.fsum(item['original_joint_input_fields'].values()),bridge_closure=(F-Y)+(Y-N)-(F-N),stored_difference_residuals={name:float(item[name])-val for name,val in (('F_joint_minus_Y_public',F-Y),('public_minus_norm_boundary',Y-N),('F_joint_minus_Y_norm_boundary',F-N))},fields=dict(item['original_joint_input_fields']),layout=dict(item['layout']),coordinates=dict(item['coordinates']))
   heads.append(row);bylayerslot[layer,slot].append(row)
   hdelta.append(item['incoming_first_chunk_h_pair_absmax'])
   for k in ('actual_native_do','CPU_do','actual_FLA_incoming_h'):dtypes[k].add(item[k]['dtype'])
  for (layer,slot),groupheads in bylayerslot.items():
   groupheads.sort(key=lambda a:a['head_start'])
   assert [a['head_start'] for a in groupheads]==[0,8,16,24]
   assert len(groupheads)==4
   F,Y,N=[math.fsum(a[k] for a in groupheads) for k in ('F','Y','N')]
   sample=group['samples'][slot]
   row=dict(rank=rank,group=number,slot=slot,decoder_index=layer,traj_uid=sample['traj_uid'],source_step=sample['source_step'],F=F,Y=Y,N=N,F_minus_Y=F-Y,Y_minus_N=Y-N,F_minus_N=F-N,field_sum=math.fsum(a['field_sum'] for a in groupheads),F_field_closure=F-math.fsum(a['field_sum'] for a in groupheads),bridge_closure=(F-Y)+(Y-N)-(F-N),source_start=sample['source_start'],source_end=sample['source_end'],source_input_hashes={k:group[k] for k in ('selected_input_ids_sha256','reference_input_ids_sha256')},layout=dict(groupheads[0]['layout']),coordinates=dict(groupheads[0]['coordinates']))
   rawrows.append(row);cuts[str(layer)+':'+str(row['layout']['capture_start'])]+=1
 assert len(full_by_group)==7
 for case in run['cases']:
  if case['variant']=='full_response_eos':continue
  number=case['original_owner_batch_index'];probe=int(case['variant'].rsplit('_',1)[1])
  native=case['conditional_boundary_contractions']['conditional_primitives']['conditional_gdn']['conditional_fla']
  assert not native['diagnostics']
  for item in native['layers']:
   layer,slot=item['decoder_index'],item['original_slot'];sample=case['samples'][slot]
   assert sample['source_step']==0
   F=float(item['F_input_projection']);Y=float(item['Y_public_output_projection'])
   single_rows.append(dict(rank=rank,group=number,slot=slot,decoder_index=layer,traj_uid=sample['traj_uid'],source_step=sample['source_step'],source_position=sample['probe_source_positions'][probe],probe_index=probe,F=F,Y=Y,F_minus_Y=F-Y))
numbers=('F','Y','N','F_minus_Y','Y_minus_N','F_minus_N','F_field_closure','bridge_closure')
summary=[];unique=[];singleunique=[]
for layer in layers:
 rows=[x for x in rawrows if x['decoder_index']==layer]
 assert len(rows)==26
 means=pool(rows,('traj_uid','source_step'),numbers);assert len(means)==21
 unique.extend([dict(decoder_index=layer,**r) for r in means])
 singles=[x for x in single_rows if x['decoder_index']==layer];assert len(singles)==52
 smean=pool(singles,('traj_uid','source_step','source_position'),('F','Y','F_minus_Y'));assert len(smean)==42
 singleunique.extend([dict(decoder_index=layer,**r) for r in smean])
 summary.append(dict(decoder_index=layer,transport_samples=26,unique_UIDs=21,replica_counts=dict(collections.Counter(r['replicas'] for r in means)),transport_statistics={k:stats(r[k] for r in rows) for k in numbers},unique_UID_statistics={k:stats(r[k] for r in means) for k in numbers},UID_sign_counts={'F_vs_public_opposite':sum(r['F']*r['Y']<0 for r in means),'public_vs_norm_opposite':sum(r['Y']*r['N']<0 for r in means),'F_vs_norm_opposite':sum(r['F']*r['N']<0 for r in means)},max_replica_ranges={k:max(r['ranges'][k]['range'] for r in means) for k in numbers},same_run_full_to_single_projection=dict(transport_samples=52,unique_probes=42,statistics={k:stats(r[k] for r in smean) for k in ('F','Y','F_minus_Y')},opposite_sign_probes=sum(r['F']*r['Y']<0 for r in smean)),denominator_scope='Same-full-pair uses 21 successful first-response UID means; full-to-single uses two fixed sampled positions per UID (42 unique probes). They are separate populations, not pooled.'))
result=dict(status='independent_raw_descriptive_review_no_official_tolerance_gate',analysis_backend='Python standard library only; no import of main analyzers/Torch/Ray/model/connector',sources=[provenance(directory/f'rank{r}-readout.json') for r in (0,1)]+[provenance(packpath),provenance(directory/'completed.json')]+[provenance(base/name) for name in ('observe_textcraft_joint_fla.py','verify_textcraft_joint_fla.py','stage_textcraft_joint_fla.py')],literal_full_pack_sha256=packsha,checkpoint=runs[0]['checkpoint'],original_runtime_sources=owner_sources,coverage=dict(original_groups_per_rank=[7,7],full_traces=14,single_root_cases=28,raw_head_slot_rows=len(heads),layer_transport_rows=len(rawrows),unique_layer_UID_rows=len(unique),same_run_single_transport_rows=len(single_rows),same_run_single_unique_layer_probe_rows=len(singleunique)),calls=calls,capture_counters=dict(counter),original_capture_sources_and_metadata=capture_sources,completeness=completeness,actual_dtype_sets={k:sorted(v) for k,v in dtypes.items()},incoming_first_chunk_h_pair_differences=stats(hdelta),capture_start_population=dict(cuts),raw_head_closure_statistics={k:stats(r[k] for r in heads) for k in ('F_field_closure','bridge_closure')},stored_head_difference_closure_statistics={k:stats(r['stored_difference_residuals'][k] for r in heads) for k in heads[0]['stored_difference_residuals']},CPU_output_resources=resource,CPU_output_payload=stats(r['peak_retained_cpu_output_bytes'] for r in resource),layer_summaries=summary,head_rows=heads,full_transport_rows=rawrows,unique_UID_replica_means=unique,same_run_single_transport_rows=single_rows,same_run_single_unique_probe_replica_means=singleunique,limitations=['Same-full-pair contractions use actual full-response EOS/factual endpoints at every term; full-to-single contractions reuse full-response coefficients on fixed single-token native endpoints.','No tolerance or correction is defined for non-coincident finite contraction. This does not report an official native FLA kernel acceptance result.','Actual public FP16 output and subsequent BF16 norm input are contracted with the same native do; earlier FP32 mo to BF16/FP16 seed conversion is not separated here.','21 UID are successful first responses from the existing checkpoint25 diagnostic minibatch; this is not all training samples and no gradient-impact or world causal-error rate is measured.','Transport replicas are retained with min/max dispersion; UID means are explicit, not silently deduplicated.','Counts and exact source/layout matches are observation completeness checks. Recorded arithmetic closure is descriptive and not a new numerical gate.','Peak CPU output payload is copied retained tensor bytes, not RSS/GPU capacity. Zero added native calls does not imply zero observation overhead.'])
output=directory/'independent-joint-review.json'
with output.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,separators=(',',':'),allow_nan=False);f.write('\n')
print(json.dumps(dict(output=provenance(output),coverage=result['coverage'],dtype=result['actual_dtype_sets'],calls=calls,counters=dict(counter),h0=result['incoming_first_chunk_h_pair_differences'],payload=result['CPU_output_payload'],layers=[dict(decoder_index=r['decoder_index'],same_pair={k:r['unique_UID_statistics'][k] for k in ('F','Y','F_minus_Y','Y_minus_N','F_minus_N')},signs=r['UID_sign_counts'],single_projection=r['same_run_full_to_single_projection']) for r in summary]),ensure_ascii=False))

