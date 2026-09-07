"""Run the unchanged author sampler with a frozen unused original-source cohort.

Default mode only checks readiness and never sends an API request. --execute
requires configured API base/key; key values never enter argv or audit output.
Actual sampler call/return observations preserve responses and usage, while
the original prompt generation, judge, format filter and span logic execute.
"""
import hashlib,json,os,runpy,sys,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent
p=json.loads((HERE/'protocol.json').read_text())
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
assert sha(Path(__file__))==p['study_sha256']
root=Path(p['original_repository']);source=root/'exp/exp2/sample_and_filter.py'
for name,digest in p['original_normalized_sources'].items():assert hashlib.sha256((root/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==digest
input_file=HERE/'sampling_original_source.json'
assert sha(input_file)==p['sampling_source_sha256']
rows=json.loads(input_file.read_text());assert len(rows)==p['maximum_raw_examples']
base=os.environ.get('FLASHTRACE_API_BASE') or os.environ.get('OPENAI_BASE_URL') or os.environ.get('OPENAI_API_BASE')
key_present=bool(os.environ.get('FLASHTRACE_API_KEY') or os.environ.get('OPENAI_API_KEY'))
readiness={'status':'ready' if base and key_present else 'missing_API_configuration','source_validated':True,
    'API_base_configured':bool(base),'API_key_configured':key_present,'API_calls':0,'execution_requested':'--execute' in sys.argv,
    'required_model_names':[p['generator_model'],p['judge_model']]}
(HERE/'readiness.json').write_text(json.dumps(readiness,indent=2));print(json.dumps(readiness),flush=True)
if '--execute' not in sys.argv or not(base and key_present):raise SystemExit(0)
if (HERE/'actual_api_events.jsonl').exists():raise RuntimeError('Prior execution evidence exists: inspect it; do not silently restart or duplicate generation.')
activity={'status':'running','protocol_sha256':sha(HERE/'protocol.json'),'started_at_unix':time.time(),'attempted_API_calls':0,'returned_API_calls':0,'generator_attempts':0,'judge_attempts':0,'reported_usage':[]}
def save():
    temp=HERE/'results.partial';temp.write_text(json.dumps(activity,indent=2));temp.replace(HERE/'results.json')
def event(record):
    with (HERE/'actual_api_events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush()
pending={}
def checkpoint_main(frame):
    while frame is not None:
        if frame.f_code.co_filename==str(source) and frame.f_code.co_name=='main':
            kept=frame.f_locals.get('kept')
            if isinstance(kept,list):
                temp=HERE/'retained_samples_checkpoint.partial'
                frame.f_globals['write_cache'](temp,kept)
                temp.replace(HERE/'retained_samples_checkpoint.jsonl')
                activity['retained_checkpoint_rows']=len(kept)
            return
        frame=frame.f_back
def observe(frame,kind,arg):
    if frame.f_code.co_filename!=str(source):return
    if frame.f_code.co_name=='main' and kind=='return':checkpoint_main(frame);save()
    if frame.f_code.co_name!='call_chat_api':return
    if kind=='call':
        model=frame.f_locals['model'];assert model in [p['generator_model'],p['judge_model']]
        activity['attempted_API_calls']+=1
        assert activity['attempted_API_calls']<=p['maximum_API_attempts']
        group='generator' if model==p['generator_model'] else 'judge';activity[group+'_attempts']+=1
        number=activity['attempted_API_calls'];pending[id(frame)]=(number,time.time())
        messages=frame.f_locals['messages']
        event({'event':'request_started','number':number,'model_requested':model,'messages':messages,
            'messages_sha256':hashlib.sha256(json.dumps(messages,ensure_ascii=False,sort_keys=True).encode()).hexdigest(),
            'max_tokens':frame.f_locals['max_tokens'],'temperature':frame.f_locals['temperature']})
        checkpoint_main(frame);save()
    elif kind=='return':
        number,started=pending.pop(id(frame));response=frame.f_locals.get('response',{})
        usage=response.get('usage') if isinstance(response,dict) else None
        event({'event':'request_returned','number':number,'seconds':time.time()-started,'returned_content':arg,
            'reported_model':response.get('model') if isinstance(response,dict) else None,
            'response_id':response.get('id') if isinstance(response,dict) else None,'usage':usage,
            'HTTP_response_received':'resp_bytes' in frame.f_locals,'successful_content':isinstance(arg,str)})
        activity['returned_API_calls']+=1
        if usage is not None:activity['reported_usage'].append(usage)
        save()
save()
old_argv=sys.argv[:];old_cwd=Path.cwd()
sys.argv=[str(source),'--dataset',str(input_file),'--max_examples',str(p['kept_examples_target']),
    '--generator_model',p['generator_model'],'--judge_model',p['judge_model'],'--tokenizer_model',p['tokenizer_model'],
    '--api_base',base,'--api_max_tokens','8192','--api_temperature','0.0','--retries','2','--seed','42',
    '--api_cache_ttl','600','--api_cache_namespace',p['cache_namespace'],'--out',str(HERE/'new_confirmation_cache.jsonl')]
try:
    os.chdir(root);sys.path.insert(0,str(root));sys.setprofile(observe)
    runpy.run_path(str(source),run_name='__main__')
    activity['status']='sampling_complete_pending_independent_cache_review'
except BaseException as exc:
    activity['status']='sampling_failed';activity['error_type']=type(exc).__name__
    # Do not serialize arbitrary exception strings that might contain API data.
    raise
finally:
    sys.setprofile(None);sys.argv=old_argv;os.chdir(old_cwd)
    activity['elapsed_seconds']=time.time()-activity['started_at_unix']
    for name in ['new_confirmation_cache.jsonl','retained_samples_checkpoint.jsonl','actual_api_events.jsonl']:
        f=HERE/name
        if f.exists():activity[name]={'sha256':sha(f),'bytes':f.stat().st_size}
    save()
