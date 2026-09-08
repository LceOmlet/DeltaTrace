"""Only accept native None/empty unused FA return; retain the failed V1."""
import ast,base64,copy,hashlib,json,shlex,zlib
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';D=A/'snapshot${ARTIFACT_ROOT}/codex_dt_MH0_FA19_PV_orders_20260909_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_bytes())
p=read(D/'protocol.json');r=read(D/'results.json');receipt=read(D/'terminal_receipt.json')
assert receipt.get('proc_exists',receipt.get('pid_alive')) is False
for n,v in receipt['files'].items():assert sha(D/n)==(v if isinstance(v,str) else v['sha256'])
assert r['status']=='failed' and r['native_FA_entered']==r['native_FA_returned']==1
assert r['finite_FA_entered']==r['finite_FA_returned']==0 and 'assert isinstance(unused,torch.Tensor)' in r['error']
old=(D/'study.py').read_text(encoding='utf-8')
before="""    assert isinstance(unused,torch.Tensor)
    r['public_FA_unused_return_numel']=unused.numel()
    assert unused.numel()==0,'Unexpected explicit attention probability output.'
"""
after="""    r['public_FA_unused_return_kind']='None' if unused is None else type(unused).__name__
    assert unused is None or isinstance(unused,torch.Tensor)
    r['public_FA_unused_return_numel']=0 if unused is None else unused.numel()
    assert r['public_FA_unused_return_numel']==0,'Unexpected explicit attention probability output.'
"""
assert old.count(before)==1;new=old.replace(before,after)
assert ast.dump(ast.parse(old.replace(before,'')))==ast.dump(ast.parse(new.replace(after,'')))
target=R/'research/reproduction_templates/dt_MH0_FA19_PV_orders_20260909_v2.py'
assert not target.exists();target.write_text(new,encoding='utf-8')
files={n:(D/n).read_bytes() for n in p['files_sha256']}
for n,v in p['files_sha256'].items():assert hashlib.sha256(files[n]).hexdigest()==v
files['study.py']=target.read_bytes();p=copy.deepcopy(p)
p['version']='v2_only_native_unused_return_None_or_empty_guard'
p['v1_failure_provenance']={'directory':'/tmp/'+D.name,'results_sha256':sha(D/'results.json'),'protocol_sha256':sha(D/'protocol.json'),
    'receipt_sha256':sha(D/'terminal_receipt.json'),'study_sha256':sha(D/'study.py'),'seconds':r['seconds'],
    'native_FA_entered':1,'native_FA_returned':1,'finite_FA_entered':0,'finite_FA_returned':0,
    'cause':'Native no-dropout FA returned None for unused probabilities. The V1 Tensor-only interface guard failed before any finite FA call; this is not FA/method failure.',
    'patch_scope':'Only accept None or empty Tensor and record actual kind/zero numel. AST excluding this guard is identical; mathematics, operands, precision and calls unchanged.'}
p['protected_sources'] += [{'path':'/tmp/'+D.name+'/'+n,'sha256':sha(D/n)} for n in ('results.json','protocol.json','terminal_receipt.json')]
p['numerical_policy']=p['numerical_policy'].replace('actual unused return numel must0','actual unused return must None or empty Tensor with recorded kind and numel0')
p['files_sha256']={n:hashlib.sha256(v).hexdigest() for n,v in files.items()}
for n,v in files.items():
    if n.endswith('.py'):ast.parse(v,filename=n)
files['protocol.json']=json.dumps(p,indent=2).encode()
pp=A/'dt_MH0_FA19_PV_orders_protocol_20260909_v2.json';lp=A/'launch_dt_MH0_FA19_PV_orders_20260909_v2.json'
assert not pp.exists() and not lp.exists();remote='${ARTIFACT_ROOT}/codex_dt_MH0_FA19_PV_orders_20260909_v2';py='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python'
blob=base64.b64encode(zlib.compress(json.dumps({n:base64.b64encode(v).decode() for n,v in files.items()}).encode())).decode()
loader=('import pathlib,subprocess,json,base64,zlib;d=pathlib.Path('+repr(remote)+');d.mkdir(exist_ok=False);'
    'files=json.loads(zlib.decompress(base64.b64decode('+repr(blob)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];'
    'f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(py)+',"-B",str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
    '(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))')
pp.write_bytes(files['protocol.json']);lp.write_text(json.dumps({'cmd':py+' -c '+shlex.quote(loader),'timeout':10}))
print(json.dumps({'payload':str(lp),'protocol_sha256':sha(pp),'study_sha256':p['files_sha256']['study.py'],'remote_directory':remote,'v1_counts':p['v1_failure_provenance']}))
