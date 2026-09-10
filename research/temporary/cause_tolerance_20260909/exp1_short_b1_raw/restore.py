"""Restore the already audited environment after the server /tmp reset."""
from pathlib import Path
import hashlib, json, os, subprocess, sys, time, traceback, zipfile

A = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
env = dict(os.environ, MACA_PATH='/opt/maca', HF_HUB_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
r = {'status':'started','model_loads':0,'steps':[]}
def save():
    (A/'restore.json').write_text(json.dumps(r,indent=2))
def run(name, cmd, timeout=240):
    save(); start=time.monotonic()
    with (A/(name+'.log')).open('w') as f:
        p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,timeout=timeout)
    r['steps'].append({'name':name,'exit':p.returncode,'seconds':time.monotonic()-start});save()
    assert p.returncode == 0, name
try:
    with zipfile.ZipFile(A/'release.zip') as z:
        for name in z.namelist():
            assert not name.startswith('/') and '..' not in Path(name).parts
        z.extractall(A/'release')
    paths=json.loads((A/'restore_payload.json').read_text())
    for name,digest in paths['files'].items():
        assert sha(A/'release'/name)==digest,name
    V=A/'qwen35env'
    run('venv',[sys.executable,'-m','venv','--system-site-packages',str(V)])
    wheels=Path('/mnt/geogpt-doc-new/deepresearch/tool_jepa_qwen35_9b/wheels')
    pinned={
      'tf513_full/transformers-5.13.0-py3-none-any.whl':'8adbc1d20bd5463cd6876b2eb7cb31971e1065788e7dc6bc12bab597a7c504b7',
      'veomni/flash_linear_attention-0.4.1-py3-none-any.whl':'d18bdfe9d1f4b424676444eac9d50fb8433b70e5d4e0e0878b20bcbcdbea57ce',
      'veomni/fla_core-0.4.1-py3-none-any.whl':'93c6afe4c80fc7bc705fa8aeea6a46d2cf2d77383f9619a41863c7114c801bab'}
    for name,digest in pinned.items(): assert sha(wheels/name)==digest,name
    py=str(V/'bin/python')
    run('pinned_install',[py,'-m','pip','install','--no-index','--disable-pip-version-check','--find-links',str(wheels/'tf513_full'),'--find-links',str(wheels/'veomni'),*[str(wheels/x) for x in pinned]])
    sys.path.insert(0,str(A/'release'))
    from fla_maca_device_mapping import apply_mapping
    r['mapping']=apply_mapping(V,V/'lib/python3.12/site-packages/fla/utils.py',A/'mapping.json')
    # Install byte-pinned previously used non-numerical dependencies in an overlay.
    requirements=[]
    for record in paths['dependency_archives']:
        requirements.append(record['url']+'#sha256='+record['sha256'])
    run('auxiliary_install',[sys.executable,'-m','pip','install','--no-deps','--disable-pip-version-check','--target',str(A/'deps'),*requirements],480)
    env['PYTHONPATH']=str(A/'deps')
    probe="""import inspect,json,hashlib,torch,transformers,triton
from transformers.models.qwen3_5 import modeling_qwen3_5 as m
from fla import utils as u
assert u.device_platform=='maca' and u.device_name=='cuda' and not u.IS_NVIDIA
assert not hasattr(torch,'maca')
assert hashlib.sha256(open(inspect.getfile(m),'rb').read()).hexdigest()=='cf085792cb59e5bdf9b88a3d20bd353892289d054662a9c2b662221b97caefba'
import wordfreq,spacy
print(json.dumps({'torch':torch.__version__,'transformers':transformers.__version__,'triton':triton.__version__,'fast_path':m.is_fast_path_available}))
"""
    run('qwen35_import',[py,'-c',probe])
    run('qwen3_import',[sys.executable,'-c',"import sys;sys.path.insert(0,'/root/flashtrace-vjp-official');import llm_attr,ft_ifr_improve;print('official_import_ok')"])
    r['status']='restored_and_imported_no_model_calls'
except Exception:
    r['status']='failed';r['error']=traceback.format_exc()
finally:
    save(); print(json.dumps(r),flush=True)
