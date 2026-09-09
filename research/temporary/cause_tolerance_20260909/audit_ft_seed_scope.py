"""Read the two fixed FT sources with AST; never import or modify FT."""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('release','environment','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();env=json.loads(args.environment.read_bytes());protocol=json.loads((args.release/'experiments/official/protocol.json').read_bytes())
    paths={'qwen3':Path(env['qwen3']['official_root'])/'ft_ifr_improve.py',
           'qwen35':Path(env['qwen35']['ft_extension_root'])/'flashtrace/improved.py'}
    report={'status':'complete','scope':'read-only source comparison; fixed FT unchanged','models':{}}
    for family,path in paths.items():
        raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        if family=='qwen3':assert hashlib.sha256(raw.replace(b'\r\n',b'\n')).hexdigest()==protocol['official_normalized_sources']['ft_ifr_improve.py']
        else:assert blob==env['qwen35']['official_extension_blob_sha1']['flashtrace/improved.py']
        nodes=[n for n in ast.walk(ast.parse(raw.decode())) if isinstance(n,ast.FunctionDef) and n.name=='calculate_ifr_multi_hop_both']
        assert len(nodes)==1;calls=[]
        for n in ast.walk(nodes[0]):
            if not isinstance(n,ast.Call):continue
            keywords={k.arg:ast.unparse(k.value) for k in n.keywords}
            if all(k in keywords for k in ('sink_start','sink_end','sink_weights')):
                calls.append({'line':n.lineno,'arguments':{k:keywords[k] for k in ('sink_start','sink_end','sink_weights')}})
        calls.sort(key=lambda c:c['line']);assert len(calls)==2
        report['models'][family]={'path':str(path),'sha256':digest,'git_blob_sha1':blob,'aggregate_calls':calls}
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
