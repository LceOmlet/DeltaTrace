"""CPU-only provenance and inheritance checks for streaming storage changes."""
import argparse,ast,hashlib,json,sys
from pathlib import Path


def verify(root):
    root=Path(root).resolve();study=root/'research/temporary/qwen3_stream_memory_20260910'
    previous=root/'research/temporary/qwen3_short_efficiency_20260910'
    sys.path.insert(0,str(previous))
    from verify_graph_sources import verify as verify_base
    base=verify_base(root)
    sha=lambda data:hashlib.sha256(data).hexdigest()
    raw=(root/'deltatrace/accelerated/graphed_qwen3_streamed_sources.json').read_bytes()
    manifest=json.loads(raw)
    assert manifest['base_graphed_manifest_sha256']==sha((root/'deltatrace/accelerated/graphed_qwen3_sources.json').read_bytes())
    for name,digest in manifest['files'].items():
        data=(root/name).read_bytes();assert sha(data)==digest,name;ast.parse(data)
    derivation=json.loads((study/'production_derivation.json').read_bytes())
    for name,row in derivation['files'].items():
        original=(root/row['source']).read_bytes()
        assert sha(original)==row['source_sha256']
        transformed=original.decode()
        for before,after in derivation['import_renames'].items():transformed=transformed.replace('from '+before+' import','from '+after+' import')
        assert transformed==(root/name).read_text()
        assert sha((root/name).read_bytes())==row['sha256']
    hot=ast.parse((root/'deltatrace/accelerated/qwen3/qwen3_streamed_graph.py').read_bytes())
    cold=ast.parse((root/'deltatrace/accelerated/qwen3/qwen3_streamed_build_graph.py').read_bytes())
    classes={node.name:node for tree in [hot,cold] for node in tree.body if isinstance(node,ast.ClassDef)}
    adopted=classes['AdoptedProgram']
    assert [ast.unparse(x) for x in adopted.bases]==['Program']
    assert {x.name for x in adopted.body if isinstance(x,ast.FunctionDef)}=={'__init__','validate_adoption'}
    assert [ast.unparse(x) for x in classes['StreamedFiniteGraphQwen3'].bases]==['FiniteGraphQwen3']
    assert [ast.unparse(x) for x in classes['StreamedBuildFiniteGraphQwen3'].bases]==['StreamedFiniteGraphQwen3']
    assert [ast.unparse(x) for x in classes['StreamTape'].bases]==['NativeRootTape']
    for tree in [hot,cold]:
        calls=[ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)]
        assert 'capture_checkpoint_pair' in calls and 'self.program.replay' in calls
        assert not any(n in calls for n in ['torch.matmul','torch.mm','torch.bmm','torch.autograd.grad'])
    compact=ast.parse((root/'deltatrace/accelerated/qwen3/qwen3_streamed_compact_graph.py').read_bytes())
    omitted=('a','attn_out','mlp_in','mlp_out')
    assert ast.literal_eval(next(n.value for n in compact.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='OMITTED' for t in n.targets)))==omitted
    read_checks={}
    for filename in ['qwen3_graph_tensor_finite.py','qwen3_finite_graph.py','qwen3_root_retained.py']:
        tree=ast.parse((root/'deltatrace/accelerated/qwen3'/filename).read_bytes())
        if filename=='qwen3_root_retained.py':tree=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='NativeRootAccess')
        reads={n.slice.value for n in ast.walk(tree) if isinstance(n,ast.Subscript) and isinstance(n.ctx,ast.Load) and isinstance(n.slice,ast.Constant) and isinstance(n.slice.value,str)}
        assert not set(omitted).intersection(reads),(filename,reads)
        read_checks[filename]=sorted(reads)
    p=json.loads((study/'confirmation_v2_protocol.json').read_bytes())
    assert sha((study/'benchmark_confirmation_v2.py').read_bytes())==p['driver_sha256']
    for name,digest in p['runtime_files'].items():assert sha((root/name).read_bytes())==digest,name
    benchmark=(study/'benchmark_confirmation_v2.py').read_text()
    assert 'bench.measure(runner,[0],catch_oom=True)' in benchmark
    assert 'bench.make_attr_runner(args.method,model,tokenizer,128,32,' in benchmark
    return {'base_graph_and_clean_verification':base,'new_runtime_files_verified':len(manifest['files']),
        'metadata_only_operands':omitted,'original_numerical_program_literal_reads':read_checks,
        'stream_manifest_sha256':sha(raw),'production_matches_screened_pilot_except_import_name':True,
        'native_root_capture_function_unchanged':True,
        'original_Program_invoke_build_replay_inherited_without_overrides':True,
        'original_finite_math_FA_half_GEMMs_and_CPU_resolver_unchanged':True,
        'benchmark_uses_original_full_author_timer_and_FT_runner_factory':True,
        'frozen_confirmation_runtime_files_verified':len(p['runtime_files'])}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[3]);p.add_argument('--output',type=Path)
    args=p.parse_args();result=verify(args.repo)
    if args.output:args.output.write_text(json.dumps(result,indent=2)+'\n',newline='\n')
    print(json.dumps(result,indent=2))
