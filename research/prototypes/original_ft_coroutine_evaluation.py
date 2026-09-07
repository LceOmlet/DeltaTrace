"""Batch independent original FT curves without placeholder scores or new metrics.

Only the two original model-score call sites become coroutine waits. Native
token logprob tensors resume the unchanged .sum()/deletion/metric code.
"""
import ast,copy,hashlib,inspect
from original_ft_batched_evaluation import evaluate_requests

def make_original_curve_generator(module):
    original=module.faithfulness_test_skip_tokens
    source=inspect.getsource(original);tree=ast.parse(source)
    original_ast=ast.dump(tree)
    class SuspendNativeScore(ast.NodeTransformer):
        count=0
        def visit_Call(self,node):
            if isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='llm_evaluator' and node.func.attr=='compute_logprob_response_given_prompt':
                assert len(node.args)==2 and not node.keywords
                self.count+=1
                return ast.copy_location(ast.Yield(value=ast.Tuple(elts=node.args,ctx=ast.Load())),node)
            return self.generic_visit(node)
    change=SuspendNativeScore();tree=change.visit(tree);assert change.count==2
    class Restore(ast.NodeTransformer):
        def visit_Yield(self,node):
            assert isinstance(node.value,ast.Tuple) and len(node.value.elts)==2
            return ast.Call(func=ast.Attribute(value=ast.Name(id='llm_evaluator',ctx=ast.Load()),attr='compute_logprob_response_given_prompt',ctx=ast.Load()),args=node.value.elts,keywords=[])
    assert ast.dump(Restore().visit(copy.deepcopy(tree)))==original_ast
    namespace=dict(vars(module))
    exec(compile(ast.fix_missing_locations(tree),'<original_ft_curve_coroutine>','exec'),namespace)
    return namespace[original.__name__],{'original_function_source_sha256':hashlib.sha256(source.encode()).hexdigest(),
        'suspended_native_calls_in_source':2,'restored_AST_identical':True,'placeholder_scores':False}

def evaluate_curves_batched(evaluator,module,jobs,batch_size,activity,request_observer=None):
    """jobs key -> kwargs of original faithfulness_test_skip_tokens except evaluator.

    Batching is across ready curves, grouped by exact input/response length by
    the already verified scheduler. Actual batch sizes may be below the limit.
    """
    import torch
    generator,proof=make_original_curve_generator(module)
    activity.update(source_proof=proof,physical_evaluation_forwards=0,evaluation_trajectories=0,
                    actual_batch_sizes=[],curves={key:[] for key in jobs})
    generators={key:generator(evaluator,**kwargs) for key,kwargs in jobs.items()}
    pending={key:next(gen) for key,gen in generators.items()};results={}
    class CaptureActualTokenValues:
        def __init__(self):self.rows=[]
        def compute_logprob_response_given_prompt(self,prompt,response):
            # Original evaluator runs unchanged. This wrapper records its real
            # return tensor; it never supplies a guessed or placeholder value.
            activity['physical_evaluation_forwards']+=1
            activity['evaluation_trajectories']+=prompt.shape[0]
            activity['actual_batch_sizes'].append(prompt.shape[0])
            value=evaluator.compute_logprob_response_given_prompt(prompt,response)
            self.rows.extend(value[i:i+1].detach() for i in range(value.shape[0]))
            return value
    with torch.no_grad():
        while pending:
            if request_observer is not None:
                for key,(prompt,response) in pending.items():request_observer(key,prompt,response)
            recorder=CaptureActualTokenValues()
            scores,cost=evaluate_requests(recorder,pending,batch_size)
            assert len(scores)==len(recorder.rows)==cost['evaluation_trajectories']
            actual=dict(zip(scores,recorder.rows))
            for key,value in actual.items():assert float(value.sum())==scores[key]
            following={}
            for key in pending:
                activity['curves'][key].append(scores[key])
                try:following[key]=generators[key].send(actual[key])
                except StopIteration as done:results[key]=done.value
            pending=following
    assert set(results)==set(jobs)
    return results
