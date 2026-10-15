"""Reuse the accepted passive recorder without replacing v4 computation."""
import ast
import copy
import hashlib
from pathlib import Path
import textwrap

HERE=Path(__file__).resolve().parent
ROOT='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922'
REMOTE=ROOT+'/receipts/textcraft-weighted-reward-targets-20261015-v4'
baseline=(HERE/'candidate-source/reward_readout.py').read_text()
prior=(HERE/'baseline/passive_reward_readout.py').read_text()
tree=ast.parse(prior)
cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='DirectActionTargetReadout')
helper=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='_record_joint_dt_batch')
helper_text=textwrap.indent(textwrap.dedent(ast.get_source_segment(prior,helper)),'    ')
helper_text=helper_text.replace('readout_source=__file__, rows=rows)',
    "readout_source=__file__, rows=rows,\n                        credit_transform='linear_finite_allocation',\n                        coefficient_kind='official_GRPO_group_multiplier',\n                        source_log_ratios_is_legacy_field_name=True)")
anchor='    @torch.no_grad()\n    def trajectories(self, rows: list[dict[str, Any]])'
assert baseline.count(anchor)==1
instrumented=baseline.replace(anchor,helper_text+'\n\n'+anchor)
anchor="            print(f'[DT direct joint minibatch]"
assert instrumented.count(anchor)==1
instrumented=instrumented.replace(anchor,'            self._record_joint_dt_batch(batch, detail)\n'+anchor)

class RemoveRecorder(ast.NodeTransformer):
    def visit_ClassDef(self,node):
        if node.name=='DirectActionTargetReadout':
            node.body=[x for x in node.body if not(isinstance(x,ast.FunctionDef) and x.name=='_record_joint_dt_batch')]
        return self.generic_visit(node)
    def visit_Expr(self,node):
        if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='_record_joint_dt_batch':return None
        return self.generic_visit(node)

assert ast.dump(RemoveRecorder().visit(ast.parse(instrumented)),include_attributes=False)==ast.dump(ast.parse(baseline),include_attributes=False)
path=HERE/'passive_reward_readout_v4.py';path.write_text(instrumented,encoding='utf-8',newline='\n')
installer=(HERE/'baseline/install_passive_records_native_graph.py').read_text()
installer=installer.replace("SOURCE = ROOT + '/receipts/direct-credit-records-20261009-v1/reward_readout.py'",
    "SOURCE = ROOT + '/receipts/textcraft-weighted-reward-targets-20261015-v4/source/passive_reward_readout_v4.py'")
installer=installer.replace('814cfe929b4afcc5ce3570450ea8aca269b1097db7abed83dc917df3f1d1cc9b',hashlib.sha256(baseline.encode()).hexdigest())
installer=installer.replace('2609d93f91e9e16b8963ed5ca2226a00e28190005dbd51e56a214d3d35696eb2',hashlib.sha256(path.read_bytes()).hexdigest())
(HERE/'install_passive_records_v4.py').write_text(installer,encoding='utf-8',newline='\n')
print('passive_recorder_removed_AST_exactly_equals_tested_v4_owner')
