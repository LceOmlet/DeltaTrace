"""Split the unchanged finite FA residual using two original public FA calls.

These are diagnostic mixed operands, not new model endpoints, token credits,
counterfactual answers, or a finite-rule candidate. Exact original dense input
artifacts and call arguments are retained by the original passive capture.
"""
import hashlib
import inspect
import os
from pathlib import Path
import time

import torch

from passive_attention_gate import PassiveAttentionGate


class PassiveAttentionPV(PassiveAttentionGate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.read_input
        self.mixed_calls = 0
        self.mixed_seconds = 0.
        self.native_device = {}
        self.mixed = {}
        self.artifacts = []
        self.artifact_seconds = 0.

    def extra_native_names(self):
        return {'dense_q', 'dense_k', 'dense_v'}

    def native_capture_type(self, capture_type):
        observer = self
        class KeepOriginalDense(capture_type):
            def retain(self, name, value):
                if name in ('dense_q', 'dense_k', 'dense_v'):
                    observer.native_device[name] = value.detach()
                else:
                    super().retain(name, value)
        return KeepOriginalDense

    def native_capture_finished(self, capture):
        begin = time.perf_counter()
        q,k,v = (self.native_device[n] for n in ('dense_q','dense_k','dense_v'))
        # The saved original native experiment uses all-one attention masks
        # and equal full extents. Use its actual dense owner and arguments;
        # no unpadding, mask, softmax, attention or backend is reimplemented.
        native = capture.native_dense
        arguments = capture.dense_arguments
        assert capture.calls['native_dense'] == 1
        assert arguments['causal'] and not arguments['return_attn_probs']
        width = q.shape[1]
        valueR = self.gate_bank['value'][0::2, :, :width].transpose(1,2)
        assert valueR.shape == v[1::2].shape
        valueR = valueR.repeat_interleave(2, dim=0).to(v.device, dtype=v.dtype)
        self.mixed['reference_values'] = self.cpu(native(q,k,valueR,**arguments))
        self.mixed_calls += 1
        del valueR
        queryF = q[1::2].repeat_interleave(2, dim=0)
        keyF = k[1::2].repeat_interleave(2, dim=0)
        self.mixed['factual_weights'] = self.cpu(native(queryF,keyF,v,**arguments))
        self.mixed_calls += 1
        self.mixed_owner = dict(path=inspect.getsourcefile(native),
            sha256=hashlib.sha256(Path(inspect.getsourcefile(native)).read_bytes()).hexdigest(),
            arguments=arguments, query_shape=list(q.shape),key_shape=list(k.shape),value_shape=list(v.shape),
            actual_dtype=str(q.dtype),new_FA_calls_this_native_forward=2,
            dense_original_view_identity_matches={name:capture._view_identity(tensor)==capture._interface_views[source]
                for name,source,tensor in (('q','query',q),('k','key',k),('v','value',v))})
        self.native_device.clear()
        self.mixed_seconds += time.perf_counter()-begin
        tick=time.perf_counter()
        if not self.artifacts:
            self.save_original_operands('joint',dict(bank={n:self.gate_bank[n] for n in
                ('query','key','value','dq','dk','dv','mc','o')}))
        self.save_original_operands('native-'+str(self.mixed_calls//2-1),dict(
            endpoints={n:self.native_gate[n] for n in ('query','key','value','attention_output')},
            queries=self.active['queries'],owner=self.mixed_owner))
        self.artifact_seconds += time.perf_counter()-tick

    def save_original_operands(self, name, data):
        path=self.artifact_root/f'pid{os.getpid()}-batch{self.batch_index}-{name}.pt'
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream:
            torch.save(dict(**data,worker_pid=os.getpid(),batch=self.batch_index,
                trajectories=[r['row']['traj_uid'] for r in self.rows],starts=self.starts,
                actual_context_lengths=[r['selected'].numel() for r in self.rows],
                scope='Actual original operator operands only, not a checkpoint or candidate'),stream)
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for piece in iter(lambda:stream.read(8<<20),b''):
                digest.update(piece)
        self.artifacts.append(dict(path=str(path),bytes=path.stat().st_size,sha256=digest.hexdigest()))

    def finish_point(self, point, data):
        super().finish_point(point, data)
        begin = time.perf_counter()
        row = next(i for i,q in enumerate(self.active['queries'])
            if q is not None and self.rows[i]['row']['traj_uid']==point['traj_uid'])
        start = self.starts[row]
        length = self.rows[row]['selected'].numel()-start
        bank,native = self.gate_bank,self.native_gate
        route,value,background,content,control = 0.,0.,0.,0.,0.
        native_QKV = {}
        for operand,coefficient in (('query','dq'),('key','dk'),('value','dv')):
            m,pair = bank[coefficient],native[operand]
            total=0.
            for first in range(0,length,128):
                last=min(first+128,length)
                f=pair[2*row+1,:,start+first:start+last].double()
                d=pair[2*row,:,start+first:start+last].double()
                c=m[row,:,first:last].double().reshape(pair.shape[1],-1,last-first,m.shape[-1]).sum(1)
                total += float((c*(f-d)).sum())
            native_QKV[operand]=total
        for first in range(0,length,128):
            last=min(first+128,length)
            m=bank['mc'][row,first:last].double()
            def native_o(endpoint):
                return native['attention_output'][2*row+endpoint,start+first:start+last].double()
            def mixed_o(name,endpoint):
                return self.mixed[name][2*row+endpoint,start+first:start+last].double()
            OF,OD=native_o(1),native_o(0)
            OFR,ODR=mixed_o('reference_values',1),mixed_o('reference_values',0)
            OFD,OFF=mixed_o('factual_weights',0),mixed_o('factual_weights',1)
            route += float((m*(OFR-ODR)).sum())
            value += float((m*(OF-OFD)).sum())
            background += float((m*((OFR-ODR)-(OFD-OD))).sum())
            content += float((m*(OF-OD)).sum())
            control=max(control,float((OFF-OF).abs().max()))
        route_error=native_QKV['query']+native_QKV['key']-route
        value_error=native_QKV['value']-value
        native_core=sum(native_QKV.values())-content
        matched_core=point['final_FA_input_ledger']['finite_FA_core_residual']
        point['final_FA_PV_ledger']=dict(
            native_QKV_contractions=native_QKV,native_content_contraction=content,
            reference_V_route_contraction=route,factual_P_value_contraction=value,
            joint_QK_softmax_residual=route_error,factual_P_value_residual=value_error,
            PV_background_residual=background,native_pair_core_residual=native_core,
            matched_minus_native_endpoint_residual=matched_core-native_core,
            decomposition_roundoff=route_error+value_error+background-native_core,
            original_FA_factual_replay_maxabs=control,actual_owner=self.mixed_owner,
            extra_FA_calls_cumulative=self.mixed_calls,extra_readout_seconds_cumulative=self.mixed_seconds,
            original_operator_inputs=[self.artifacts[0],self.artifacts[-1]],
            interpretation='Native F/D pair; original joint reference V. The exact rounded-output ledger '
                'separates QK/softmax propagation, factual-P value propagation, and PV background. '
                'Each includes applicable native arithmetic; this is not a causal share or credit correction. '
                'DT/native factual drift is separate. The replay control is an observation, not a new tolerance.')
        self.mixed_seconds += time.perf_counter()-begin

    def bytes(self):
        return super().bytes()+sum(v.numel()*v.element_size() for v in self.mixed.values())
