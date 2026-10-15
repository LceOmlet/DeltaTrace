"""Observe the actual native head adjoint in the existing partial-state observer.

The only extension reads the incoming native gradient; it returns None and
never substitutes a seed, gradient, loss, model, update or normalization.
"""
import hashlib
from pathlib import Path

source = Path(__file__).with_name('observe_upper_adapter_partition_v81.py')
expected = 'ef401f39e7f292dac5afd656e4e86c1246c3105364d9bad84b40284dcaa97327'
assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
code = source.read_text()
needle = "exec(compile(code, str(source), 'exec'), globals())"
assert code.count(needle) == 1
extension = '''
old_forward = """        self.micro+=1
        return self.forward(*args,**kwargs)"""
new_forward = """        self.micro+=1
        micro_batch = args[0] if args else kwargs['micro_batch']
        fields = {}
        for name in ('input_ids', 'attention_mask', 'position_ids', 'responses'):
            if name in micro_batch:
                value = micro_batch[name].detach().to(device='cpu').contiguous()
                fields[name] = dict(shape=list(value.shape), dtype=str(value.dtype),
                    SHA256=hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest())
        self.native_forward_inputs.append(dict(micro=self.micro, fields=fields))
        return self.forward(*args,**kwargs)"""
assert code.count(old_forward) == 1
code = code.replace(old_forward, new_forward)
code = code.replace('        self.branch_forward=[];self.branch_backward=[];self.branch_owners=[]',
    '        self.branch_forward=[];self.branch_backward=[];self.branch_owners=[];self.native_forward_inputs=[]')
code = code.replace("                'native_accumulation_readout':readout.accumulation_report(),",
    "                'native_forward_inputs':readout.native_forward_inputs,\\n                'native_accumulation_readout':readout.accumulation_report(),")
old_return = "        return {'rank':self.rank,'pid':os.getpid(),'birth':psutil.Process().create_time(),"
new_return = """        buffers = []
        for name, value in self.actor.actor_module.named_buffers():
            actual = value.detach().to(device='cpu').contiguous()
            buffers.append(dict(name=name, shape=list(actual.shape), dtype=str(actual.dtype),
                SHA256=hashlib.sha256(actual.view(torch.uint8).numpy().tobytes()).hexdigest()))
        return {'native_buffers':buffers,'rank':self.rank,'pid':os.getpid(),'birth':psutil.Process().create_time(),"""
assert code.count(old_return) == 1
code = code.replace(old_return, new_return)
old_condition = "self.case.get('boundary_seed_source_case')):"
new_condition = "self.case.get('boundary_seed_source_case') or self.case.get('read_boundary_seed_reference_case')):"
assert code.count(old_condition) == 1
code = code.replace(old_condition, new_condition)
old_capture = "                path=DEST/f'seed-{self.case[\\"name\\"]}-rank{self.rank}-micro{micro}.pt'"
new_capture = """                reference = self.case.get('read_boundary_seed_reference_case')
                if reference:
                    path = DEST/f'seed-{reference}-rank{self.rank}-micro{micro}.pt'
                    saved = torch.load(path, map_location='cpu', weights_only=True)
                    current = gradient.detach().to(device='cpu', copy=True)
                    assert saved.shape == current.shape and saved.dtype == current.dtype
                    a, b = saved.reshape(-1), current.reshape(-1)
                    aa = bb = ab = 0.0
                    for start in range(0, a.numel(), 65536):
                        left = a[start:start+65536].double()
                        right = b[start:start+65536].double()
                        aa += float(left.square().sum())
                        bb += float(right.square().sum())
                        ab += float((left * right).sum())
                    record.update(mode='observe_native_head_adjoint_no_replacement',
                        reference=str(path), reference_SHA256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        reference_sumsq=aa, current_sumsq=bb, dot_with_reference=ab,
                        applied_norm=record['original_incoming_norm'])
                    self.seed_records.append(record)
                    return None
                path=DEST/f'seed-{self.case["name"]}-rank{self.rank}-micro{micro}.pt'"""
assert code.count(old_capture) == 1
code = code.replace(old_capture, new_capture)
'''
code = code.replace(needle, extension + '\n' + needle)
exec(compile(code, str(source), 'exec'), globals())
