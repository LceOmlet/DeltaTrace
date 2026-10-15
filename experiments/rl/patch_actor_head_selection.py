"""Honor the caller's existing HF logits_to_keep selection in VERL's Torch head.

Skip only complete original512-token chunks that contain no requested row.
Never repack hidden rows or shift a GEMM's original chunk boundaries. The
decoder and the original FusedLinearForPPO remain the computation owners.
Default logits_to_keep=0 preserves the original path without an extra view.
This candidate was verified but not deployed: the actual TextCraft prompt is
shorter than one head chunk, so it has no measured current-workload benefit.
"""
from pathlib import Path


def patch_torch_head_selection(source):
    if '\r\n' in source:
        return patch_torch_head_selection(source.replace('\r\n','\n')).replace('\n','\r\n')
    start=source.index('def forward_with_torch_backend(')
    end=source.index('\ndef forward_with_triton_backend(',start)
    before=source[start:end]
    if 'logits_to_keep = kwargs.pop("logits_to_keep", 0)' in before:
        return source
    anchors=[
        (
            '    outputs = self.model(input_ids, **kwargs)\n',
            '    logits_to_keep = kwargs.pop("logits_to_keep", 0)\n'
            '    outputs = self.model(input_ids, **kwargs)\n',
        ),
        (
            '    fused_linear_for_ppo = FusedLinearForPPO()\n',
            '    selected_rows = None\n'
            '    if not isinstance(logits_to_keep, int) or logits_to_keep:\n'
            '        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep\n'
            '        selected_rows = torch.arange(hidden_states.shape[0] * hidden_states.shape[1], device=hidden_states.device)\n'
            '        selected_rows = selected_rows.view(hidden_states.shape[:2])[:, slice_indices].reshape(-1)\n\n'
            '    fused_linear_for_ppo = FusedLinearForPPO()\n',
        ),
        (
            '        temperature=temperature,\n    )\n    return Qwen3VLCausalLMOutputForPPO(\n',
            '        temperature=temperature,\n        selected_rows=selected_rows,\n    )\n'
            '    if selected_rows is not None:\n'
            '        log_probs = log_probs[:, slice_indices]\n'
            '        entropy = entropy[:, slice_indices]\n'
            '    return Qwen3VLCausalLMOutputForPPO(\n',
        ),
    ]
    after=before
    for old,new in anchors:
        if after.count(old)!=1:raise ValueError('Pinned Torch head selection anchor changed: '+old)
        after=after.replace(old,new,1)
    return source[:start]+after+source[end:]


def patch_owner_selected_chunks(source):
    """Filter the original owner loop without changing any computed chunk."""
    replacements=[
        (
            '        chunk_size: int = 512,\n    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:\n',
            '        chunk_size: int = 512,\n        selected_rows: Optional[torch.LongTensor] = None,\n'
            '    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:\n',
        ),
        (
            '        # Allocate memory for outputs\n        output_requires_grad = hidden_states.requires_grad or vocab_weights.requires_grad\n',
            '        chunk_starts = range(0, T, chunk_size) if selected_rows is None else (\n'
            '            (selected_rows // chunk_size).unique(sorted=True).cpu().tolist()\n'
            '        )\n'
            '        if selected_rows is not None:\n'
            '            chunk_starts = [index * chunk_size for index in chunk_starts]\n\n'
            '        # Allocate memory for outputs\n        output_requires_grad = hidden_states.requires_grad or vocab_weights.requires_grad\n',
        ),
        (
            '        ctx.chunk_size = chunk_size\n',
            '        ctx.chunk_size = chunk_size\n        ctx.chunk_starts = chunk_starts\n',
        ),
        (
            '            None,  # chunk_size\n',
            '            None,  # chunk_size\n            None,  # selected_rows\n',
        ),
        (
            '            None,  # selected_rows\n        )\n',
            '            None,  # selected_rows\n        )[:len(ctx.needs_input_grad)]\n',
        ),
        (
            '        temperature: float = 1.0,\n    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:\n',
            '        temperature: float = 1.0,\n        selected_rows: Optional[torch.LongTensor] = None,\n'
            '    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:\n',
        ),
        (
            '            self.chunk_size,\n',
            '            self.chunk_size,\n            selected_rows,\n',
        ),
    ]
    if '        ctx.chunk_starts = chunk_starts\n' in source:return source
    for old,new in replacements:
        if source.count(old)!=1:raise ValueError('Pinned selected-chunk owner anchor changed: '+old)
        source=source.replace(old,new,1)
    loop='        for chunk_start in range(0, T, chunk_size):\n'
    assert source.count(loop)==2
    source=source.replace(loop,'        for chunk_start in chunk_starts:\n',1)
    source=source.replace(loop,'        for chunk_start in ctx.chunk_starts:\n',1)
    return source


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('owner_root',type=Path)
    args=p.parse_args()
    for relative,patch in [('verl/models/transformers/qwen3_vl.py',patch_torch_head_selection),
                           ('verl/utils/experimental/torch_functional.py',patch_owner_selected_chunks)]:
        path=args.owner_root/relative
        value=patch(path.read_bytes().decode('utf-8'))
        compile(value,str(path),'exec');path.write_bytes(value.encode('utf-8'))
