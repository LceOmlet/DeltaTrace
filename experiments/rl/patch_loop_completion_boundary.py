"""Expose LOOP's already-normalized completion request to the VERL transport.

The original client owns optional defaults and request construction. Without
the explicitly attached callback its HTTP path and result parser are unchanged.
"""
from pathlib import Path
import sys


def patch(source):
    if 'transport = getattr(self, "_completion_transport", None)' in source:
        return source
    model = '        model_id = lora_model_id if lora_model_id is not None else self._get_base_model_id()\n'
    start = ('        transport = getattr(self, "_completion_transport", None)\n'
             '        model_id = lora_model_id if lora_model_id is not None else (\n'
             '            self._get_base_model_id() if transport is None else None\n'
             '        )\n')
    anchor = '        args["presence_penalty"] = presence_penalty\n'
    hook = (anchor + '\n'
            '        if transport is not None and not self._cancelled():\n'
            '            return transport(args)\n')
    if source.count(model) != 1 or source.count(anchor) != 1:
        raise ValueError('Pinned LOOP completion boundary changed')
    return source.replace(model, start, 1).replace(anchor, hook, 1)


if __name__ == '__main__':
    path = Path(sys.argv[1]) / 'phi_agents/rl/vllm_client.py'
    source = patch(path.read_text())
    compile(source, str(path), 'exec')
    path.write_text(source, newline='\n')
