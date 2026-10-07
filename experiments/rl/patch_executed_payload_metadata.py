"""Minimal, default-compatible metadata seams in pinned task parsers.

The owning regex and normalization still produce the executed action/code.
No parser is implemented by the training bridge.
"""
from __future__ import annotations


def textcraft_parser(source):
    if '_executed_payload_sink' in source:
        return source
    original = '        action_matches = re.findall(r"Action:\\s*(.*?)(?=\\n|$)", action, re.DOTALL)\n'
    replacement = ('        action_matches = list(re.finditer(r"Action:\\s*(.*?)(?=\\n|$)", action, re.DOTALL))\n'
                   '        payload_spans = ([list(action_matches[-1].span(0))] if action_matches else [])\n')
    action = '        action = action_matches[-1] if action_matches else ""\n'
    post = '        response = self._post("step", {"action": action})\n'
    if any(source.count(anchor) != 1 for anchor in (original, action, post)):
        raise ValueError('Pinned TextCraft parser changed')
    return source.replace(original, replacement, 1).replace(action,
        '        action = action_matches[-1].group(1) if action_matches else ""\n', 1).replace(post,
        post + '        sink = getattr(self, "_executed_payload_sink", None)\n'
        '        if sink is not None:\n'
        '            sink(dict(source_spans=payload_spans, executed_payload=action))\n', 1)


def loop_extractor(source):
    if 'return_source_spans' in source:
        return source
    signature = 'def extract_code_format_output(msg_content: str) -> str:\n'
    output = '    output_code: str = ""\n'
    full = '        code = re_match.group(2).strip()\n'
    partial = '        output_code += partial_m.group(2).strip()\n'
    empty = '        return ""\n    else:\n        return output_code\n'
    if any(source.count(anchor) != 1 for anchor in (signature, output, full, partial, empty)):
        raise ValueError('Pinned LOOP extractor changed')
    return source.replace(signature,
        'def extract_code_format_output(msg_content: str, *, return_source_spans: bool = False):\n', 1).replace(output,
        output + '    source_spans = []\n', 1).replace(full,
        full + '        source_spans.append(list(re_match.span(0)))\n', 1).replace(partial,
        '        raw_code = partial_m.group(2)\n'
        '        code = raw_code.strip()\n'
        '        left = match_end + partial_m.start(1) - 3\n'
        '        source_spans.append([left, match_end + partial_m.end(0)])\n'
        '        output_code += code\n', 1).replace(empty,
        '        return ("", source_spans) if return_source_spans else ""\n'
        '    else:\n'
        '        return (output_code, source_spans) if return_source_spans else output_code\n', 1)
