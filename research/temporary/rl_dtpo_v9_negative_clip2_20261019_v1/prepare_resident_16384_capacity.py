"""Resize only the existing artificial capacity fixture to native B4/16384."""
import hashlib
from pathlib import Path

owner = Path(__file__).with_name('prepare_root_pageable_capacity.py')
assert hashlib.sha256(owner.read_bytes()).hexdigest() == 'f5a5bfb67f6bf2fc028bee762d2b805d1e83017d93561373fc86bb67f937e5d8'
code = owner.read_text()
anchor = "exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), namespace)"
assert code.count(anchor) == 1
code = code.replace(anchor, '''class CapacityExtent(ast.NodeTransformer):
    def visit_Constant(self, node):
        if node.value == 32768:
            return ast.copy_location(ast.Constant(value=16384), node)
        if node.value == 'capacity-32768':
            return ast.copy_location(ast.Constant(value='capacity-16384'), node)
        return node
functions = [CapacityExtent().visit(copy.deepcopy(node)) for node in functions]
''' + anchor)
code = code.replace('(4,32768)', '(4,16384)')
code = code.replace('root-pageable-capacity-inputs', 'resident-16384-capacity-inputs')
code = code.replace("receipt = dict(status=", "receipt = dict(fixture_extent=16384,fixture_change='Only capacity extent constants; original native request owner retained',status=")
code = code.replace('Original artificial B4/exact-32768 storage fixture.', 'Original artificial B4/exact-16384 storage fixture for simultaneous eight-rank GPU-retained replay.')
exec(compile(code, str(owner), 'exec'), globals())
