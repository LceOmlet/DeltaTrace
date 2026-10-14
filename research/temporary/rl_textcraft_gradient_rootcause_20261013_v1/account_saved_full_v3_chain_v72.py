"""Account completed full-v3 native derivative records locally, without models."""
import hashlib
from pathlib import Path

source = Path(__file__).with_name('account_common_update_chain_v58.py')
expected = '07c78c6ae969cdb213c446df035d0b63957dae41e819666911978c289b0ae85c'
assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
code = source.read_text(encoding='utf-8')
changes = {
    'Locate completed v47 gain changes in actual saved adjoints/gradient shards.':
        'Locate completed full-v3 gain changes in actual saved adjoints/gradient shards.',
    "source=HERE/'read_native_gradient_chain_v4-dt.json'":
        "source=HERE/'native-full-current-v3-feedback-v69-status-final.json'",
    'without_constant_positive_component': 'approved_v3_full_method',
    'completed-common-update-chain-v58.json': 'completed-full-current-v3-chain-v72.json',
}
for old, new in changes.items():
    assert old in code, old
    code = code.replace(old, new)
exec(compile(code, str(source), 'exec'), globals())
report.update(accounting_source=dict(path=str(Path(__file__).with_name(
    'account_common_update_chain_v58.py')), SHA256=expected),
    rootcause_repair_verified=False)
report['band_gain_ratios'] = {}
for arm in report['arms']:
    before = report['cases'][arm + '_before']['layers']
    after = report['cases'][arm + '_after']['layers']
    report['band_gain_ratios'][arm] = {
        f'{first}-{first+7}':
            (after[str(first)]['input_norm'] / after[str(first+7)]['output_norm']) /
            (before[str(first)]['input_norm'] / before[str(first+7)]['output_norm'])
        for first in (0, 8, 16, 24)
    }
path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print('band gain ratios', report['band_gain_ratios'])
