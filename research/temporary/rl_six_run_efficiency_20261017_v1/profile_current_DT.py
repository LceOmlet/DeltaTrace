"""Use the existing bounded, nonblocking Python and physical phase observer."""
import hashlib
from pathlib import Path
import sys

source = Path(__file__).resolve().parent.parent / 'rl_upstream_alignment_20260929/profile_current_native_phases.py'
assert hashlib.sha256(source.read_bytes()).hexdigest() == 'f1640fcf6a3df528b344531ea88e2f4e5c27f4ef1163f9b9ab748793b9bce2fb'
code = source.read_text()


def replace(before, after):
    global code
    assert code.count(before) == 1, before
    code = code.replace(before, after)


replace("active=json.loads((root/'active-training.json').read_text())", """active=json.loads((root/'active-training.json').read_text())
plan=json.loads((root/'receipts/two-task-eight-gpu-sequential-20261017-v2/six-combination-manifest.json').read_bytes())
versions={j['version'] for j in plan['combinations']}
active['jobs']=[j for j in active['jobs'] if j.get('version') in versions and j.get('active')]
assert len(active['jobs'])==1 and active['jobs'][0]['method']=='dt'
binding=Path(active['jobs'][0]['run'])/'runtime-overrides/actual-worker-imports-after-first-DT-v1.json'
actual=json.loads(binding.read_bytes())
targeted={r['pid']:r['birth'] for r in actual['results'] if r['rank'] in (0,6)}
assert len(targeted)==2""")
replace("out=root/'receipts/owner-b8-dispatch-20260930'/f'phase-cost-profile-{int(time.time())}'",
        "out=root/'receipts/six-run-native-utilization-20261017-v1'/f'bounded-DT-phase-{int(time.time())}'")
replace("if 'WorkerDict' not in child.name():continue", """if 'WorkerDict' not in child.name() or child.pid not in targeted:continue
            assert child.create_time()==targeted[child.pid]""")
replace("argv=['mx-smi','--show-usage','--show-pcie-bandwidth','--show-hbm-bandwidth','-j']",
        "argv=['mx-smi','--show-usage','--show-pcie-bandwidth','--show-hbm-bandwidth']")
# JSON mode on this installed mx-smi ignores the bandwidth selection. Query
# each original supported command separately, keeping raw output intact.
replace("r=subprocess.run(argv,capture_output=True,text=True,timeout=10)\n    physical.append(dict(unix=time.time(),argv=argv,returncode=r.returncode,\n                         stdout=r.stdout,stderr=r.stderr))",
        """for flag in argv[1:]:
        native=['mx-smi',flag]
        r=subprocess.run(native,capture_output=True,text=True,timeout=10)
        physical.append(dict(unix=time.time(),argv=native,returncode=r.returncode,
                             stdout=r.stdout,stderr=r.stderr))""")
replace("manifest=active['manifest']", "manifest=str(root/'active-training.json')")
if '--native' in sys.argv:
    raise RuntimeError('Installed py-spy cannot combine native unwinding with nonblocking sampling; retain the nonblocking owner path')
sys.path.insert(0, str(source.parent))
exec(compile(code, str(source), 'exec'), globals())
