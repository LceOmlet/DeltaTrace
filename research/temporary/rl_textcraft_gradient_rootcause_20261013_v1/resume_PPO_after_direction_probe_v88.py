"""Use original VERL full-state restore after the full-loss direction probe."""
from pathlib import Path

source = Path(__file__).with_name('resume_PPO_after_partition_probe_v82.py')
code = source.read_text().replace('native-upper-adapter-partition-v81',
                                'native-full-policy-direction-exchange-v86')
code = code.replace('partition-v81', 'direction-v86').replace('v82', 'v88')
exec(compile(code, str(source), 'exec'), globals())
