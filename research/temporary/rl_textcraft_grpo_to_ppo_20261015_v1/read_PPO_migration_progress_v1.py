"""Read the existing native PPO restore progress after the device migration."""
from pathlib import Path

source = Path(__file__).resolve().parent.parent/'rl_textcraft_gradient_rootcause_20261013_v1/read_PPO_restore_progress_v94.py'
code = source.read_text()
code = code.replace('native-PPO-resume-after-saved-update-v90-v91-job.json', 'native-PPO-resume-GPU45-v117-job.json')
code = code.replace('v91', 'v117')
exec(compile(code, str(source), 'exec'), globals())
