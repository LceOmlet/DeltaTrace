"""Compose the existing diagnostic stop-only guard for the failed label import."""
import subprocess
from stage_environment_entry import SSH
from stop_native_minibatch_wrong_scope import script as ORIGINAL_SCRIPT

SCRIPT = (ORIGINAL_SCRIPT
    .replace('textcraft-native-minibatch-20261006-v2', 'textcraft-label-encoding-20261006-v1')
    .replace('4050498', '3000361')
    .replace('verify_textcraft_native_minibatch.py', 'verify_textcraft_label_encoding.py')
    .replace('DiagnosticTrainer alias did not bind to original Ray-serialized TaskRunner globals; original 256 rollout began instead of requested64. Stop before update and correct the diagnostic seam only.',
             'Fresh Ray worker cannot import the unchanged native-reader helper; stop before model initialization and fix only diagnostic PYTHONPATH. Preserve failed candidate and raw log.')
    .replace('stopped-wrong-diagnostic-scope.json', 'stopped-diagnostic-import.json'))

if __name__ == '__main__':
    result = subprocess.run(SSH + ['bash', '-s'], input=SCRIPT.encode(), capture_output=True)
    print(result.stdout.decode(errors='replace'))
    print(result.stderr.decode(errors='replace'))
    result.check_returncode()
