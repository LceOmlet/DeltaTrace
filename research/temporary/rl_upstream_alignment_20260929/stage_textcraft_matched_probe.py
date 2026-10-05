"""Reuse the existing bounded original-owner loader; freeze separate v2 receipts."""
import hashlib
import subprocess
import tarfile

from stage_environment_entry import AUDIT, ROOT, SSH, SCP
from stage_textcraft_degradation_probe import SCRIPT

OUT=ROOT+'/receipts/textcraft-degradation-probe-20261005-v5'
NAMES=['verify_textcraft_credit_degradation.py','observe_native_actor_loss_gradients.py',
       'observe_textcraft_matched_attribution.py','observe_textcraft_layer3_finite_stages.py']

if __name__ == '__main__':
    hashes={name:hashlib.sha256((AUDIT/name).read_bytes()).hexdigest() for name in NAMES}
    archive=AUDIT/'textcraft-degradation-20261005/diagnostic-matched-source-v5.tar'
    with tarfile.open(archive,'w') as tar:
        for name in NAMES:tar.add(AUDIT/name,arcname=name)
    subprocess.run(SSH+['mkdir','-p',OUT],check=True)
    subprocess.run(SCP+[str(archive),f'{SSH[-1]}:{OUT}/source.tar'],check=True)
    script=SCRIPT.replace('__ROOT__',ROOT).replace('__OUT__',OUT).replace('__HASHES__',repr(hashes))
    script=script.replace("DT_TEXTCRAFT_GRADIENT_DIAGNOSTIC='1'", "DT_TEXTCRAFT_GRADIENT_DIAGNOSTIC='0', DT_TEXTCRAFT_MATCHED_DIAGNOSTIC='1', DT_TEXTCRAFT_DECODER3_LEDGER='1'")
    result=subprocess.run(SSH+['bash','-s'],input=script.encode(),capture_output=True,check=True)
    (AUDIT/'textcraft-degradation-20261005/probe-matched-submission-v5.json').write_bytes(result.stdout)
    print(result.stdout.decode(errors='replace'))
