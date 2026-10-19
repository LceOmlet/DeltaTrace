"""Restore the provisioned author service through its accepted start block."""
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import time
import urllib.request
import psutil

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
RECEIPT = ROOT/'receipts/textcraft-v5-eight-gpu-20261019-v1'

def save(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')

def main():
    RECEIPT.mkdir(parents=True, exist_ok=True)
    with socket.socket() as client:
        client.settimeout(2)
        listening = client.connect_ex(('127.0.0.1', 36005)) == 0
    if listening:
        print(json.dumps(dict(status='existing_TextCraft_service_reused',port=36005)))
        return
    owner = ROOT/'receipts/textcraft-restart-utilization-20261012-v1/source/restart.py'
    if not owner.is_file():
        owner = RECEIPT/'source/accepted_restart_20261012.py'
    source = owner.read_text()
    start = source.index('    # Read the previously provisioned environment through its existing source.')
    end = source.index('    jobs = []', start)
    block = '\n'.join(line[4:] if line.startswith('    ') else line for line in source[start:end].splitlines())
    exec(compile(block, str(owner), 'exec'), globals())
    print(json.dumps(dict(status='original_TextCraft_service_ready',
        receipt=str(RECEIPT/'official-service.json'),owner=str(owner),
        owner_SHA256=hashlib.sha256(owner.read_bytes()).hexdigest())))

if __name__ == '__main__':
    main()
