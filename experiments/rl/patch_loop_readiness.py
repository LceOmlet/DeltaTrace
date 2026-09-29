"""Route a reproduced non-AppWorld HTTP response into LOOP's existing retry.

The MetaX namespace can reach a 404 endpoint on a port absent from its psutil
socket inventory. Do not parse that response as a ready AppWorld service.
Original port selection, retry counts, timing, restart and errors are retained.
"""
from pathlib import Path
import sys


def patch(source):
    anchor='                    if response is not None:\n                        # sanity check on response\n'
    replacement=('                    if response is not None and response.status_code != 200:\n'
                 '                        response = None\n'+anchor)
    if replacement in source:return source
    if source.count(anchor)!=1:raise ValueError('Pinned LOOP readiness boundary changed')
    return source.replace(anchor,replacement,1)


if __name__=='__main__':
    path=Path(sys.argv[1])/'phi_agents/appworld/interface.py'
    source=patch(path.read_text())
    compile(source,str(path),'exec')
    path.write_text(source,newline='\n')
