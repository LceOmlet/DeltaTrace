"""Repair the reproduced HTTP status and final-poll LOOP readiness boundaries.

The MetaX namespace can reach a 404 endpoint on a port absent from its psutil
socket inventory. Do not parse that response as a ready AppWorld service.
The original final wait precedes its final GET, so readiness during that wait
is checked within the same five checks/five waits. Port selection, total wait
budget, retry counts, restarts, task-ID check and errors are retained.
"""
from pathlib import Path
import sys


def patch(source):
    anchor='                    if response is not None:\n                        # sanity check on response\n'
    replacement=('                    if response is not None and response.status_code != 200:\n'
                 '                        response = None\n'+anchor)
    if replacement not in source:
        if source.count(anchor)!=1:raise ValueError('Pinned LOOP readiness boundary changed')
        source=source.replace(anchor,replacement,1)
    before=('                for attempt_wait in range(self._max_wait_tries):\n'
            '                    try:\n')
    before_new=('                for attempt_wait in range(self._max_wait_tries):\n'
                '                    if attempt_wait == self._max_wait_tries - 1:\n'
                '                        time.sleep(self._wait_seconds)\n'
                '                    try:\n')
    after=('                    time.sleep(self._wait_seconds)\n\n'
           '                if attempt_restart < self._max_restarts_on_error:\n')
    after_new=('                    if attempt_wait < self._max_wait_tries - 1:\n'
               '                        time.sleep(self._wait_seconds)\n\n'
               '                if attempt_restart < self._max_restarts_on_error:\n')
    if before_new in source and after_new in source:return source
    if source.count(before)!=1 or source.count(after)!=1:
        raise ValueError('Pinned LOOP final-wait boundary changed')
    return source.replace(before,before_new,1).replace(after,after_new,1)


if __name__=='__main__':
    path=Path(sys.argv[1])/'phi_agents/appworld/interface.py'
    source=patch(path.read_text())
    compile(source,str(path),'exec')
    path.write_text(source,newline='\n')
