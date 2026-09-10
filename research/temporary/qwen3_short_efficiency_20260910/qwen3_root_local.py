"""Use the already validated code-local observer for root FA capture."""
import sys
from qwen3_root_tape import NativeRootTape
from native_capture_events import LocalCaptureEvents


class LocalNativeRootTape(NativeRootTape):
    def __enter__(self):
        super().__enter__();sys.setprofile(None)
        self.local_events=LocalCaptureEvents([self.public_code],self.observe_fa,returns_only=True)
        try:self.local_events.__enter__()
        except BaseException:
            super().__exit__(*sys.exc_info());raise
        return self

    def __exit__(self,*args):
        try:self.local_events.__exit__(*args)
        finally:super().__exit__(*args)
