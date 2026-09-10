"""Deliver only selected native Python events; preserve every capture callback."""
import sys
from native_capture_events import LocalCaptureEvents
from qwen35_retained_capture import NativeDecoderCapture
from qwen35_retained_capture import NativeGDNCapture as _GDN
from qwen35_retained_capture import NativeDenseAttentionCapture as _Attention


class NativeGDNCapture(_GDN):
    def __enter__(self):
        self.local_events=LocalCaptureEvents(self.codes,self.event)
        self.local_events.__enter__()
        return self

    def __exit__(self,*args):
        try:self.local_events.__exit__(*args)
        finally:super().__exit__(*args)


class NativeDenseAttentionCapture(_Attention):
    def __enter__(self):
        # Install the existing passive module hooks through the retained helper.
        super().__enter__()
        sys.setprofile(None)
        self.local_events=LocalCaptureEvents(
            [self.native_dense.__code__,self.native_varlen.__code__,self.interface.__code__],self.profile)
        try:self.local_events.__enter__()
        except BaseException:
            super().__exit__(*sys.exc_info())
            raise
        return self

    def __exit__(self,*args):
        try:self.local_events.__exit__(*args)
        finally:super().__exit__(*args)
