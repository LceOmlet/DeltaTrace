"""Change only DT's passive event delivery, retaining the original callbacks."""
import sys
from local_capture_events import LocalCaptureEvents


class CaptureEfficiency:
    def __init__(self,family,mode):
        assert mode in ('retained','filtered','local')
        self.family=family;self.mode=mode;self.saved=None

    def __enter__(self):
        if self.mode=='retained':return self
        if self.family=='qwen3':
            assert self.mode=='local'
            import qwen3_retained_pair as module
            from qwen3_code_local_replay import NativeLayerReplay
            self.saved=(module,module.NativeLayerReplay)
            module.NativeLayerReplay=NativeLayerReplay
            return self
        import qwen35_retained_controller as module
        GDN,Attention=module.NativeGDNCapture,module.NativeDenseAttentionCapture
        self.saved=(module,GDN,Attention)
        class FilteredGDN(GDN):
            def event(self,frame,kind,value):
                # The original callback otherwise materializes f_locals even
                # for every unrelated Python/C event in native FLA/Triton.
                if kind not in ('call','return') or frame.f_code not in self.codes:return
                return super().event(frame,kind,value)
        class LocalGDN(GDN):
            def __enter__(self):
                self.local_events=LocalCaptureEvents(self.codes,self.event)
                self.local_events.__enter__();return self
            def __exit__(self,*args):
                self.local_events.__exit__(*args)
                return super().__exit__(*args)
        class LocalAttention(Attention):
            def __enter__(self):
                # Reuse unchanged installation of the original module hooks.
                super().__enter__();sys.setprofile(None)
                self.local_events=LocalCaptureEvents(
                    [self.native_dense.__code__,self.native_varlen.__code__,self.interface.__code__],self.profile)
                try:self.local_events.__enter__()
                except BaseException:
                    super().__exit__(*sys.exc_info());raise
                return self
            def __exit__(self,*args):
                self.local_events.__exit__(*args)
                return super().__exit__(*args)
        module.NativeGDNCapture=FilteredGDN if self.mode=='filtered' else LocalGDN
        if self.mode=='local':module.NativeDenseAttentionCapture=LocalAttention
        return self

    def __exit__(self,*args):
        if self.saved is None:return
        if self.family=='qwen3':self.saved[0].NativeLayerReplay=self.saved[1]
        else:self.saved[0].NativeGDNCapture,self.saved[0].NativeDenseAttentionCapture=self.saved[1:]
        self.saved=None
