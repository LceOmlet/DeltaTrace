"""Observe selected native code objects using CPython's public monitoring API.

This controls observation only. No native callable, bytecode, tensor or result
is replaced. Events are scoped to reviewed code objects and the owning thread.
"""
import sys
import threading


class LocalCaptureEvents:
    def __init__(self,codes,callback,*,returns_only=False):
        self.codes=tuple(codes);self.callback=callback;self.returns_only=returns_only
        self.tool=None;self.thread=None;self.registered=[];self.enabled=[]

    def __enter__(self):
        if sys.getprofile() is not None:
            raise RuntimeError('Refuse to interfere with an existing Python profiler.')
        if not hasattr(sys,'monitoring'):
            raise RuntimeError('Code-local capture requires CPython 3.12 or newer.')
        monitor=sys.monitoring
        # Keep debugger, coverage, profiler and optimizer reserved IDs available.
        self.tool=next((i for i in (3,4) if monitor.get_tool(i) is None),None)
        if self.tool is None:raise RuntimeError('No free monitoring ID for native capture.')
        monitor.use_tool_id(self.tool,'DeltaTrace native capture')
        self.thread=threading.get_ident()
        def on_start(code,offset):
            if threading.get_ident()!=self.thread:return
            frame=sys._getframe(1)
            assert frame.f_code is code
            self.callback(frame,'call',None)
        def on_return(code,offset,value):
            if threading.get_ident()!=self.thread:return
            frame=sys._getframe(1)
            assert frame.f_code is code
            self.callback(frame,'return',value)
        try:
            callbacks=[(monitor.events.PY_RETURN,on_return)]
            if not self.returns_only:callbacks.append((monitor.events.PY_START,on_start))
            flags=0
            for event,callback in callbacks:
                previous=monitor.register_callback(self.tool,event,callback)
                assert previous is None
                self.registered.append(event);flags|=event
            for code in self.codes:
                assert monitor.get_local_events(self.tool,code)==0
                monitor.set_local_events(self.tool,code,flags);self.enabled.append(code)
            return self
        except BaseException:
            self.__exit__(None,None,None)
            raise

    def __exit__(self,*args):
        if self.tool is None:return
        monitor=sys.monitoring;tool=self.tool
        try:
            for code in self.enabled:monitor.set_local_events(tool,code,0)
            for event in self.registered:monitor.register_callback(tool,event,None)
            monitor.set_events(tool,0)
        finally:
            monitor.free_tool_id(tool);self.tool=None;self.enabled.clear();self.registered.clear()


def check_runtime():
    """Small API/cleanup check; no model or GPU work."""
    records=[]
    def observed(x):
        intermediate=x+3
        return intermediate*2
    def capture(frame,event,value):
        records.append((event,frame.f_locals['x'],value))
    before=[sys.monitoring.get_tool(i) for i in range(6)]
    with LocalCaptureEvents([observed.__code__],capture):
        assert observed(5)==16
    assert records==[('call',5,None),('return',5,16)]
    assert [sys.monitoring.get_tool(i) for i in range(6)]==before
    try:
        with LocalCaptureEvents([observed.__code__],capture):raise ValueError('cleanup probe')
    except ValueError:pass
    assert [sys.monitoring.get_tool(i) for i in range(6)]==before
    return {'callback_frames_and_values':True,'tool_ids_restored':True,'exception_cleanup':True}
