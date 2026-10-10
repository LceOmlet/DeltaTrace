"""Run the previously failed native replay unchanged; retain exception tensors.

The original replay source remains e48ed002. This wrapper performs no tensor
observation or GPU operation before that method raises. On the demonstrated
post-failure observer exception, its traceback still owns h and ids; save those
without reading the released FSDP vocabulary weight, then re-raise the original
exception. This is a diagnostic, not a production patch or a formal restart.
"""
import functools
import hashlib
import json
from pathlib import Path
import time

import replay_native_incident37_old_ref_dt_20261011 as original


ORIGINAL_SHA = 'e48ed0021391b2d6c1473714d9e90ab6d44bfed6026697572dc8f0596e40ccb9'


def main():
    assert hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest() == ORIGINAL_SHA
    native_factory = original.make_worker

    def observed_factory():
        worker_class = native_factory()
        native_method = worker_class.replay_incident

        @functools.wraps(native_method)
        def observed(self, output):
            try:
                return native_method(self, output)
            except BaseException as error:
                # All access below happens after the original native method
                # has failed and unwound its own exception/finally handlers.
                import torch
                report = dict(unix=time.time(), rank=self.rank,
                              original_source=original.__file__,
                              original_sha256=ORIGINAL_SHA,
                              wrapper_source=__file__,
                              wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                              error=repr(error), captures=[],
                              pre_exception_tensor_observations=0,
                              released_vocabulary_weight_read=False)
                tb = error.__traceback__
                while tb is not None:
                    frame = tb.tb_frame
                    values = frame.f_locals
                    if (frame.f_code.co_name == 'gradient_hook'
                            and frame.f_code.co_filename == original.__file__
                            and values.get('finite') is False
                            and all(name in values for name in ['h', 'ids', 'node'])):
                        try:
                            node = values['node']
                            path = Path(output) / f'rank{self.rank}-exception-head-H-IDs.pt'
                            torch.save(dict(
                                hidden_states=values['h'].detach().to('cpu', copy=True),
                                input_ids=values['ids'].detach().to('cpu', copy=True),
                                temperature=node.temperature,
                                chunk_size=node.chunk_size,
                                orig_batch_size=node.orig_batch_size,
                                orig_ndim=node.orig_ndim,
                                original_source=original.__file__,
                                original_sha256=ORIGINAL_SHA,
                                source_frame_line=tb.tb_lineno,
                                microbatch_index=values.get('index'),
                                vocabulary_weight_not_read=True), path)
                            report['captures'].append(dict(
                                path=str(path), bytes=path.stat().st_size,
                                sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
                        except Exception as capture_error:
                            report['captures'].append(dict(capture_error=repr(capture_error)))
                    tb = tb.tb_next
                (Path(output) / f'rank{self.rank}-exception-capture.json').write_text(
                    json.dumps(report, indent=2) + '\n')
                raise

        worker_class.replay_incident = observed
        return worker_class

    original.make_worker = observed_factory
    try:
        original.main()
    finally:
        original.make_worker = native_factory


if __name__ == '__main__':
    main()
