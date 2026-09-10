"""One common, fully charged original-model loading path for DT and both FT."""
import time
import torch


def load_common_staged_model(author_benchmark,checkpoint,device='cuda:0'):
    start=time.perf_counter()
    model,tokenizer=author_benchmark.load_model_balanced(checkpoint,'cpu')
    cpu_seconds=time.perf_counter()-start
    transfer_start=time.perf_counter();model.to(device);torch.cuda.synchronize()
    transfer_seconds=time.perf_counter()-transfer_start
    assert all(str(p.device)==device for p in model.parameters())
    return model,tokenizer,{'strategy':'Original author CPU loading followed by unmodified model.to(device). Same path for DT and both FT.',
        'cpu_stage_seconds':cpu_seconds,'native_to_device_seconds':transfer_seconds,'loading_seconds':time.perf_counter()-start,
        'full_parameter_and_buffer_identity_evidence':'cpu_staged_loading_verification.json',
        'host_memory_tradeoff':'CPU model staging uses host RAM; record complete process RSS/highwater with load and every call.',
        'model_forward_modified':False,'FT_implementation_modified':False}
