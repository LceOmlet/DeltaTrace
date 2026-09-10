"""Omit numerical storage for four native operands used only as metadata.

The original finite program never reads a/attn_out/mlp_in/mlp_out. Actual
native mutation/version checks are still performed before dropping them;
their complete original layout descriptors are restored in the CPU result.
"""
import time
import torch
from qwen3_streamed_graph import StreamTape
from qwen3_streamed_build_graph import BuildingStreamTape,StreamedBuildFiniteGraphQwen3

OMITTED=('a','attn_out','mlp_in','mlp_out')


def layouts(values):
    answer={}
    for name in OMITTED:
        value=values[name];shape=list(value.shape);stride=list(value.stride())
        assert shape[0]==2
        answer[name]={'shape':shape,'stride':stride,'dtype':str(value.dtype),
            'endpoint_strides':[list(stride),list(stride)],
            'endpoint_storage_offsets':[value.storage_offset(),value.storage_offset()+stride[0]]}
    return answer


class CompactStreamTape(StreamTape):
    def finish_layer(self,index):
        values=self.layers[index]
        assert layouts(values)==self.program.omitted_layouts[index]
        assert tuple(k for k,v in values.items() if v is not None)==self.program.layout_order[index]
        # Parent still checks all twenty native operands, including the four
        # omitted values. Only its graph-input copy plan is smaller.
        return super().finish_layer(index)


class CompactBuildingStreamTape(BuildingStreamTape):
    def __init__(self,model,*,mutation_audit=False):
        super().__init__(model,mutation_audit=mutation_audit)
        self.omitted_layouts=[None]*len(self.layers)
        self.layout_order=[None]*len(self.layers)

    def finish_layer(self,index):
        values=self.layers[index]
        self.omitted_layouts[index]=layouts(values)
        self.layout_order[index]=tuple(k for k,v in values.items() if v is not None)
        for name in OMITTED:
            value=values.pop(name)
            assert value._version==self.versions.pop((index,name)),(index,name)
            if self.mutation_audit:
                assert torch.equal(value,self.snapshots.pop((index,name))),(index,name)
                self.mutation_checks+=1
        return super().finish_layer(index)


class StreamedCompactFiniteGraphQwen3(StreamedBuildFiniteGraphQwen3):
    hot_tape_type=CompactStreamTape
    cold_tape_type=CompactBuildingStreamTape

    def initialize_storage_metadata(self,tape):
        self.program.omitted_layouts=tape.omitted_layouts
        self.program.layout_order=tape.layout_order

    def attribute(self,*args,**kwargs):
        result=super().attribute(*args,**kwargs)
        started=time.perf_counter()
        for row in result['native_paired_root_layouts']:
            index=row['layer'];kept=row['tensors'];omitted=self.program.omitted_layouts[index]
            assert not set(kept).intersection(OMITTED)
            row['tensors']={name:omitted[name] if name in omitted else kept[name]
                            for name in (*self.program.layout_order[index],*(k for k in kept if k not in self.program.layout_order[index]))}
        result['streamed_root']['metadata_only_native_operands']=list(OMITTED)
        result['streamed_root']['metadata_layout_restore_seconds']=time.perf_counter()-started
        result['streamed_root']['all_original_native_operand_checks_retained']=True
        result['streamed_root']['layout_restore_time_scope']='Additional CPU work after the base resolver; included in the external synchronized complete-call timer.'
        return result
