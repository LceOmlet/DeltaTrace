"""Read saved native logs; no tensors, model calls, or acceptance tolerance."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "native-MCCL-collective-log-terminal-artifacts-20261011-v1"
log = ARTIFACTS / "mccl.%h.%p.log"
manifest = json.loads((ARTIFACTS / "manifest.json").read_text(encoding="utf-8"))
text = log.read_text(encoding="utf-8", errors="strict")
prefix = re.compile(r"(?P<host>[^\s:]+):(?P<pid>\d+):(?P<tid>\d+) \[(?P<device>\d+)\] MCCL (?P<level>\w+) ")
message = re.compile(
    r"(?P<op>\w+): rank (?P<rank>\d+) opCount (?P<opcount>[0-9a-f]+) "
    r"sendbuff (?P<send>0x[0-9a-f]+) recvbuff (?P<recv>0x[0-9a-f]+) "
    r"count (?P<count>\d+) datatype (?P<datatype>\d+) op (?P<reduceop>\d+) "
    r"root (?P<root>\d+) comm (?P<comm>0x[0-9a-f]+) \[nranks=(?P<nranks>\d+)\] "
    r"stream (?P<stream>0x[0-9a-f]+|\(nil\))\s*"
)
starts = list(prefix.finditer(text))
records = []
discarded = Counter()
for i, match in enumerate(starts):
    stop = starts[i + 1].start() if i + 1 < len(starts) else len(text)
    body = text[match.end():stop]
    parsed = message.fullmatch(body)
    if parsed is None:
        discarded[match.group("pid")] += 1
        continue
    row = dict(match.groupdict(), **parsed.groupdict())
    row["log_character_offset"] = match.start()
    for key in ("pid", "tid", "device", "rank", "count", "datatype", "reduceop", "root", "nranks"):
        row[key] = int(row[key])
    for key in ("send", "recv", "comm", "opcount"):
        row[key] = int(row[key], 16)
    row["stream"] = 0 if row["stream"] == "(nil)" else int(row["stream"], 16)
    records.append(row)

rank_summaries = []
for gemm in manifest["GEMM_rows"]:
    rank = gemm["rank"]
    process = next(r for r in manifest["ranks"] if r["rank"] == rank)
    head_pointers = sorted({r["right_pointer"] for r in gemm["metadata"]["mm_calls"]})
    rows = [r for r in records if r["rank"] == rank and r["pid"] == process["pid"]]
    # Compare exact address identities only. The native datatype enumeration
    # has not been mapped to PyTorch dtypes; do not infer byte spans from it.
    layer0 = [r for r in rows if r["op"] == "AllGather" and r["count"] == 219095744]
    exact_fingerprint_address_matches = [
        dict(record=r, head_pointer=p)
        for r in layer0 for p in head_pointers
        if r["recv"] + 67698688 == p + 1302921216
    ]
    rank_summaries.append(dict(
        rank=rank, pid=process["pid"], complete_records=len(rows),
        operations=dict(Counter(r["op"] for r in rows)),
        head_pointers=head_pointers,
        candidate_layer0_raw_count_records=len(layer0),
        candidate_layer0_recv_pointers=sorted({r["recv"] for r in layer0}),
        exact_fingerprint_address_matches=exact_fingerprint_address_matches,
        captured_weight_hashes=gemm["weight_row_hash_counts"],
    ))

report = dict(
    native_log=dict(path=str(log.relative_to(Path.cwd())), bytes=log.stat().st_size,
                    sha256=hashlib.sha256(log.read_bytes()).hexdigest()),
    prefix_segments=len(starts), complete_records=len(records),
    incomplete_segments_by_pid=dict(discarded), ranks=rank_summaries,
    limitations=[
        "Installed MetaX library wrote the literal mccl.%h.%p.log filename; rank processes shared and overwrote/interleaved the file.",
        "Only complete same-prefix messages were parsed; no reconstruction of missing or interleaved fields.",
        "Record order is file order, not a GPU execution timeline.",
        "Datatype enumeration is unmapped; no byte-range inference from native count.",
        "The replay was finite and captured normal head rows; this does not locate or exclude the failed run's faulty writer.",
    ],
    production_changes=0, model_calls=0, cuda_initialized=False,
)
output = ROOT / "native-MCCL-complete-record-analysis-20261011-v1.json"
output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"output": str(output), "prefix_segments": len(starts),
                  "complete_records": len(records), "ranks": rank_summaries}, ensure_ascii=False))
