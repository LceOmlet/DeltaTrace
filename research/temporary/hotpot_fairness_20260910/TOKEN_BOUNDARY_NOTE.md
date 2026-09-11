The source-only token-map validation found three indivisible BPE tokens crossing
native sentence boundaries: `?"` in case 12, `!"` in case 31, and `.k` in case 44.
The original HotpotQA arrays themselves split at those positions. This was found
before scoring corrected views. Keep the protocol's first non-whitespace anchor:
assign the whole token to that native unit and charge it exactly once. Do not
resplit native sentences, discard a token, duplicate it, or divide its cost.
The earlier defensive rejection was relaxed to implement this total mapping.
These three crossings are recorded in source_audit.json. This is a tokenization
approximation at native boundaries; whole-sentence retrieval means all eligible
tokens assigned to the native sentence, not substring-perfect tokenization.
