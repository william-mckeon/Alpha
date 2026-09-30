"""Fix pinned LightEval padding consuming the entire generation budget."""
import importlib.util
from pathlib import Path
root=Path(importlib.util.find_spec('lighteval').origin).parent
path=root/'models/base_model.py';text=path.read_text()
start=text.index('    def greedy_until(');end=text.index('    def _generate(',start)
section=text[start:end]
old='padding="max_length"'
if section.count(old)!=1:raise RuntimeError('Unexpected pinned generation implementation')
section=section.replace(old,'padding="longest"')
old='max_length=max_context_continuation_size_allowed,'
if section.count(old)!=1:raise RuntimeError('Unexpected generation truncation implementation')
section=section.replace(old,'max_length=min(max_context_continuation_size_allowed, self.max_length - 1),')
path.write_text(text[:start]+section+text[end:])

# The installed xxhash requires bytes; the pinned logger passes text.
# Serialize structured chat values too, without changing prompts or metrics.
path=root/'logging/info_loggers.py';text=path.read_text()
if text.count('xxhash.xxh64(')!=9:raise RuntimeError('Unexpected pinned detail hashing implementation')
text=text.replace('xxhash.xxh64(', '_audit_hash(')
text=text.replace('import xxhash', 'import xxhash\n\ndef _audit_hash(value):\n    return xxhash.xxh64(value if isinstance(value, bytes) else str(value).encode("utf-8"))\n')
path.write_text(text)
