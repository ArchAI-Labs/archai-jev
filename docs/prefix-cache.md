# What is reused between questions (the prefix cache)

A prompt for the default model has a part that does not depend on your text (the system block and
the `State:` header) and a part that you send once and ask about many times (the state).
`archai-jev` computes each of these parts
once and keeps the result in memory, so asking ten questions about one source, or many short
messages of the same task, costs much less than ten independent calls.

## What is reused

- **Inside one call**: the questions of an `ask` share the fixed block and, when the template puts
  the state first, the state. They are computed once for the whole call.
- **Between calls** (in memory, in the same process, per loaded model): the fixed block, and a
  state that is asked about again (it is kept the second time it is seen, or at once
  when two questions of one call share it).

Nothing is written to disk and nothing is shared between models: a saved prefix is only used by the
same model file, the same template and the same engine settings.

## Do the answers change?

Reusing does not change an answer *because of the cache*: with the memory empty, warm or cleared the
numbers are identical bit for bit. What differs from 0.1.0 is that every prompt is computed in
segments (at the end of the fixed block and of the state) instead of in one pass; the numbers move
by the usual rounding of the order of calculation, within the tolerances in the spec: for the default
model the largest difference measured between the two ways of computing was 0.043 on a probability
(23 requests, Q8_0). Set
`prefix_cache_mib=0` to get the computation of 0.1.0 back.

## Control

```python
Jev.from_pretrained(prefix_cache_mib=128)   # default; or ARCHAI_JEV_PREFIX_CACHE_MIB
Jev.from_pretrained(prefix_cache_mib=0)     # off: one pass per question, as in 0.1.0
jev.cache_info()                            # limit, bytes, entries, hits, misses, evictions, tokens
jev.clear_cache()                           # forget everything; never changes an answer
```

The argument wins over the variable, the variable over the default (128 MiB). The limit is in
bytes of saved state: about 29 KB per token for the default model, about 20 MB plus 12 KB per
token for Kev (so 128 MiB holds only a few Kev prefixes: raise it if you reuse Kev a lot). When the
limit is reached the least recently used prefix goes first.

## When it helps

- Many questions about one text in one call or in a stream of calls (grounding checks, duplicate
  finders): the largest gain.
- Many short messages with the same question: the fixed block is skipped each time.
- A single question about a new text: nothing to reuse, and a little more work than before (the
  prefix is saved for next time).

Speed depends on the machine. After the CPU has been idle it runs slower for the first requests, so
a call after a pause is slower than the same call in a series; measure both ways before comparing.

## Privacy

The memory holds states derived from the text you sent. They stay in the process and are never
written out, but in a server shared between users, response time can reveal whether a text was
already seen. In that case turn the memory off (`prefix_cache_mib=0`) or call `clear_cache()`.
