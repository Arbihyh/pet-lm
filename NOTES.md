# Engineering notes

Measurements and derivations behind the design choices. Kept separate from the
README so the two kinds of number stay distinguishable: everything in section 1
was measured by the upstream project on real hardware, everything in section 2 is
derived from those measurements and has **not** been observed on a device.

---

## 1. Measured upstream (ESP32-S3 N16R8)

From `manjunathshiva/esp32-tinyllm` RESULTS.md, its 28.9M-parameter TinyStories
model. Reported here because every projection below is calibrated against it.

| stage | ms/token | note |
|---|---:|---|
| output head | 57.6 | identical in every run |
| attention | 15.7–34.1 | scales with context position |
| PLE path | 8.5 | constant |
| FFN | 6.9 | constant |
| input | 4.4 | constant |
| **total** | **103** | 9.5 tok/s |

| bandwidth | value |
|---|---|
| PSRAM sequential read | 60.7 MB/s |
| internal SRAM sequential read | 240 MB/s |
| flash random read, 512 B row | 20.3 us |

The head is constant while attention tracks context length, which is what makes
the profile credible: the head scans a fixed 2.54 MB of int8 weights per token
regardless of position, attention is O(pos).

### Effective throughput per path

Dividing bytes moved by measured time gives each path's real rate, which is more
useful than the raw bandwidth ceiling:

| path | effective | against ceiling |
|---|---:|---|
| output head | **56.9 MB/s** | 94% of PSRAM's 60.7 — bandwidth-bound, already optimal |
| core | **18.1 MB/s** | 7.5% of SRAM's 240 — compute-bound |
| attention | 8.9 M units/s | — |

**The core is 3.1x slower per byte than the head.** Cause is visible in
`llm.h`: the head is staged from int4 to int8 in PSRAM once at boot, so per token
it does only int8 x int8 -> int32. The core re-does four things for every weight
on every token — nibble extract, -8 offset and int->float convert, `half2float`
on the group scale, then float multiply-add.

This inverts the obvious optimisation target. Upstream's stated next step is an
int4 head with SIMD unpack, but the head is already at 94% of the bus; the
headroom is in the core.

---

## 2. Projected for this model — NOT MEASURED

Calibrated against section 1's effective rates. `baseline` arm, vocab 312,
d_model 192, 6 layers, ffn 384, seq_len 24.

| | value |
|---|---|
| core parameters | 2,214,336 |
| output head | 312 x 192 -> 0.06 MiB int8 |
| flash table | none (baseline arm) |
| KV cache | 2 x 6 x 24 x 192 x 4 B = 216 KiB |

Per-token time at the *current* core rate (18.1 MB/s) versus with the core given
the same int8 staging the head has (56.9 MB/s):

| | head | core | attn | total | tok/s |
|---|---:|---:|---:|---:|---:|
| as-is | 1.1 ms | 61.1 ms | 6.2 ms | ~68 ms | ~15 |
| core staged int8 | 1.1 ms | 19.4 ms | 6.2 ms | ~27 ms | **~37** |

A plan is 3–4 tokens, so response latency lands near 0.1 s once the core is
staged, against ~0.27 s as-is. The core fix is the whole difference: with vocab
312 the head costs 1.1 ms and nothing is bandwidth-bound any more, so the entire
bottleneck is the core's per-weight dequantisation arithmetic.

Footprint: ~1.11 MiB flash (core int4 1.05 + head 0.06) and ~0.29 MiB PSRAM
(head 0.06 + KV 0.21 + scratch), against the upstream model's 13.8 MB and
4.4 MB. **An N8R2 part suffices; N16R8 is not needed.**

### Why the vocabulary cut is the whole story

Output head size is vocabulary x d_model, and the head is scanned in full every
token. Upstream needs vocab 32768 to write English prose. Twelve fixed actions
do not:

| vocab | head (int8) | head read at 56.9 MB/s |
|---:|---:|---:|
| 32768 | 3.12 MiB | 57.6 ms |
| 4096 | 0.38 MiB | 6.6 ms |
| **312** | **0.06 MiB** | **1.1 ms** |

The same cut removes the PLE table's reason to exist: its advantage scales with
vocabulary, and at 312 there is no large sparse table left to be cheap.

---

## 3. Why not Needle

`cactus-compute/needle` markets itself for microcontrollers and would cover this
use case directly. It was evaluated and rejected. Recorded here so the decision
is not revisited from the marketing copy.

Measured from the shipped `needle3.cact` header (first 4 KB, parsed against the
byte-layout spec in the repo's `export.py`):

| field | documented | **actual** |
|---|---|---|
| file size | 29 MB | **35,335,380 B (33.7 MiB)** |
| vocab | 16384 (source default) | **8192** |
| bits/weight | 2.125 | **2.45** |
| max_seq_len | 4096 (source default) | **8192** |
| kv_bits | — | 8 |
| kv_window | — | 256 |

Verified: 20 layers, d_model 768, 5 engram sites at layers {3,7,11,15,19},
18432 slots x 6 tables x 128 dims, 581 tensors. The CQ unpack formula from the
spec reproduces six tensors' `nbytes` exactly, so the weight format is fully
decodable.

Blocking issues, in order:

1. **No bare-metal target.** All 13 platform folders require an OS; the smallest
   is `linux-mipsel`. `PLATFORMS` in `needle/agent/fetch.py` contains no Xtensa,
   Cortex-M, Zephyr or FreeRTOS entry.
2. **The engine is closed.** The repository contains zero C/C++/asm/Rust source
   files; inference is `ctypes.CDLL` into a prebuilt `libneedle.a` per platform.
   `needle.h` exposes five functions. Apache-2.0 covers the 7,566 lines of Python
   only.
3. **Two components exist nowhere but in that binary.** The byte-level grammar
   compiler and the deterministic repair step (described in prose in `llms.txt`,
   with no format specification) are where its accuracy comes from. Decoding the
   weights cannot recover them — they are engine logic, not data.
4. **Size.** Even the smallest 2-layer rung is ~6 MB of weights plus a hardcoded
   `KV_BUDGET_BYTES = 11.5 MB` in `architecture.py`, so ~20 MB RAM minimum.

Note that `architecture.py` (1,136 lines) *is* a complete, runnable JAX reference
for all five custom operators — Monarch Hadamard MLP, engram hashing, mHC lanes,
GQA with conv taps, quantisation. So a port would be transcription rather than
reverse engineering. It is items 1 and 3 that make it not worth doing.

---

## 4. Seed variance

Three seeds, `baseline` arm. Reported because single-seed differences in this
project turned out to be meaningless, and one of them was published as a result
before being checked.

| config | L3 | refusal |
|---|---|---|
| `--decorate 0`, 800 steps | 93.6 +/- 6.1 | 61.6 +/- 30.6 |
| `--decorate 8`, 3000 steps | 94.4 +/- 4.9 | 20.1 +/- 3.2 |

Per-seed refusal at `--decorate 0`: 83.3 / 26.6 / 75.0. A single run of that
configuration produced the 83.3 that an earlier README reported as the headline
number. The standard deviation is 30.6 -- the model does not refuse, it sometimes
happens to.

L3 spread is +/- 5, so any comparison closer than ~10 points is noise. That
covers both the `baseline` vs `ple` gap (1.7 points) and the decoration
comparison (0.8 points).

## 5. Open questions

- Real-speech accuracy. Everything measured so far is inside the template
  distribution, including the held-out split, which shares its vocabulary and
  sentence shapes with training.
- Whether refusal is reachable with data at all. It trains on a different
  schedule from composition -- best early, degrading after -- so one checkpoint
  may not serve both. A confidence gate at inference is the more likely answer:
  reading plan-token probability and declining below a threshold turns a wrong
  action into a question.
- Chinese. Character-level vocabulary is larger and single characters are
  ambiguous; whether 2.2M parameters holds up there is untested.
