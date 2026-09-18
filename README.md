# pet-lm

A ~2.2M-parameter instruction model for a desk-pet robot: English command in,
action sequence out, small enough to target an ESP32-S3.

The robot is an STM32F103C8T6 desk pet driving five servos (four legs, one tail)
with twelve behaviours. Its existing control path is fixed voice command words.
The question this repo answers is whether a model small enough to sit beside it
can accept free-form instructions instead, including composed ones
("walk forward two steps and afterwards lie down").

**Status: feasibility confirmed on host. Not yet on hardware.**

## Result

Three seeds, `baseline` arm, ~25–95 s per run on an RTX 5060. Held-out accuracy,
exact match on the full action sequence under greedy decoding constrained to the
plan-token span.

| level | what it tests | accuracy |
|---|---|---:|
| L1 | single action, unseen wording | 97.6–100% |
| L2 | action + repeat count | 98.8–100% |
| **L3** | **two actions, unseen connective** | **94.4% ± 4.9** |
| neg | refuse what it cannot act on | **20% ± 3** — see below |

Core parameters: 2,214,336.

**Order is never reversed.** Across 1500 composed held-out instructions the model
did not once emit the two actions in the wrong order. Composition semantics are
learned; 2.2M parameters is enough for this task.

**Refusal does not work.** This is the honest headline. A single seed once scored
83%, which an earlier version of this file reported as the result; across three
seeds the same configuration gives 83 / 27 / 75, standard deviation 30.6. The 83%
was luck. Refusal is not a capability this model has, it is noise.

### Two things that look like wins and are not

**Training longer.** Loss reaches 0.0000 well before the best operating point,
which is the tell that the templates are memorisable. Single seed, undecorated:

| steps | L1 | L2 | L3 | neg |
|---:|---:|---:|---:|---:|
| 400 | 100.0 | 99.4 | 95.6 | 75.0 |
| **800** | 100.0 | 98.8 | **97.5** | 83.3 |
| 1200 | 100.0 | 99.4 | 91.5 | 41.7 |
| 2000 | 100.0 | 99.4 | 73.9 | 41.7 |
| 3000 | 100.0 | 99.4 | 88.9 | 55.6 |

The 2000-step row shows how far off an arbitrary step count can land, and the
non-monotonic recovery at 3000 is the variance that the three-seed run confirms.

**More rows.** `--train 40000` over ~500 hand-written phrasings is a repeat
factor of 60x per template, so raising it adds no information. `--decorate N`
generates modifier variants instead (prefix, suffix or wrapper, one slot per
phrasing) and takes unique L1 texts from 226 to 1310. Three seeds:

| config | L3 | neg |
|---|---:|---:|
| `--decorate 0`, 800 steps | 93.6 ± 6.1 | 61.6 ± 30.6 |
| `--decorate 8`, 3000 steps | 94.4 ± 4.9 | 20.1 ± 3.2 |

Decoration buys nothing on L3 — within noise — and it costs the ability to look
like it refuses. Its one real effect is that the model now needs 3000 steps
rather than 800, because there is more surface to fit.

**Decorating the negatives was strictly worse.** An earlier attempt applied the
same modifiers to negative rows for symmetry. That put "please", "can you",
"now" into 86% of negatives alongside 89% of positives, erasing the only cue
separating an instruction from a remark, and refusal fell to 27% while L3 did not
move. Negatives are now never decorated — which does not fix refusal either, it
just removes one way of breaking it.

### Why refusal fails

Refusal and composition train at different rates. L3 peaks late (3000 steps with
decoration) while refusal is best early and degrades from there, so no single
step count serves both. Underneath that, there are 69 hand-written negatives
against ~2400 positive phrasings: at that ratio the model's prior is "this is an
instruction", and 34 held-out negative rows is too few to measure the result
reliably. Refusal is likely to need a confidence gate at inference rather than
more data — the pet can ask instead of guessing.

## What this is built on

The architecture is [manjunathshiva/esp32-tinyllm](https://github.com/manjunathshiva/esp32-tinyllm)
(MIT), itself building on [slvDev/esp32-ai](https://github.com/slvDev/esp32-ai)
(MIT), which established Per-Layer-Embeddings on a microcontroller. `src/model.py`
from that repo is imported unchanged; this repo replaces only the data path and
the metric. Clone it as a sibling directory:

```
<parent>/
  tinyllm/     # git clone https://github.com/manjunathshiva/esp32-tinyllm
  pet-lm/      # this repo
```

## Why the design is shaped this way

**Word-level vocabulary, not BPE.** English commands in this domain draw on a
closed set of 291 word forms. BPE would buy nothing and cost a 197 KB merge
table plus the failure mode upstream documents: a prompt tokenised differently
from training is silently out of distribution with nothing in the logs. Splitting
on whitespace is exact by construction and the device tokeniser becomes a hash
lookup.

**Action tokens are the output vocabulary.** Decoding is masked to a contiguous
token span, so a malformed plan is unrepresentable rather than merely unlikely.
This replaces a constrained-decoding grammar for free.

**PLE is probably not worth keeping.** Its published advantage comes from a large
vocabulary (upstream ablation: +0.025 nats at vocab 4096 versus +0.098 at
32768). At vocab 312 the two arms are within noise of each other, and `ple` costs
10% more parameters plus four tensor groups in the firmware. Deciding this
properly needs multiple seeds; it has not been done.

**Loss is masked to the plan region.** Training the model to predict the
operator's own words would spend capacity on a distribution it never generates.

## Files

| path | what |
|---|---|
| `templates.py` | 12 actions x 4 registers, count patterns, connectives, negatives |
| `gen_data.py` | three difficulty levels, held-out splits, negative sampling |
| `vocab.py` | word-level vocabulary, encoding, plan-token span |
| `train_pet.py` | training with per-level exact-match scoring |
| `inspect_errors.py` | failure-mode attribution (order / drop / substitution / count) |

## Run

```sh
python gen_data.py            # -> train.jsonl test.jsonl
python vocab.py               # -> vocab.json
python train_pet.py --arm baseline --steps 3000 --out pet.pt
python inspect_errors.py pet.pt
```

## Evaluation discipline

Two mistakes were made and fixed while building this, both of which produced
plausible-looking numbers:

**Held-out phrasings introduced unseen words.** A word-level model cannot
generalise to a word it has no embedding for — there is no subword structure to
fall back on. 46% of held-out tokens encoded as `<unk>`, so those rows carried no
signal, and L1 accuracy *fell* from 61% to 49% as training sharpened a mapping
for a token that was never supervised. The vocabulary now covers the domain's
closed word set and generalisation is tested where it is testable: unseen
*combinations* of known words. This mirrors deployment, where the recogniser's
word list is fixed and known.

## Known limits

- **Accuracy is an upper bound.** Templates are narrower than real speech, and
  the held-out split shares their vocabulary and sentence shapes. A real
  evaluation needs human-written instructions, which do not exist yet. That test
  decides whether anything below is worth building.
- **Refusal does not work** (20% ± 3, and the one 83% reading was noise). The
  held-out negative set is 34 rows, too few to measure properly even if it did.
- **English only.** The pet's voice module is Chinese. A character-level Chinese
  vocabulary is larger, ambiguous at the single-character level ("走" spans
  several senses), and needs UTF-8 handling in the device tokeniser. English
  validates the architecture with one variable changed at a time.
- **L3 has ±5 spread across seeds**, so differences under ~10 points mean
  nothing. Both the `baseline`/`ple` gap and the decoration comparison fall
  inside that band.
- **Nothing has run on hardware.** The device numbers in `NOTES.md` are
  projections calibrated against upstream's measurements, not measurements.

## Next

1. **Human-written test set.** Everything else is downstream of knowing the real
   accuracy. ~200 instructions from someone who has not seen `templates.py`.
2. **A confidence gate rather than more negative data.** Refusal trains on a
   different schedule from composition, so the two probably cannot be served by
   one checkpoint. Reading the plan-token probability and declining below a
   threshold turns a wrong action into a question, which is the correct
   behaviour for a desk pet anyway.
3. **Drop `ple`.** Within noise of `baseline` at this vocabulary, and it costs
   10% more parameters plus four tensor groups in the firmware.
4. **Port the core matvec to the int8-staged path** used for the output head.
   Upstream stages only the head, leaving the core at a measured 18.1 MB/s
   against the head's 56.9 MB/s; this is the single largest device-side win.
5. Export to `.cact`, pass the host golden gate, flash, measure.

## Licence

MIT, matching the upstream projects this derives from.
