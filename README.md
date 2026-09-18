# pet-lm

A ~2.2M-parameter instruction model for a desk-pet robot: English command in,
action sequence out, small enough to target an ESP32-S3.

The robot is an STM32F103C8T6 desk pet driving five servos (four legs, one tail)
with twelve behaviours, plus a thirteenth `stop` this project added. Its existing control path is fixed voice command words.
The question this repo answers is whether a model small enough to sit beside it
can accept free-form instructions instead, including composed ones
("walk forward two steps and afterwards lie down").

**Status: feasibility confirmed on host. Not yet on hardware.**

## Result

Three seeds, `baseline` arm, ~95 s per run on an RTX 5060. Held-out accuracy,
exact match on the full action sequence under greedy decoding constrained to the
plan-token span.

| level | what it tests | accuracy |
|---|---|---:|
| L1 | single action, unseen wording | 98.5% +/- 1.3 |
| L2 | action + repeat count | 99.8% +/- 0.4 |
| L3 | two actions, unseen connective | 88.8% +/- 14.9 |
| prohibit | "do not sit down" -> no action | 65.7% +/- 4.2 |
| neg | refuse what it cannot act on | 18.0% +/- 4.6 |

Core parameters: 2,214,336. Thirteen actions.

**Order is never reversed.** Across 1500 composed held-out instructions the model
did not once emit the two actions in the wrong order. Composition semantics are
learned; 2.2M parameters is enough for this task.

**L3's spread is the headline caveat.** Per seed: 71.7 / 95.5 / 99.2. A single run
reading 97.5% is not evidence of anything. This has now happened twice in this
project, both times on a number that looked like a result.

**Refusal does not work** and is not close. See below.

## Negation

The worst bug found so far, and the reason `probe.py` exists:

```
do not sit down     -> <squat>   P(stop) = 0.000
don't lie down      -> <lie>     P(stop) = 0.000
don't jump          -> <jump>    P(stop) = 0.000
```

Every prohibition executed the action it forbade, with no hesitation. The cause
was not that the model misread the sentence. `not`, `don't` and `never` appeared
in no template, so they had no vocabulary entry and encoded as `<unk>`:

```
"do not sit down"  ->  ['do', '<unk>', 'sit', 'down']
"do sit down"      ->  ['do',          'sit', 'down']
```

Identical input. Negation was deleted at tokenisation, before the model saw it.
The lesson is about the pipeline, not the model: `template_words()` derived the
vocabulary from the templates, so any word class absent from the templates was
absent from the model's world, silently.

Fixed by adding prohibition patterns (180 of them, `PROHIBIT_PATTERNS` x
`ACTION_VERBS`), a thirteenth `<stop>` action, and by making `template_words()`
collect every source of surface text. Now:

```
do not sit down     -> (nothing)  P(stop) = 1.000
never lie down      -> (nothing)  P(stop) = 1.000
stop / wait / stay  -> <stop>
don't move          -> <stop>
```

`stop` is an action, not a refusal: halting the servos is something the pet does,
and an empty plan would leave it mid-gait.

Prohibitions sit at 65.7% on held-out patterns -- worse than the commands but
stable across seeds, and the failures are silence-vs-action rather than executing
the forbidden move.

## Refusal

This is the honest gap. 18% +/- 4.6, and it is not a variance problem:

```
are you a robot     -> <squat>   P(stop) = 0.000
call my mother      -> <squat>   P(stop) = 0.000
feed the cat        -> <squat>   P(stop) = 0.000
ah / eh / hm        -> <squat>   P(stop) = 0.000
```

32 of 33 unseen non-instructions produced an action, mostly the same default one,
at probability 0.000. There is no uncertainty to threshold on -- the model is
confidently wrong, which rules out the confidence gate that seemed like the
obvious fix.

An earlier probe set suggested refusal worked; 8 of its 14 sentences turned out to
be verbatim training rows. Within a single kind, trained items are right and
held-out items are wrong: `and` refuses correctly at P(stop) = 0.999 while `ah`,
`eh`, `hm` all act. That is memorisation of 69 negative sentences, not a learned
distinction.

Two structural reasons. Refusal and composition train on opposite schedules --
refusal is best early and decays, L3 peaks late -- so one checkpoint cannot serve
both. And there are 69 hand-written negatives against ~2400 positive phrasings, a
ratio at which "this is an instruction" is simply the right prior.

### Two things that look like wins and are not

**Training longer.** Loss reaches 0.0000 well before the best operating point,
which is the tell that the templates are memorisable. Single seed, undecorated:

| steps | L1 | L2 | L3 | neg |
|---:|---:|---:|---:|---:|
| 400 | 100.0 | 99.4 | 95.6 | 75.0 |
| 800 | 100.0 | 98.8 | 97.5 | 83.3 |
| 1200 | 100.0 | 99.4 | 91.5 | 41.7 |
| 2000 | 100.0 | 99.4 | 73.9 | 41.7 |
| 3000 | 100.0 | 99.4 | 88.9 | 55.6 |

**More rows.** `--train 40000` over ~500 hand-written phrasings is a repeat factor
of 60x per template, so raising it adds no information. `--decorate N` generates
modifier variants instead (prefix, suffix or wrapper, one slot per phrasing) and
takes unique L1 texts from 226 to ~1450. It buys nothing on L3 within noise and
needs 3000 steps rather than 800.

**Decorating the negatives was strictly worse.** Applying the same modifiers to
negative rows put "please", "can you", "now" into 86% of negatives alongside 89%
of positives, erasing the only cue separating an instruction from a remark.
Refusal fell to 27% while L3 did not move. Negatives are never decorated now,
which does not fix refusal either -- it just removes one way of breaking it.

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
| `probe.py` | emit the plan and P(stop) for one sentence; how the negation bug surfaced |

## Run

```sh
python gen_data.py            # -> train.jsonl test.jsonl
python vocab.py               # -> vocab.json
python train_pet.py --arm baseline --steps 3000 --out pet.pt
python inspect_errors.py pet.pt
python probe.py pet.pt          # sanity-check commands, prohibitions, refusals
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

- **Accuracy is an upper bound.** Templates are narrower than real speech, and the
  held-out split shares their vocabulary and sentence shapes. A real evaluation
  needs human-written instructions, which do not exist yet. That test decides
  whether anything below is worth building.
- **Refusal does not work** (18% +/- 4.6) and a confidence gate will not fix it:
  the wrong answers come at P(stop) = 0.000.
- **L3 varies +/- 15 across seeds.** Any comparison closer than ~30 points is
  noise at this sample size, which covers both the `baseline`/`ple` gap and the
  decoration comparison.
- **Prohibition-plus-alternative is unsupported.** "don't sit, lie down instead"
  needs a plan representation that can express suppression.
- **English only.** The pet's voice module is Chinese. A character-level Chinese
  vocabulary is larger, ambiguous at the single-character level ("走" spans
  several senses), and needs UTF-8 handling in the device tokeniser. English
  validates the architecture with one variable changed at a time.
- **Nothing has run on hardware.** The device numbers in `NOTES.md` are
  projections calibrated against upstream's measurements, not measurements.

## Next

1. **Human-written test set.** Everything else is downstream of knowing the real
   accuracy. ~200 instructions from someone who has not seen `templates.py`.
2. **Refusal needs a different mechanism, not more data.** A separate binary
   "is this an instruction?" classifier trains on its own schedule and can be
   tiny. One model serving both objectives is the current bottleneck.
3. **Drop `ple`.** Within noise of `baseline` at this vocabulary, and it costs 10%
   more parameters plus four tensor groups in the firmware.
4. **Port the core matvec to the int8-staged path** used for the output head.
   Upstream stages only the head, leaving the core at a measured 18.1 MB/s against
   the head's 56.9 MB/s; this is the single largest device-side win.
5. Export to `.cact`, pass the host golden gate, flash, measure.

## Licence

MIT, matching the upstream projects this derives from.
