"""Command templates for the 12-action desk-pet instruction model.

DESIGN NOTES
------------
The model's job is translation, not generation: an English sentence goes in and a
short action sequence comes out. Everything here exists to make that mapping
learnable by a ~2.4M-parameter core.

Three deliberate choices:

1. ACTIONS ARE THE OUTPUT VOCABULARY. The model can only emit action/count
   tokens, so a malformed plan is unrepresentable. This replaces a constrained
   decoder for free.

2. PHRASINGS ARE HAND-WRITTEN, NOT PARAPHRASED BY A MODEL. Generated paraphrases
   drift toward one register and inflate held-out accuracy, because the held-out
   split ends up sampling the same generator. These are grouped by register
   (bare / polite / question / casual) so a split can hold out a whole register
   and measure something real.

3. NEGATIVES ARE FIRST-CLASS. Without them the model emits its best guess for
   "what is your name", which is worse than emitting nothing: a desk pet that
   lies down when spoken to is broken in a way a silent one is not.

Held-out discipline: `HELDOUT_PHRASINGS` are never used in training. They are
real alternate wordings, not template recombinations, so accuracy on them
reflects generalisation rather than memorisation.
"""

# ---------------------------------------------------------------------------
# Action inventory. Matches the 12 behaviours the STM32 firmware already drives.
# `takes_count` marks the ones a repeat count is meaningful for; asking a dog to
# "sleep three times" is not a command anyone gives, so those stay unary.
# ---------------------------------------------------------------------------
ACTIONS = {
    "stand":   {"token": "<stand>",   "takes_count": False},
    "forward": {"token": "<fwd>",     "takes_count": True},
    "back":    {"token": "<back>",    "takes_count": True},
    "left":    {"token": "<left>",    "takes_count": True},
    "right":   {"token": "<right>",   "takes_count": True},
    "tail":    {"token": "<tail>",    "takes_count": False},
    "lie":     {"token": "<lie>",     "takes_count": False},
    "squat":   {"token": "<squat>",   "takes_count": False},
    "sleep":   {"token": "<sleep>",   "takes_count": False},
    "jump":    {"token": "<jump>",    "takes_count": True},
    "greet":   {"token": "<greet>",   "takes_count": False},
    "stretch": {"token": "<stretch>", "takes_count": False},
}

# ---------------------------------------------------------------------------
# L1 - single action, no count. Grouped by register on purpose: holding out
# "casual" entirely is a much harder and more honest test than holding out a
# random 10% of everything.
# ---------------------------------------------------------------------------
PHRASINGS = {
    "stand": {
        "bare":     ["stand", "stand up", "attention", "stand at attention",
                     "stand still", "at attention"],
        "polite":   ["please stand up", "stand up please", "come to attention"],
        "question": ["can you stand up", "will you stand", "would you stand up"],
        "casual":   ["up you get", "on your feet", "straighten up"],
    },
    "forward": {
        "bare":     ["forward", "go forward", "walk forward", "move forward",
                     "step forward", "go ahead", "walk ahead", "move ahead",
                     "go straight"],
        "polite":   ["please go forward", "walk forward please", "move forward please"],
        "question": ["can you walk forward", "will you go forward",
                     "would you move ahead"],
        "casual":   ["scoot forward", "come here", "come forward", "head forward"],
    },
    "back": {
        "bare":     ["back", "go back", "walk back", "move back", "back up",
                     "step back", "go backwards", "move backwards", "reverse"],
        "polite":   ["please go back", "back up please", "move back please"],
        "question": ["can you back up", "will you go back", "would you step back"],
        "casual":   ["scoot back", "get back", "back off"],
    },
    "left": {
        "bare":     ["left", "turn left", "go left", "turn to the left",
                     "face left", "turn left side"],
        "polite":   ["please turn left", "turn left please"],
        "question": ["can you turn left", "will you turn left", "would you face left"],
        "casual":   ["swing left", "spin left", "look left"],
    },
    "right": {
        "bare":     ["right", "turn right", "go right", "turn to the right",
                     "face right", "turn right side"],
        "polite":   ["please turn right", "turn right please"],
        "question": ["can you turn right", "will you turn right", "would you face right"],
        "casual":   ["swing right", "spin right", "look right"],
    },
    "tail": {
        "bare":     ["wag tail", "wag your tail", "shake tail", "shake your tail",
                     "tail wag", "wave your tail"],
        "polite":   ["please wag your tail", "wag your tail please"],
        "question": ["can you wag your tail", "will you wag your tail"],
        "casual":   ["wag it", "give me a wag", "waggle that tail"],
    },
    "lie": {
        "bare":     ["lie down", "lay down", "lie", "get down", "down",
                     "lie flat", "go down"],
        "polite":   ["please lie down", "lie down please"],
        "question": ["can you lie down", "will you lie down", "would you lay down"],
        "casual":   ["flop down", "get down there", "hit the deck"],
    },
    "squat": {
        "bare":     ["sit", "sit down", "squat", "squat down", "take a seat",
                     "crouch", "crouch down"],
        "polite":   ["please sit down", "sit please", "sit down please"],
        "question": ["can you sit", "will you sit down", "would you sit"],
        "casual":   ["park it", "have a seat", "sit tight"],
    },
    "sleep": {
        "bare":     ["sleep", "go to sleep", "sleep now", "take a nap",
                     "go to bed", "time to sleep", "rest"],
        "polite":   ["please go to sleep", "go to sleep please"],
        "question": ["can you go to sleep", "will you sleep", "would you take a nap"],
        "casual":   ["nap time", "get some rest", "lights out", "catch some sleep"],
    },
    "jump": {
        "bare":     ["jump", "jump forward", "hop", "hop forward", "leap",
                     "leap forward", "jump up"],
        "polite":   ["please jump", "jump please"],
        "question": ["can you jump", "will you jump", "would you hop"],
        "casual":   ["give me a jump", "hop to it", "bounce"],
    },
    "greet": {
        "bare":     ["say hi", "say hello", "greet me", "wave hello", "wave hi",
                     "hello", "hi there", "say hey"],
        "polite":   ["please say hello", "say hello please", "greet me please"],
        "question": ["can you say hi", "will you say hello", "would you greet me"],
        "casual":   ["give me a hello", "wave at me", "hey there"],
    },
    "stretch": {
        "bare":     ["stretch", "stretch yourself", "have a stretch",
                     "do a stretch", "stretch out"],
        "polite":   ["please stretch", "stretch please"],
        "question": ["can you stretch", "will you stretch", "would you stretch out"],
        "casual":   ["big stretch", "loosen up", "stretch it out"],
    },
}

# Never trained on. Real alternate wordings, so accuracy here is generalisation.
HELDOUT_PHRASINGS = {
    "stand": ["get up", "rise up", "stand to attention", "stand tall"],
    "forward": ["advance", "proceed forward", "walk on", "move along",
                "keep going forward"],
    "back": ["retreat", "go in reverse", "pull back", "move away"],
    "left": ["rotate left", "turn towards the left", "veer left"],
    "right": ["rotate right", "turn towards the right", "veer right"],
    "tail": ["swish your tail", "move your tail", "flick your tail"],
    "lie": ["lie on the floor", "settle down", "get flat"],
    "squat": ["be seated", "sit yourself down", "lower yourself"],
    "sleep": ["go and sleep", "have a nap", "doze off", "sleep tight"],
    "jump": ["spring up", "take a leap", "jump once"],
    "greet": ["introduce yourself", "welcome me", "say good morning"],
    "stretch": ["extend yourself", "have a good stretch", "reach out"],
}

# ---------------------------------------------------------------------------
# L2 - count-carrying phrasings. `{n}` is the numeral, `{unit}` the noun.
# Count words stay 1..5: past that the count stops being an instruction anyone
# gives and the model would be memorising numerals it never has to act on.
# ---------------------------------------------------------------------------
COUNT_PATTERNS = {
    "forward": [
        "go forward {n} {unit}", "walk forward {n} {unit}",
        "move forward {n} {unit}", "step forward {n} {unit}",
        "forward {n} {unit}", "walk {n} {unit} forward",
        "go {n} {unit} forward", "take {n} {unit} forward",
        "move ahead {n} {unit}",
    ],
    "back": [
        "go back {n} {unit}", "walk back {n} {unit}", "move back {n} {unit}",
        "back up {n} {unit}", "step back {n} {unit}", "back {n} {unit}",
        "walk {n} {unit} back", "take {n} {unit} back",
    ],
    "left": [
        "turn left {n} times", "turn left {n}", "left {n} times",
        "spin left {n} times", "turn to the left {n} times",
    ],
    "right": [
        "turn right {n} times", "turn right {n}", "right {n} times",
        "spin right {n} times", "turn to the right {n} times",
    ],
    "jump": [
        "jump {n} times", "hop {n} times", "jump {n}", "leap {n} times",
        "give me {n} jumps",
    ],
}

NUMERALS = {1: ["one", "1"], 2: ["two", "2"], 3: ["three", "3"],
            4: ["four", "4"], 5: ["five", "5"]}
# "one steps" is ungrammatical, so the generator must agree number with unit.
UNITS_SINGULAR = ["step", "pace"]
UNITS_PLURAL = ["steps", "paces"]

# ---------------------------------------------------------------------------
# L3 - composition. The real test: two actions in a stated order.
# `{a}` and `{b}` are filled with L1/L2 phrasings, so composition inherits the
# whole phrasing space rather than a fixed pair list.
# ---------------------------------------------------------------------------
JOINERS = [
    "{a} then {b}",
    "{a} and then {b}",
    "{a} then {b} please",
    "first {a} then {b}",
    "{a} after that {b}",
    "{a} and {b}",
    "{a} next {b}",
    "{a} followed by {b}",
    "do {a} then {b}",
]

# Held-out connectives must not share a word with any other role. An earlier
# version used "{a} once done {b}" while the count patterns had been fixed to
# emit "jump once" for n=1, so `once` was simultaneously a numeral and half a
# connective: L3 accuracy on that joiner collapsed to 26% and count errors rose
# to 8%, because "X once done Y" reads as "X one time". `done` also appeared in
# no training row at all, so the connective was untestable by construction.
# Each must also be order-unambiguous. "{a} straight after {b}" was rejected:
# it reads as B happening first, so the gold plan would be reversed.
HELDOUT_JOINERS = [
    "{a} and afterwards {b}",
    "{a} and later {b}",
    "{a} and subsequently {b}",
]

# ---------------------------------------------------------------------------
# Negatives -> empty plan. Four kinds, because a single kind teaches the model
# to reject one surface pattern rather than to reject what it cannot act on.
# ---------------------------------------------------------------------------
NEGATIVES = {
    # chit-chat
    "chat": [
        "what is your name", "how are you", "how are you today",
        "nice weather isn't it", "tell me a joke", "what time is it",
        "who made you", "do you like me", "are you happy", "good morning",
        "what are you doing", "how old are you", "where are you from",
        "sing me a song", "what do you think", "are you a robot",
        "do you have a name", "who is your owner", "what can you do",
        "how was your day", "are you tired", "do you dream",
        "what is that noise", "is anyone there", "are you listening",
        "how much do you weigh", "what colour are you", "do you get bored",
    ],
    # asks for capabilities this pet does not have -> must NOT map to a neighbour
    "unsupported": [
        "make me a coffee", "turn on the light", "play some music",
        "call my mother", "open the door", "take a photo",
        "read me the news", "set an alarm", "what is the weather",
        "order some food", "send a message", "roll over", "fetch the ball",
        "speak louder", "count to ten", "turn off the tv",
        "water the plants", "clean the floor", "find my keys",
        "book me a table", "translate this", "solve this problem",
        "paint a picture", "drive me home", "wash the dishes",
        "check my email", "start the car", "feed the cat",
    ],
    # near-misses: action words present but no actual instruction
    "nearmiss": [
        "do you know how to sit", "i can walk forward",
        "my dog can jump", "the tail is broken", "sleeping is nice",
        "left and right are directions", "standing is tiring",
        "i like it when you wag your tail", "jumping looks fun",
        "he told me to sit down", "i am going to lie down",
        "she said turn left", "walking is good exercise",
        "do you like to jump", "was that a stretch",
        "my cat sleeps all day", "the left side is broken",
        "i forgot how to squat", "they asked you to stand",
        "stretching helps your back", "greeting people is polite",
        "his tail wags a lot", "we walked forward together",
        "you were sleeping earlier", "that jump was high",
        "turning right was wrong", "i sat down already",
    ],
    "empty": ["", "um", "uh", "hmm", "ok", "well", "so", "and",
              "er", "ah", "oh", "eh", "mm", "hm", "yeah", "no",
              "maybe", "sure", "right then", "anyway"],
}
