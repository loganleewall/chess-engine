# chess-agent

A chess engine in Python for the AI Chessathon: an alpha-beta search, and a
small trained evaluation network that runs in integer arithmetic.

Needs Python 3.9 or newer and numpy. Training the network also needs torch.

## Files

| | |
|---|---|
| `agent.py` | The entry point the competition harness calls, and the safety net under it. |
| `engine.py` | The search. Negamax with alpha beta, iterative deepening, quiescence, transposition table, move ordering. |
| `board.py` | The rules. Bitboards, move generation, make and unmake, Zobrist hashing, and the piece-square evaluation used when the network is off. |
| `nnue.py` | The evaluation network. King-zoned inputs, two perspectives, eight output heads, all integer arithmetic. |
| `nnue.npz` | The trained weights. |

The training pipeline, which produces `nnue.npz` and is never imported while
playing:

| | |
|---|---|
| `prepare.py` | Lichess evaluation dump to shards. The filtering is the interesting part. |
| `train.py` | Torch training and the quantized int16 export. |
| `data/sample_*.npz` | 400,000 real positions so `train.py` runs out of the box. |

## How it works

The harness gives us a position as a FEN string and a clock in milliseconds.
We give back a move.

```
get_move(fen, ms)                 agent.py    parse, guard, return
  └── Engine.search(board, secs)  engine.py   how deep can we get in the time
        └── _negamax(...)         engine.py   the recursion
              └── evaluate()      board.py    how good is this position
                    └── nnue      nnue.py     the trained network
```

### The split that decides everything else

A search calculates, a network judges. Chess is decided by what happens twelve
moves from now, and a network handed only the position would have to have
memorised that. A search computes it fresh every time. But a search needs a
number for each position it reaches, and counting material is not a good
enough number.

So the two halves do different jobs. The search handles the lookahead and
resolves anything tactical. The network only ever has to answer one question:
what is this quiet position worth? That is what makes a network this small
useful, and it is why the training data throws away every position that is in
check or where the best move is a capture.

### The search

**Bitboards.** A position is twelve integers. Bit *n* of `pieces[p]` is set
when a piece of kind *p* stands on square *n*. "Every white pawn that can
capture to the left" is one shift and one AND, not a loop over 64 squares.
Python integers have no fixed width, so there are never top bits to mask off.

**Pseudo-legal generation.** `generate_moves()` produces moves that look right
for the piece and ignores whether they leave our own king in check. `push()`
plays the move, checks, and returns False if it was illegal. That is faster
than proving legality up front, because most moves are legal and alpha beta
never looks at most of them.

**Negamax with alpha beta.** One function scores every position from the point
of view of whoever is to move, and the caller negates. Alpha beta then skips
any branch the opponent would never allow. With good move ordering it examines
roughly the square root of the positions plain minimax would.

**Move ordering.** Alpha beta only cuts when it finds a good move early, so
the order matters more than anything else in the file. Try the transposition
table's move first, then captures sorted by most valuable victim and least
valuable attacker, then promotions, then the rest.

**Quiescence.** At depth zero, keep searching captures until the position is
quiet. Without this the engine stops counting in the middle of a trade, sees
that it has just won a pawn, and misses that its queen is hanging.

Two more things earn their place. The transposition table remembers positions
already searched, because the same position is reachable by many move orders.
Iterative deepening searches depth 1, then 2, then 3, keeping the last
completed answer, so there is always a move ready when the clock runs out and
each shallow pass improves the ordering of the next.

### The network

    (16 x 768 -> 256) x 2 -> 8 x 1

- **768** is the plain feature count: 2 colours x 6 piece types x 64 squares.
  One input per fact of the form "there is a white knight on f3".
- **x 16** is the king zone. Which of 16 zones *your own king* stands in
  shifts every one of your input indices, so a knight on f5 is a different
  input when your king is castled short than when it is on e1. This is what
  makes king safety easy to learn. A plain 768-feature net can only get at it
  indirectly, through pairs of pieces squeezed through 256 shared hidden
  units; with king zones, "knight near my king" is a single input.
- **x 2** is the two perspectives. Each side sees the board from its own point
  of view, colours swapped and ranks mirrored, and both share one weight
  table.
- **8 x 1** is eight output heads chosen by how many pieces are left, so an
  endgame gets its own scale.

The first layer is an embedding lookup, not a matrix multiply. A position has
at most 32 pieces, so at most 32 of 12,288 inputs are set, and the hidden
layer is the sum of those 32 rows. The nonlinearity is "clamp to [0, 255] and
square", which is one multiply.

Everything at inference is integer arithmetic, because integer adds are what
vectorise. The trainer works in float and the export rounds to int16, which
means two implementations of the same network exist and one of them could be
wrong in a way that never crashes. `train.py` compares the two on held-out
positions after every export.

## Training the network

`nnue.npz` did not come from anywhere. `prepare.py` and `train.py` are how it
was made, and both run.

```bash
pip install torch
python3 train.py
```

```
1 shard(s), 397,312 positions per epoch, 20,000 held out
hidden 128, batch 4096, 10 epochs, device mps

  before training   val 0.06549
  epoch   1        train 0.04412   val 0.03556       2s
  epoch   5        train 0.01407   val 0.02608      10s
  epoch  10        train 0.00973   val 0.02472      20s

  quantization error   mean 8.3 cp, max 42 cp
  vs stockfish         mean 341 cp (predicting 0 would be 506 cp)

  wrote mynet.npz
  to use it:  cp mynet.npz nnue.npz
```

### Where the data comes from

The labels are Stockfish evaluations, taken from the Lichess evaluation dump:
395 million positions somebody else already paid the compute to evaluate
deeply, published CC0. `prepare.py` streams that file, filters it, and writes
shards.

```bash
curl -s https://database.lichess.org/lichess_db_eval.jsonl.zst \
  | zstd -dc | head -n 4000000 | python3 prepare.py - data/shard
```

It keeps about 15% of what it reads, and the discards are the interesting
part:

| Dropped | Why |
|---|---|
| evaluated shallower than depth 10 | the label is not worth learning from |
| the position is in check | the search resolves it |
| the best move is a capture | the search resolves it too |
| the position could not occur in a game | malformed records |

Throwing away every sharp position looks wasteful and is the opposite. The
search has quiescence, which plays out all the captures before it asks for a
score, so the network is never called on a position in the middle of a trade.
Train it on those anyway and it spends its capacity on a job the search does
better.

The shards hold raw piece lists, not network input indices. Those depend on
where each king stands, so they are derived at training time by `train.py` and
at play time by `nnue.py`, which means the king zone scheme can change without
reprocessing 21 GB.

### What the network is asked to predict

    target = sigmoid(centipawns / 400)

Not centipawns directly. One position at +12000 would otherwise outweigh a
thousand ordinary ones at +30, and the network would spend itself learning to
shout. Squashing to a win probability bounds the label, so being wrong about a
won position costs about what being wrong about a level one costs.

Weights are clamped to +/-1.98 after every optimiser step. That is what makes
`round(weight * 255)` safe to put in an int16 at export time, and it is why
the quantization error above is 8 cp rather than nonsense.

### The bundled sample is deliberately too small

400,000 positions, against the 96 million the shipped network was trained on.
Enough to watch it learn, far too few to learn properly. The dump is mostly
real games near material balance, so 400,000 of them contain almost nothing
as lopsided as a side a queen down, and a net trained on the sample has no
idea what that is worth.

Measured on the same held-out positions:

| | mean error vs Stockfish |
|---|---|
| predicting 0 for everything | 506 cp |
| trained on the bundled 400k sample | 341 cp |
| the shipped `nnue.npz`, 96M positions | 252 cp |

## Speed

About 39,000 positions per second, which gets it to around depth 4. It plays
real opening moves, punishes a hanging piece instantly, and will not trouble
anything serious.

Turning the network off with `CHESS_NNUE=0` roughly triples the node rate and
buys about one more ply, and it still plays worse. That trade is the whole
argument for the network.
