# chess-agent-simple

A chess engine in 762 lines of Python, including the trained evaluation
network. `python3 play.py` and it plays.

This is the teaching version of a competition engine I wrote for the AI
Chessathon. It runs the same network, off the same weights file, and produces
the same evaluations to the integer. What it does not have is the speed work:
the full engine is compiled with numba and keeps the network's hidden layer
incrementally, which makes it about sixty times faster and a few hundred Elo
stronger, and considerably harder to read.

Every idea that produces the strength is here. Only the optimisation is gone.

## Files

| | code | |
|---|---|---|
| `board.py` | 454 | The rules. Bitboards, move generation, make and unmake, Zobrist hashing, and the piece-square evaluation the network replaced. |
| `engine.py` | 195 | The search. Negamax with alpha beta, iterative deepening, quiescence, transposition table, move ordering. |
| `nnue.py` | 46 | The evaluation network. King-zoned inputs, two perspectives, eight output heads, all integer arithmetic. |
| `agent.py` | 67 | The entry point the competition harness calls, and the safety net under it. |
| `nnue.npz` | 6.5 MB | The trained weights. The same file the competition engine ships. |

| | | |
|---|---|---|
| `perft_test.py` | | Proves move generation is exactly right against published node counts. |
| `check_nnue.py` | | Proves the network is wired up right, and optionally that it matches the full engine exactly. |
| `play.py` | | Watch the engine play itself. |
| `human.py` | | Play against it yourself. |

`DEMO.md` is a five minute walkthrough if you just want to see it work.

## Running it

Python 3.9 or newer, and numpy for the network.

```bash
pip install numpy

python3 perft_test.py     # the rules are right, ~2s, must say ALL PASS
python3 check_nnue.py     # the network is right, ~20s, must say ALL PASS
python3 play.py           # self play at 0.2s a move
python3 human.py          # play it yourself
python3 agent.py          # the harness API on three test positions
```

Without numpy, or with `CHESS_NNUE=0`, the same code falls back to the
piece-square evaluation and everything still runs.

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
Python integers have no fixed width, so unlike the real engine this version
never has to mask off the top bits.

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
  input when your king is castled short than when it is on e1. This is the
  trick that lets the network learn king safety, which a plain 768-feature net
  cannot represent at all.
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
wrong in a way that never crashes. `check_nnue.py` is the answer to that.

**The one thing the full engine does differently** is that it never rebuilds
the hidden layer. A move touches at most two pieces, so `make` adds two weight
rows and subtracts two, `unmake` restores a copy it saved, and a king move
rebuilds that side from scratch because every index changed. Four rows of 256
integer adds per move, instead of summing 32 rows per evaluation. That is
worth about 60 lines of the most delicate code in the real `board.py`, and
skipping it is most of why this version is slower.

## What was dropped, and where it went

Everything in this table is in the full engine and not here. None of it
changes what the engine understands, only how fast it gets there.

| Dropped | What it does | Cost of dropping it |
|---|---|---|
| numba compilation | compiles the search to machine code | roughly 60x fewer positions per second |
| incremental accumulator | updates the network's hidden layer per move | this version rebuilds it per evaluation, about 7x slower |
| magic bitboards | sliding attacks by one table lookup | this version walks the rays instead |
| null move pruning | assume a free move for the opponent, prune if still winning | depth |
| late move reductions | search unpromising moves shallower first | depth |
| futility pruning | skip quiet moves that cannot reach alpha | depth |
| principal variation search | search all but the first move with a zero-width window | depth |
| aspiration windows | start each iteration near the last score | depth |
| killer and history heuristics | remember moves that cut elsewhere | ordering, so depth |
| packed 32 bit moves | a move is one integer, not a tuple | allocation |
| flat array transposition table | numba has no dicts | this version uses a dict |
| contempt | score a draw slightly below equality so the engine plays on | takes draws it should decline |

## Correctness

Two suites, and both have to pass before anything else means anything.

**`perft_test.py`** counts the legal move sequences of each length from six
positions. The counts are published and agreed on, so matching them exactly
means the generator handles every rule, including castling through check, en
passant, promotion and pins. The six positions are the ones that break
generators.

```
ALL PASS   518,468 nodes in 1.9s
```

**`check_nnue.py`** tests the network three ways:

1. *Mirror symmetry.* Swap every piece's colour, flip the board top to bottom,
   give the other side the move. That is the same position from the other
   chair, so the evaluation must be exactly equal. This catches a wrong rank
   flip or colour swap, which is the likeliest way to get the port wrong, and
   it needs nothing but this repo.
2. *Sanity anchors.* The start position is roughly level, a side up a queen is
   winning by roughly a queen.
3. *The shipping engine.* With `--engine ../chess-agent` it evaluates the same
   positions with the full engine's compiled integer path and requires exact
   equality, not a tolerance.

```
1. mirror symmetry
   3,000 of 3,000 positions match exactly
3. against the shipping engine
   2,000 of 2,000 positions match exactly
ALL PASS
```

That third check is what makes the claim "same network, same numbers" a
measurement rather than an assertion.

## What it is not

It is not fast. About 39,000 positions per second against the real engine's
2.4 million, so it reaches depth 4 where the full version reaches 12 to 18.
The evaluation is identical; the search is what is missing. It plays real
opening moves, punishes a hanging piece instantly, and will not trouble
anything serious.

Turning the network off with `CHESS_NNUE=0` roughly triples the node rate and
buys about one more ply, and it still plays worse. That trade is the whole
argument for the network, and it is easy to run both ways and watch.
