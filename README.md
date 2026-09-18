# chess-agent-simple

A chess engine in 686 lines of pure Python. No dependencies, no build step,
no data files. `python3 play.py` and it plays.

This is the teaching version of a competition engine I wrote for the AI
Chessathon. The full engine is compiled with numba and evaluates positions
with a trained neural net, which makes it about a hundred times faster and a
few hundred Elo stronger. It also makes it harder to read. Every idea that
produces the strength is in this repo; only the speed work is missing.

## Files

| | code | |
|---|---|---|
| `board.py` | 424 | The rules. Bitboards, move generation, make and unmake, Zobrist hashing, evaluation. |
| `engine.py` | 195 | The search. Negamax with alpha-beta, iterative deepening, quiescence, transposition table, move ordering. |
| `agent.py` | 67 | The entry point the competition harness calls, and the safety net under it. |

| | | |
|---|---|---|
| `perft_test.py` | | Proves move generation is exactly right against published node counts. |
| `play.py` | | Watch the engine play itself. |
| `human.py` | | Play against it yourself. |

`DEMO.md` is a two minute walkthrough if you just want to see it work.

## Running it

Python 3.9 or newer. Nothing to install.

```bash
python3 perft_test.py     # correctness, ~2s, must say ALL PASS
python3 play.py           # self play at 0.2s a move
python3 human.py          # play it yourself
python3 agent.py          # the harness API on three test positions
```

## How it works

The harness gives us a position as a FEN string and a clock in
milliseconds. We give back a move.

```
get_move(fen, ms)                 agent.py    parse, guard, return
  └── Engine.search(board, secs)  engine.py   how deep can we get in the time
        └── _negamax(...)         engine.py   the recursion
              └── evaluate()      board.py    how good is this position
```

Five ideas carry almost all of the strength.

**Bitboards.** A position is twelve integers. Bit *n* of `pieces[p]` is set
when a piece of kind *p* stands on square *n*. "Every white pawn that can
capture to the left" is one shift and one AND, not a loop over 64 squares.
Python integers have no fixed width, so unlike the real engine this version
never has to mask off the top bits.

**Pseudo-legal generation.** `generate_moves()` produces moves that look
right for the piece and ignores whether they leave our own king in check.
`push()` plays the move, checks, and returns False if it was illegal. This is
faster than proving legality up front because most moves are legal and
alpha-beta never looks at most of the ones that are.

**Negamax with alpha-beta.** One function scores every position from the
point of view of whoever is to move, and the caller negates. Alpha-beta then
skips any branch the opponent would never allow. With good move ordering it
examines roughly the square root of the positions that plain minimax would,
which is the difference between seeing three moves ahead and seeing eight.

**Move ordering.** Alpha-beta only cuts when it finds a good move early, so
the order matters more than anything else in the file. Try the transposition
table's move first, then captures sorted by most valuable victim and least
valuable attacker, then promotions, then the rest.

**Quiescence.** At depth zero, keep searching captures until the position is
quiet. Without this the engine stops counting in the middle of a trade, sees
that it has just won a pawn, and misses that its queen is hanging.

Two more things earn their place. The transposition table remembers positions
already searched, which matters because the same position is reachable by
many move orders. Iterative deepening searches depth 1, then 2, then 3, and
keeps the last completed answer, so there is always a move ready when the
clock runs out, and each shallow pass improves the ordering of the next.

## What was dropped, and where it went

Everything here is in the full engine. None of it changes what the engine
plays, only how fast it gets there, which is why this version is a fair
description of the real one.

| Dropped | What it does | Cost of dropping it |
|---|---|---|
| numba compilation | compiles the search to machine code | roughly 100x fewer nodes per second |
| magic bitboards | sliding attacks by one table lookup | this version walks the rays instead |
| NNUE evaluation | a trained net scores the position | material and piece-square tables instead |
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

`perft(n)` counts the legal move sequences of length *n*. The counts are
published and agreed on, so matching them exactly means the generator handles
every rule, including castling through check, en passant, promotion and pins.
Six positions, chosen because they are the ones that break generators.

```
ALL PASS   518,468 nodes in 2.1s
```

Run it after any change to `board.py`. A move generator that is subtly wrong
produces an engine that looks fine and loses.

## What it is not

It is not fast. 250,000 nodes per second against the real engine's 2.4
million, so it reaches depth 3 or 4 where the full version reaches 12 to 18.
It plays reasonable club chess and will punish a hanging piece instantly, but
it is not going to trouble Stockfish.

The evaluation is the weaker half. Material plus piece-square tables knows
that a knight belongs in the centre and that a king belongs behind pawns, and
knows nothing else. That is exactly the limitation the full engine's trained
net exists to fix.
