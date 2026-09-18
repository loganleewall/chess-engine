# Demo

Six commands, about eight minutes. Python 3.9 or newer and numpy.

```bash
cd chess-agent-simple
pip install numpy
```

## 1. It obeys the rules (20 seconds)

```bash
python3 perft_test.py
```

`perft(n)` counts the legal move sequences of length *n* from a position.
Those counts are published and agreed on, so matching them exactly is proof
that the move generator handles castling through check, en passant,
promotion, pins and the rest.

```
position 6
   depth 1          46  ok
   depth 2       2,079  ok
   depth 3      89,890  ok         305,163 nps

ALL PASS   518,468 nodes in 1.9s (266,624 nps)
```

A single wrong number means a bug. This is the first thing to run after
touching `board.py`.

## 2. The network is wired up right (30 seconds)

```bash
python3 check_nnue.py
```

The evaluation is a trained network running off `nnue.npz`, in integer
arithmetic. A quantization bug there does not crash, it just makes the engine
quietly worse, so it gets its own test.

```
1. mirror symmetry
   3,000 of 3,000 positions match exactly

2. sanity anchors
        38 cp  ok    the start position is roughly level
      -782 cp  ok    White is a queen down, to move
       863 cp  ok    White is a queen up, to move
```

Mirror symmetry is the interesting one. Swap every piece's colour, flip the
board top to bottom, hand the other side the move: that is the same position
from the other chair, so the number has to come back exactly equal. The
network serves both sides from one weight table by mirroring ranks and
swapping colours, so if any of that is wrong the two numbers disagree.

To prove it is the *same* network as the competition engine, not just a
self-consistent one:

```bash
python3 check_nnue.py --engine ../chess-agent
```

```
3. against the shipping engine
   2,000 of 2,000 positions match exactly
```

That needs the full repo and its numba environment.

## 3. It plays itself (30 seconds)

```bash
python3 play.py           # 0.2s a move, 60 moves
python3 play.py 1.0 200   # slower and longer
```

Each line is one move: the depth it finished, the score in centipawns from the
mover's point of view, and how many positions it looked at.

```
  1. White e2e4   depth 4   score     37    10,455 nodes
  2. Black d7d5   depth 4   score     -5     8,216 nodes
  3. White e4d5   depth 4   score     43     9,574 nodes
  4. Black d8d5   depth 3   score    -43    10,240 nodes
  5. White b1c3   depth 3   score     27    10,240 nodes
```

That is 1.e4 d5 2.exd5 Qxd5 3.Nc3, the Scandinavian, with the knight gaining a
tempo on the queen. Nothing in this repo knows any opening theory. It is
playing a real line because the network likes those positions.

Worth running the other way for contrast:

```bash
CHESS_NNUE=0 python3 play.py
```

That turns the network off and falls back to material plus piece-square
tables. The node rate roughly triples and it gets about one more ply of depth,
and it still plays worse. That trade is the whole argument for the network.

## 4. Train one yourself (30 seconds)

```bash
pip install torch
python3 train.py
```

The weights in `nnue.npz` were made by this file. It runs on 400,000 real
positions bundled in `data/`, which is enough to watch it learn and far too
few to learn properly.

```
  before training   val 0.06549
  epoch   5        train 0.01407   val 0.02608      10s
  epoch  10        train 0.00973   val 0.02472      20s

  quantization error   mean 8.3 cp, max 42 cp
  vs stockfish         mean 341 cp (predicting 0 would be 506 cp)
```

Two things to point at. The quantization line is the float network and the
integer one being compared on the same positions after the export, because the
integer one is what plays and rounding is where a silent bug would live. And
341 against a baseline of 506 means it learned something real, while the
shipped network gets 252 on the same positions, which is what 96 million
positions buys over 400,000.

Drop it in and play with it:

```bash
cp mynet.npz nnue.npz && python3 play.py
```

Then run `check_nnue.py` on it. Mirror symmetry still passes, because the code
is right. The sanity anchors fail, because the network never saw enough
lopsided material to know what a queen is worth. One check tests the code, the
other tests the data.

## 5. It answers the harness (10 seconds)

```bash
python3 agent.py
```

The competition API is one function, `get_move(fen, time_left_ms)`, returning
a move like `"e2e4"`. Three positions with one obvious answer each:

```
opening move                     -> e2e4
mate in 1: expect a1a8           -> a1a8
free queen: expect e5d5          -> e5d5
```

The mate line is worth pointing at. It finishes at depth 1 in 34 nodes,
because iterative deepening stops as soon as it has a mate it can reach.

## 6. Play it yourself (as long as you like)

```bash
python3 human.py          # you are White, engine gets 1s
python3 human.py black 3  # you are Black, engine gets 3s
```

Moves are typed in UCI: where the piece is now, then where it goes. `e2e4`,
`g8f6`, `a7a8q` to promote. Type `moves` for the legal list, `board` to
reprint, `quit` to stop.

```
your move > e2e4

engine plays g8f6   depth 4  score -35  24,576 nodes
```

Hang a piece and it takes it immediately. That is the quiescence search: at
depth zero it keeps playing out captures until the position is quiet, so it
never stops counting halfway through a trade.

## If you want to read the code

`board.py` first, then `nnue.py`, then `engine.py`, then `agent.py`. That is
dependency order and roughly difficulty order. `prepare.py` and `train.py`
stand on their own and can be read any time. Every file opens with a docstring
saying what the whole file is for.

Five places worth going straight to:

| | |
|---|---|
| `engine.py`, `_negamax` | the search. The whole tree is just this function calling itself. |
| `engine.py`, `_order_moves` | the single biggest lever on how deep it gets. |
| `nnue.py`, `feature_indices` | how a chess position becomes network input. |
| `board.py`, `push` | make the move, then check whether it was legal. |
| `prepare.py`, `parse_record` | which positions are thrown away, and why. |
