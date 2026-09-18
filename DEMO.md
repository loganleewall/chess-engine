# Demo

Four commands, about five minutes, no setup. Python 3.9 or newer, nothing to
install.

```bash
cd chess-agent-simple
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

ALL PASS   518,468 nodes in 2.1s (251,772 nps)
```

A single wrong number means a bug. This is the first thing to run after
touching `board.py`.

## 2. It plays itself (30 seconds)

```bash
python3 play.py           # 0.2s a move, 60 moves
python3 play.py 1.0 200   # slower and longer
```

Each line is one move: the depth it finished, the score in centipawns from
the mover's point of view, and how many positions it looked at.

```
  1. White b1c3   depth 4   score      0    14,336 nodes
  2. Black b8c6   depth 4   score    -40    16,384 nodes
  3. White g1f3   depth 4   score      0    16,384 nodes
```

Watch the depth move around. Give it more time and the depth goes up, which
is iterative deepening doing its job: it searches depth 1, then 2, then 3,
and keeps the last one it finished.

## 3. It answers the harness (10 seconds)

```bash
python3 agent.py
```

The competition API is one function, `get_move(fen, time_left_ms)`, returning
a move like `"e2e4"`. Three positions with one obvious answer each:

```
opening move                     -> b1c3
mate in 1: expect a1a8           -> a1a8
free queen: expect e5d5          -> e5d5
```

The mate line is worth pointing at. It finishes at depth 1 in 34 nodes,
because iterative deepening stops as soon as it has a mate it can actually
reach.

## 4. Play it yourself (as long as you like)

```bash
python3 human.py          # you are White, engine gets 1s
python3 human.py black 3  # you are Black, engine gets 3s
```

Moves are typed in UCI: where the piece is now, then where it goes.
`e2e4`, `g8f6`, `a7a8q` to promote. Type `moves` for the legal list, `board`
to reprint, `quit` to stop.

```
your move > e2e4

engine plays g8f6   depth 4  score -35  24,576 nodes
```

Hang a piece and it takes it immediately. That is the quiescence search: at
depth zero it keeps playing out captures until the position is quiet, so it
never stops counting halfway through a trade.

## If you want to read the code

`board.py` first, then `engine.py`, then `agent.py`. That is dependency
order and also difficulty order. Every file opens with a docstring saying
what the whole file is for.

Three places worth going straight to:

| | |
|---|---|
| `engine.py`, `_negamax` | the search. The whole tree is just this function calling itself. |
| `engine.py`, `_order_moves` | the single biggest lever on strength. |
| `board.py`, `push` | make the move, then check whether it was legal. |
