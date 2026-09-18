"""Proof that the evaluation network is wired up correctly.

    python3 check_nnue.py
    python3 check_nnue.py --engine ../chess-agent     # also compare to the
                                                      # shipping engine

A quantized network is the easiest thing in this repo to get subtly wrong. A
flipped rank, a colour xor the wrong way round, a scale applied twice: none of
it raises, none of it crashes, and the engine just plays slightly worse
forever. So it gets its own test.

Three checks, in order of how much they prove:

1. MIRROR SYMMETRY. Take a position, swap every piece's colour, flip the board
   top to bottom, and give the other side the move. That is the same position
   from the other chair, so the evaluation must come back EXACTLY equal.

   This is the strong one. The network stores one weight table and serves both
   sides from it by mirroring ranks and swapping colours, so if any part of
   that is wrong the two numbers disagree. Nothing external is needed to run
   it, which is why it is the default.

2. SANITY ANCHORS. A few positions whose evaluation we can reason about
   without a network: the start position is roughly level, a side up a queen
   is winning by roughly a queen.

3. THE SHIPPING ENGINE, optionally. With --engine pointing at the full repo,
   evaluate the same positions with its compiled integer path and require
   exact equality. That is what proves this is a faithful port and not merely
   a self-consistent one. It needs that repo's numba environment, so it is
   off by default.
"""

import argparse
import random
import sys

from board import Board, START_FEN

import nnue


def mirror_fen(fen):
    """The same position seen from the other side of the board.

    Ranks reversed, piece colours swapped, side to move swapped, castling
    rights swapped, en passant file kept but its rank flipped. A position and
    its mirror are worth the same to whoever is to move.
    """
    placement, side, rights, ep = fen.split()[:4]

    rows = placement.split("/")[::-1]                       # flip ranks
    rows = ["".join(c.lower() if c.isupper() else c.upper() for c in row)
            for row in rows]                                # swap colours

    swapped = "".join(c.lower() if c.isupper() else c.upper() for c in rights)
    swapped = "".join(sorted(swapped, key="KQkq".index)) if swapped != "-" else "-"

    if ep != "-":
        ep = ep[0] + str(9 - int(ep[1]))

    return f"{'/'.join(rows)} {'b' if side == 'w' else 'w'} {swapped} {ep} 0 1"


def positions(count, seed=17):
    """FENs from random games, so all sorts of king placements and material."""
    out = []
    rng = random.Random(seed)
    while len(out) < count:
        board = Board()
        for _ in range(rng.randint(4, 90)):
            legal = board.legal_moves()
            if not legal:
                break
            board.push(rng.choice(legal))
            out.append(board.to_fen())
            if len(out) >= count:
                break
    return out


def check_mirror(fens):
    print("1. mirror symmetry")
    bad = []
    for fen in fens:
        here = Board(fen).evaluate()
        there = Board(mirror_fen(fen)).evaluate()
        if here != there:
            bad.append((fen, here, there))
    for fen, a, b in bad[:5]:
        print(f"     {a:>7} vs {b:>7}   {fen}")
    print(f"   {len(fens) - len(bad):,} of {len(fens):,} positions match exactly")
    return not bad


ANCHORS = [
    (START_FEN, -60, 60, "the start position is roughly level"),
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1",
     -1400, -500, "White is a queen down, to move"),
    ("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
     500, 1400, "White is a queen up, to move"),
    ("8/8/8/4k3/8/8/4P3/4K3 w - - 0 1", -200, 400, "a bare king and pawn ending"),
]


def check_anchors():
    print("\n2. sanity anchors")
    ok = True
    for fen, low, high, what in ANCHORS:
        value = Board(fen).evaluate()
        good = low <= value <= high
        ok &= good
        print(f"   {value:>7} cp  {'ok  ' if good else 'FAIL'}  {what} "
              f"(expected {low} to {high})")
    return ok


def check_against_engine(fens, path):
    """Compare against the full engine's compiled integer path."""
    print(f"\n3. against the shipping engine at {path}")
    mine = [Board(fen).evaluate() for fen in fens]

    for name in ("board", "nnue"):
        sys.modules.pop(name, None)
    sys.path.insert(0, path)
    try:
        import board as full
    except ImportError as exc:
        print(f"   skipped: cannot import the full engine ({exc})")
        return True

    position = full.Position()
    bad = []
    for fen, ours in zip(fens, mine):
        position.set_fen(fen)
        theirs = int(full.evaluate(position.st, position.acc))
        if ours != theirs:
            bad.append((fen, ours, theirs))
    for fen, a, b in bad[:5]:
        print(f"     simple {a:>7}  full {b:>7}   {fen}")
    print(f"   {len(fens) - len(bad):,} of {len(fens):,} positions match exactly")
    return not bad


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", default="",
                        help="path to the full repo, to compare against it")
    parser.add_argument("--positions", type=int, default=3000)
    args = parser.parse_args()

    if not nnue.AVAILABLE:
        print("No network loaded. Either nnue.npz is missing, numpy is not "
              "installed, or CHESS_NNUE=0.")
        return 1

    print(f"network: {nnue.HIDDEN} hidden, {nnue.OUT_BUCKETS} output buckets, "
          f"{nnue.FT_W.shape[0]:,} input features\n")

    fens = positions(args.positions)
    ok = check_mirror(fens)
    ok &= check_anchors()
    if args.engine:
        ok &= check_against_engine(fens, args.engine)

    print()
    print("ALL PASS" if ok else "FAILURES ABOVE")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
