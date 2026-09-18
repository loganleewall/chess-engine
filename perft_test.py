"""Proof that the simplified engine actually obeys the rules of chess.

`perft(n)` counts the legal move sequences of length n. The counts below are
published and universally agreed on, so if our generator produces exactly
these numbers, it handles every rule correctly -- castling through check, en
passant, promotion, pins, the lot. If a single number is off, there is a bug.

This is the same idea as perft_test.py in the full repo, on the same positions.
Run it after ANY change to board.py:

    python3 perft_test.py
"""

import time

from board import Board, START_FEN


def perft(board, depth):
    """Count legal move sequences of exactly `depth` plies.

    Note there is no tree here either: one board, mutated and restored. The
    recursion is the tree.
    """
    if depth == 0:
        return 1
    total = 0
    for move in board.generate_moves():
        if board.push(move):              # skips moves that leave us in check
            total += 1 if depth == 1 else perft(board, depth - 1)
            board.pop()
    return total


# (name, fen, [expected perft(1), perft(2), ...])
# Positions 2-6 are the standard "tricky" set: they exist specifically to
# catch en passant, promotion and castling bugs.
SUITE = [
    ("start position", START_FEN, [20, 400, 8902, 197281]),
    ("kiwipete",
     "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
     [48, 2039, 97862]),
    ("endgame", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
     [14, 191, 2812, 43238]),
    ("promotion / castling",
     "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
     [6, 264, 9467]),
    ("position 5",
     "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
     [44, 1486, 62379]),
    ("position 6",
     "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
     [46, 2079, 89890]),
]


def main():
    all_ok = True
    total_nodes = 0
    start = time.monotonic()

    for name, fen, expected in SUITE:
        board = Board(fen)
        print(f"{name}")
        for depth, want in enumerate(expected, start=1):
            t0 = time.monotonic()
            got = perft(board, depth)
            elapsed = time.monotonic() - t0
            total_nodes += got
            ok = got == want
            all_ok &= ok
            note = "ok" if ok else f"FAIL, expected {want:,}"
            rate = f"{got / elapsed:>10,.0f} nps" if elapsed > 0.05 else ""
            print(f"   depth {depth}  {got:>10,}  {note:<24} {rate}")
        print()

    elapsed = time.monotonic() - start
    verdict = "ALL PASS" if all_ok else "FAILURES ABOVE"
    print(f"{verdict}   {total_nodes:,} nodes in {elapsed:.1f}s "
          f"({total_nodes / elapsed:,.0f} nps)")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
