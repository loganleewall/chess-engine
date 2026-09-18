"""Turn the Lichess evaluation dump into shards the trainer can read.

    curl -s https://database.lichess.org/lichess_db_eval.jsonl.zst \\
      | zstd -dc | head -n 4000000 | python3 prepare.py - data/shard

    python3 prepare.py raw.jsonl data/shard --shard 2000000

The input is one JSON record per line, from database.lichess.org (CC0):

    {"fen": "...", "evals": [{"pvs": [{"cp": 31, "line": "e2e4 ..."}],
                              "depth": 24, "knodes": 1000}]}

395 million positions that somebody already paid the compute to evaluate
deeply. That is the whole reason this project could train a network at all:
the labels were free, and generating them from scratch would have taken
months of CPU.

FEN is parsed with our own `board.py`, so this needs nothing beyond numpy.


WHY IT THROWS MOST POSITIONS AWAY
---------------------------------
Three filters, and the second and third are the interesting ones.

    depth < 10              the label is not trustworthy enough to learn from
    the position is in check    the search resolves these itself
    the best move is a capture  so does it

Dropping every noisy position looks like throwing away data and is the
opposite. The engine's quiescence search plays out all the captures before it
ever asks the network for a number, so the network is only ever called on
quiet positions. Training it on the others teaches it to do a job the search
already does better, and spends its capacity in the wrong place.

This is the single decision that most shapes what the network becomes, and it
is the thing to point at when someone asks how the search and the evaluation
divide the work between them.


THE OUTPUT
----------
Shards of `--shard` positions each, shuffled inside the shard, as
`DEST_000.npz`, `DEST_001.npz`, and so on:

    piece   int8  (n, 32)   colour * 6 + kind, so 0..11, and -1 for padding
    square  int8  (n, 32)   a1 = 0 .. h8 = 63
    stm     int8  (n,)      0 white to move, 1 black
    cp      int16 (n,)      centipawns, from WHITE's point of view

Note what is NOT stored: the network's input indices. Those depend on where
each side's king is, so they are derived at training time by `train.py` and at
play time by `nnue.py`. Storing raw piece lists instead means the king zone
scheme can change without reprocessing 21 GB.

Shuffling within the shard matters. Consecutive lines in the dump come from
the same game, so an unshuffled batch would be 4,096 positions from a handful
of games and the gradient would be badly correlated.
"""

import argparse
import json
import sys

import numpy as np

from board import (BLACK, CASTLE_BK, CASTLE_BQ, CASTLE_WK, CASTLE_WQ, KING,
                   ROOK, WHITE, Board, piece_index)

CP_CLAMP = 2000          # mates and blowouts land here; beyond it is noise
MIN_DEPTH = 10


def parse_record(line):
    """One dump line -> (piece list, square list, side to move, centipawns).

    Returns None for anything filtered out, which is most lines.
    """
    try:
        record = json.loads(line)
    except ValueError:
        return None

    fen = record.get("fen")
    evaluations = record.get("evals") or []
    if not fen or not evaluations:
        return None

    # Keep the deepest evaluation available for this position.
    best = max(evaluations, key=lambda e: e.get("depth", 0))
    if best.get("depth", 0) < MIN_DEPTH:
        return None

    variations = best.get("pvs") or []
    if not variations:
        return None
    principal = variations[0]

    if "cp" in principal:
        centipawns = max(-CP_CLAMP, min(CP_CLAMP, int(principal["cp"])))
    elif "mate" in principal:
        centipawns = CP_CLAMP if int(principal["mate"]) > 0 else -CP_CLAMP
    else:
        return None

    board = Board(fen if len(fen.split()) >= 4 else fen + " - 0 1")

    if board.in_check() or not _is_reachable(board):
        return None

    # Is the best move a capture? The principal variation's first move tells
    # us, and we have the board in front of us to ask.
    line_moves = principal.get("line", "").split()
    if line_moves and _is_capture_or_promotion(board, line_moves[0]):
        return None

    pieces, squares = [], []
    for piece in range(12):
        bitboard = board.pieces[piece]
        while bitboard:
            square = (bitboard & -bitboard).bit_length() - 1
            bitboard &= bitboard - 1
            pieces.append(piece)
            squares.append(square)
    if not 2 <= len(pieces) <= 32:
        return None

    # The dump's scores are from White's point of view; keep them that way and
    # let the trainer flip the sign, so the stored data has one convention.
    return pieces, squares, board.side, centipawns


def _is_reachable(board):
    """Could this position actually occur in a game?

    The dump is scraped from real play but a few records are malformed, and a
    position that could never arise teaches the network nothing. Two cheap
    tests catch almost all of them:

      - The side NOT to move must not be in check. If it were, the side to
        move would already have captured the king.
      - Castling rights must match where the kings and rooks actually stand.
        A FEN claiming White can castle kingside with no rook on h1 is
        inconsistent, and usually means the record is damaged.
    """
    if board.in_check(1 - board.side):
        return False

    for right, king_square, rook_square, colour in (
            (CASTLE_WK, 4, 7, WHITE), (CASTLE_WQ, 4, 0, WHITE),
            (CASTLE_BK, 60, 63, BLACK), (CASTLE_BQ, 60, 56, BLACK)):
        if not board.castling & right:
            continue
        king = board.pieces[piece_index(colour, KING)]
        rook = board.pieces[piece_index(colour, ROOK)]
        if not (king >> king_square & 1) or not (rook >> rook_square & 1):
            return False
    return True


def _is_capture_or_promotion(board, uci):
    """Does this UCI move take something, or promote?"""
    if len(uci) < 4:
        return False
    if len(uci) > 4:                              # a promotion letter
        return True
    to_square = (ord(uci[2]) - ord("a")) + (int(uci[3]) - 1) * 8
    if board.occupied[1 - board.side] >> to_square & 1:
        return True
    return to_square == board.ep_square and _is_pawn_move(board, uci)


def _is_pawn_move(board, uci):
    from_square = (ord(uci[0]) - ord("a")) + (int(uci[1]) - 1) * 8
    return bool(board.pieces[board.side * 6] >> from_square & 1)


def write_shard(rows, destination, index, rng):
    """One .npz, shuffled, in the layout train.py's Shard expects."""
    count = len(rows)
    piece = np.full((count, 32), -1, dtype=np.int8)
    square = np.zeros((count, 32), dtype=np.int8)
    stm = np.zeros(count, dtype=np.int8)
    cp = np.zeros(count, dtype=np.int16)

    for row, (pieces, squares, side, centipawns) in enumerate(rows):
        piece[row, :len(pieces)] = pieces
        square[row, :len(squares)] = squares
        stm[row] = side
        cp[row] = centipawns

    order = rng.permutation(count)          # break up runs from the same game
    path = f"{destination}_{index:03d}.npz"
    np.savez_compressed(path, piece=piece[order], square=square[order],
                        stm=stm[order], cp=cp[order])
    print(f"wrote {path}  {count:,} positions", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Shard the Lichess eval dump.")
    parser.add_argument("source", help="a .jsonl file, or - for standard input")
    parser.add_argument("destination", help="prefix, e.g. data/shard")
    parser.add_argument("--shard", type=int, default=1_000_000,
                        help="positions per shard")
    args = parser.parse_args()

    stream = sys.stdin if args.source == "-" else open(args.source)
    rng = np.random.default_rng(0)

    rows, shard_index, read, kept = [], 0, 0, 0
    for line in stream:
        read += 1
        parsed = parse_record(line)
        if parsed is None:
            continue
        rows.append(parsed)
        kept += 1
        if len(rows) >= args.shard:
            write_shard(rows, args.destination, shard_index, rng)
            rows, shard_index = [], shard_index + 1
        if read % 500_000 == 0:
            print(f"  read {read:,}, kept {kept:,} ({kept / read:.0%})", flush=True)

    if rows:
        write_shard(rows, args.destination, shard_index, rng)

    if stream is not sys.stdin:
        stream.close()
    print(f"\nread {read:,} records, kept {kept:,} "
          f"({kept / max(read, 1):.0%}) in {shard_index + 1} shard(s)")


if __name__ == "__main__":
    main()
