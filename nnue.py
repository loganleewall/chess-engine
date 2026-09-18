"""The evaluation network: how good is this position, in centipawns?

The weights are in `nnue.npz`, which `train.py` produces.


WHY A NETWORK AT ALL
--------------------
The search decides which move to play by comparing positions at the bottom of
the tree, so it needs a number for each one. `board.evaluate()` gives one by
counting material and adding a per-square bonus. That knows a knight belongs
in the centre and a king belongs behind pawns. It does not know anything else,
because it scores every piece alone: a knight on f3 is worth the same whether
our king is safe or about to be mated.

So the number comes from a small trained network instead. The first one used
plain piece-square inputs and already beat the table, +15 =21 -4. Its rated
games then showed what it still could not see: it gave up material for an
attack that was not there, and let a winning rook endgame fizzle into a draw.
King safety and endgame scale. The king zones and the eight output heads
below are the fix for exactly those two.


THE SHAPE
---------
    (16 x 768 -> 256) x 2 -> 8 x 1

    768   2 colours x 6 piece types x 64 squares. One input per
          "there is a white knight on f3" fact.
    x 16  the king zone. Which of 16 zones YOUR OWN king stands in shifts
          every one of your input indices, so a knight on f5 is a different
          input when your king is castled short than when it is on e1.
          This is what makes king safety easy to learn: "knight near my
          king" becomes a single input instead of something the net has to
          assemble from pairs.
    x 2   two perspectives. Each side sees the board from its own point of
          view, with colours swapped and ranks mirrored. Both share one
          weight table.
    8 x 1 eight output heads, picked by how many pieces are left, so an
          endgame gets its own scale.

The input is not an image and there is no convolution. It is an EMBEDDING
LOOKUP: each piece on each square is one row of a weight table, and the
position is the sum of its pieces' rows. 32 rows added, out of 12,288.


INTEGERS, NOT FLOATS
--------------------
Everything here is integer arithmetic. The trainer works in float and the
export rounds the weights to int16, because integer adds are what a CPU
vectorises, and the search calls this on every position it scores.
That means two implementations of the same network now exist, and a
quantization bug does not crash, it just makes the engine slightly worse.
`train.py` compares the two after every export for exactly that reason.

    QA = 255    scale of the input weights and the accumulator
    QB = 64     scale of the output weights
    SCALE = 400 centipawns per unit of network output


THE ACCUMULATOR IS REBUILT EVERY TIME
-------------------------------------
This file sums up to 32 rows per side on every evaluation. A faster design
never rebuilds it: a move touches at most two pieces, so `push` would add two
weight rows and subtract two, `pop` would restore a saved copy, and only a
king move would rebuild that side from scratch, because every index changed.
Four rows per move instead of up to 32 per evaluation, so up to eight times
less arithmetic, for about 60 more lines of delicate bookkeeping in `board.py`.
"""

import os

import numpy as np

# The weights live next to this file. Delete nnue.npz, or set CHESS_NNUE=0,
# and board.evaluate() falls back to material plus piece-square tables with no
# other change.
_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nnue.npz")

AVAILABLE = os.path.exists(_PATH) and os.environ.get("CHESS_NNUE", "1") != "0"

QA, QB, SCALE = 255, 64, 400

if AVAILABLE:
    _NN = np.load(_PATH)
    HIDDEN = int(_NN["hidden"])              # 256
    OUT_BUCKETS = int(_NN["out_buckets"])    # 8
    FT_W = _NN["ft_w"].astype(np.int16)      # (16*768, 256) input weights
    FT_B = _NN["ft_b"].astype(np.int32)      # (256,)        input bias
    OUT_W = _NN["out_w"].astype(np.int64)    # (8, 512)      output weights
    OUT_B = _NN["out_b"].astype(np.int64)    # (8,)          output bias
    KING_ZONE = _NN["king_bucket"].astype(np.int64)   # (64,) square -> 0..15


def feature_indices(board, perspective):
    """Every input index that is set, from one side's point of view.

    A position has at most 32 pieces, so this returns at most 32 numbers out
    of a possible 12,288. That sparsity is the reason the first layer can be
    a lookup rather than a matrix multiply.

    Three things happen to a piece to get its index:

      1. `sq ^ flip` mirrors the ranks for black, so "my second rank" is the
         same input for both sides.
      2. `colour ^ perspective` swaps the colours, so "my knight" is the same
         input for both sides.
      3. `zone * 768` shifts the whole block by where my own king is.

    Steps 1 and 2 are what lets one weight table serve both sides. Step 3 is
    what lets the net know anything about king safety.
    """
    flip = 56 * perspective          # xor with 56 mirrors a square's rank
    king_square = _lowest_bit(board.pieces[perspective * 6 + 5])   # 5 = KING
    zone = KING_ZONE[king_square ^ flip]

    indices = []
    for piece in range(12):
        colour, kind = divmod(piece, 6)
        bitboard = board.pieces[piece]
        while bitboard:
            square = (bitboard & -bitboard).bit_length() - 1
            bitboard &= bitboard - 1
            indices.append(zone * 768
                           + (colour ^ perspective) * 384
                           + kind * 64
                           + (square ^ flip))
    return indices


def accumulator(board, perspective):
    """The hidden layer before the nonlinearity, for one perspective.

    This is the embedding sum: take the rows of the weight table named by the
    features, add them up, add the bias. Summed in int32 because a handful of
    int16 rows added together is the one place this could overflow.
    """
    rows = FT_W[feature_indices(board, perspective)]
    return rows.sum(axis=0, dtype=np.int32) + FT_B


def evaluate(board):
    """Centipawns, from the point of view of the side to move.

    The output layer, in three steps:

      1. SCReLU. Clamp each accumulator value to [0, 255] and square it.
         This is the network's only nonlinearity. The clamp is a clipped
         ReLU, which on its own already makes the network nonlinear. The
         square on top multiplies features together, so pairs of pieces can
         interact, and it is cheap: one multiply.

      2. Dot product with the output row for this position's piece count.
         The side to move's accumulator goes in the first half of the row and
         the opponent's in the second, so the same weights mean "mine" and
         "theirs" rather than "white's" and "black's".

      3. Undo the quantization scales to get centipawns.
    """
    us = board.side
    them = 1 - us

    # Which output head. More pieces on the board means a more middlegame-like
    # scale; the last bucket is for bare endgames.
    pieces_left = sum(bin(bb).count("1") for bb in board.pieces)
    bucket = min(max((pieces_left - 2) >> 2, 0), OUT_BUCKETS - 1)
    weights = OUT_W[bucket]

    mine = np.clip(accumulator(board, us), 0, QA).astype(np.int64)
    theirs = np.clip(accumulator(board, them), 0, QA).astype(np.int64)

    total = int((mine * mine * weights[:HIDDEN]).sum()
                + (theirs * theirs * weights[HIDDEN:]).sum())

    return int((total // QA + OUT_B[bucket]) * SCALE // (QA * QB))


def _lowest_bit(bitboard):
    """Index of the lowest set bit. Same helper as board.lowest_square."""
    return (bitboard & -bitboard).bit_length() - 1
