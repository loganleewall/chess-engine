"""The rules of chess: what a position is, what moves exist, how to play one.

Read this file first; it is the foundation everything else stands on.

THE THREE IDEAS IN THIS FILE
----------------------------
1. A position is 12 integers (bitboards). Bit n of `pieces[p]` is set if a
   piece of kind p stands on square n.
2. Move generation is "pseudo-legal": we generate moves that look right for
   the piece, ignoring whether they leave our own king in check. `push()`
   catches that by playing the move and looking.
3. The Zobrist key is a 64-bit fingerprint of the position, updated by XOR as
   pieces move, so we can recognise a repeated position instantly.

SQUARE NUMBERING: 0 = a1, 1 = b1, ... 7 = h1, 8 = a2, ... 63 = h8.
So `sq // 8` is the rank (0-7) and `sq % 8` is the file (0-7).
"""

import random

# The trained evaluation network. Guarded because it is the one thing here
# that needs numpy: without it, `evaluate` uses the piece-square tables and
# everything else is unchanged.
try:
    import nnue
except ImportError:
    nnue = None

# ---------------------------------------------------------------------------
# Naming things
# ---------------------------------------------------------------------------

WHITE, BLACK = 0, 1

# Piece kinds. These are indices into the `pieces` list below.
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 0, 1, 2, 3, 4, 5

# A piece is identified by `colour * 6 + kind`, so:
#   0..5  = white pawn, knight, bishop, rook, queen, king
#   6..11 = black pawn, knight, bishop, rook, queen, king
# `piece_index(BLACK, BISHOP)` is 1*6 + 2 = 8.
def piece_index(colour, kind):
    return colour * 6 + kind


# Castling rights are four bits packed into one integer.
CASTLE_WK, CASTLE_WQ, CASTLE_BK, CASTLE_BQ = 1, 2, 4, 8

# Move flags. A move needs to say if it is "special" because these three cases
# move or remove a piece that is not on the `to` square.
QUIET, DOUBLE_PAWN_PUSH, EN_PASSANT, CASTLE = 0, 1, 2, 3

# Rough piece values in centipawns (100 = one pawn). The king is never
# captured so its value only matters for move ordering.
PIECE_VALUE = (100, 320, 330, 500, 900, 20000)

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


# ---------------------------------------------------------------------------
# Bitboard helpers
#
# A bitboard is just a Python int used as a set of squares. Because Python
# ints are arbitrary precision we never have to think about overflow.
# ---------------------------------------------------------------------------

def lowest_square(bitboard):
    """Index of the lowest set bit.

    `b & -b` isolates the lowest set bit (two's complement trick), and
    `.bit_length() - 1` turns that single bit into its position.
    """
    return (bitboard & -bitboard).bit_length() - 1


def squares_of(bitboard):
    """Yield every square in the bitboard, lowest first.

    `b &= b - 1` clears the lowest set bit. This two-line loop is how you
    iterate over pieces, and you will see it everywhere.
    """
    while bitboard:
        yield lowest_square(bitboard)
        bitboard &= bitboard - 1


def square_name(sq):
    """0 -> 'a1', 63 -> 'h8'."""
    return "abcdefgh"[sq % 8] + str(sq // 8 + 1)


# ---------------------------------------------------------------------------
# Attack tables, computed once at import
#
# For knights, kings and pawns, where a piece attacks depends only on where
# it stands. So we precompute a bitboard per square, in a few loops. The
# sliding pieces are handled separately below, by walking rays.
# ---------------------------------------------------------------------------

def _offset_table(offsets):
    """Attack bitboard per square, for a piece that jumps by fixed offsets."""
    table = []
    for sq in range(64):
        file, rank = sq % 8, sq // 8
        attacks = 0
        for df, dr in offsets:
            f, r = file + df, rank + dr
            if 0 <= f < 8 and 0 <= r < 8:          # stayed on the board
                attacks |= 1 << (r * 8 + f)
        table.append(attacks)
    return table


KNIGHT_ATTACKS = _offset_table([(1, 2), (2, 1), (2, -1), (1, -2),
                                (-1, -2), (-2, -1), (-2, 1), (-1, 2)])

KING_ATTACKS = _offset_table([(1, 0), (1, 1), (0, 1), (-1, 1),
                              (-1, 0), (-1, -1), (0, -1), (1, -1)])

# Pawns are the one piece whose attacks depend on colour: white captures
# upward, black downward. PAWN_ATTACKS[colour][square].
PAWN_ATTACKS = [_offset_table([(-1, 1), (1, 1)]),      # white
                _offset_table([(-1, -1), (1, -1)])]    # black

ROOK_DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))
BISHOP_DIRECTIONS = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def sliding_attacks(sq, occupied, directions):
    """Where a rook/bishop/queen on `sq` attacks, given what is in the way.

    Sliding pieces are the hard case: a rook on a1 attacks a2 only if nothing
    is on it. So we walk outward in each direction and stop at the first
    occupied square -- including that square, because we might capture it.

    The fast alternative is "magic bitboards": ONE multiply and ONE array
    lookup, at the cost of ~150 lines of table generation. Walking the rays
    is maybe 20x slower and obviously correct.
    """
    attacks = 0
    file, rank = sq % 8, sq // 8
    for df, dr in directions:
        f, r = file + df, rank + dr
        while 0 <= f < 8 and 0 <= r < 8:
            target = r * 8 + f
            attacks |= 1 << target
            if occupied >> target & 1:      # blocked -- stop after including it
                break
            f += df
            r += dr
    return attacks


# ---------------------------------------------------------------------------
# Zobrist hashing
#
# Give every (piece, square) pair its own random 64-bit number. A position's
# key is all the applicable numbers XORed together. XOR is its own inverse,
# so moving a piece is two XORs: one to take it off the old square, one to
# put it on the new one. That is what makes the key cheap to maintain.
#
# A fixed seed means the numbers are the same every run, which makes bugs
# reproducible.
# ---------------------------------------------------------------------------

_rng = random.Random(0xC0FFEE)
ZOBRIST_PIECE = [[_rng.getrandbits(64) for _ in range(64)] for _ in range(12)]
ZOBRIST_CASTLING = [_rng.getrandbits(64) for _ in range(16)]
ZOBRIST_EP_FILE = [_rng.getrandbits(64) for _ in range(8)]
ZOBRIST_BLACK_TO_MOVE = _rng.getrandbits(64)


# ---------------------------------------------------------------------------
# Evaluation tables
#
# "Piece-square tables": what a piece is worth depends on where it stands. A
# knight in the centre is worth more than one in a corner. Written from
# white's point of view, rank 8 first, so they read like a chessboard.
#
# The trained network in nnue.py replaces these whenever it can load. The
# net is strictly better, but this is the same *shape* of idea: a number per
# piece per square, summed over the board.
# ---------------------------------------------------------------------------

_PAWN_TABLE = (
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0)

_KNIGHT_TABLE = (
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50)

_BISHOP_TABLE = (
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20)

_ROOK_TABLE = (
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0)

_QUEEN_TABLE = (
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20)

# The king wants opposite things in the opening and the endgame: hide behind
# pawns early, march to the centre once the queens are gone. So it gets two
# tables and `evaluate()` picks one by how much material is left.
_KING_MIDGAME_TABLE = (
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20)

_KING_ENDGAME_TABLE = (
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50)


def _build_score_tables():
    """Combine material value + square bonus into one lookup per piece.

    The tables above are written with rank 8 on the first line, but square 0
    is a1, so we flip with `sq ^ 56` to read them from white's point of view.
    Black's tables are the negation of white's, read from the unflipped
    square -- that is the mirroring.

    Returns two lists of 12 lists of 64 ints: one for the midgame king, one
    for the endgame king. Everything else is identical between the two.
    """
    kinds = (_PAWN_TABLE, _KNIGHT_TABLE, _BISHOP_TABLE, _ROOK_TABLE,
             _QUEEN_TABLE)
    midgame = [[0] * 64 for _ in range(12)]
    endgame = [[0] * 64 for _ in range(12)]
    for kind, table in enumerate(kinds):
        for sq in range(64):
            white_score = PIECE_VALUE[kind] + table[sq ^ 56]
            black_score = PIECE_VALUE[kind] + table[sq]
            midgame[piece_index(WHITE, kind)][sq] = white_score
            endgame[piece_index(WHITE, kind)][sq] = white_score
            midgame[piece_index(BLACK, kind)][sq] = -black_score
            endgame[piece_index(BLACK, kind)][sq] = -black_score
    for sq in range(64):
        midgame[piece_index(WHITE, KING)][sq] = _KING_MIDGAME_TABLE[sq ^ 56]
        endgame[piece_index(WHITE, KING)][sq] = _KING_ENDGAME_TABLE[sq ^ 56]
        midgame[piece_index(BLACK, KING)][sq] = -_KING_MIDGAME_TABLE[sq]
        endgame[piece_index(BLACK, KING)][sq] = -_KING_ENDGAME_TABLE[sq]
    return midgame, endgame


MIDGAME_SCORE, ENDGAME_SCORE = _build_score_tables()


# Moving from or to one of these squares gives up castling rights: the king
# leaving e1, or either white rook leaving (or being captured on) a1 / h1.
# `rights &= CASTLING_MASK[frm] & CASTLING_MASK[to]` handles all six cases,
# including "my rook got captured on h8", without any special-casing.
CASTLING_MASK = [15] * 64
CASTLING_MASK[0] = 15 ^ CASTLE_WQ                    # a1 rook
CASTLING_MASK[4] = 15 ^ (CASTLE_WK | CASTLE_WQ)      # e1 king
CASTLING_MASK[7] = 15 ^ CASTLE_WK                    # h1 rook
CASTLING_MASK[56] = 15 ^ CASTLE_BQ                   # a8 rook
CASTLING_MASK[60] = 15 ^ (CASTLE_BK | CASTLE_BQ)     # e8 king
CASTLING_MASK[63] = 15 ^ CASTLE_BK                   # h8 rook


# ---------------------------------------------------------------------------
# Moves
#
# A move is a 4-tuple: (from, to, promotion, flag).
# ---------------------------------------------------------------------------

def make_move(frm, to, promotion=0, flag=QUIET):
    """`promotion` is 0 for none, else KNIGHT/BISHOP/ROOK/QUEEN."""
    return (frm, to, promotion, flag)


def move_to_uci(move):
    """The string format the competition harness wants: 'e2e4', 'a7a8q'."""
    frm, to, promotion, _ = move
    text = square_name(frm) + square_name(to)
    if promotion:
        text += " nbrq"[promotion]
    return text


# ---------------------------------------------------------------------------
# The board
# ---------------------------------------------------------------------------

class Board:
    """A chess position, plus the stack needed to undo moves."""

    def __init__(self, fen=START_FEN):
        self.pieces = [0] * 12      # bitboard per piece, index colour*6+kind
        self.occupied = [0, 0]      # all white pieces, all black pieces
        self.side = WHITE           # whose turn
        self.castling = 0           # the four CASTLE_* bits
        self.ep_square = -1         # en passant target, or -1
        self.halfmove_clock = 0     # moves since a capture or pawn move
        self.key = 0                # Zobrist fingerprint
        self.undo_stack = []        # what pop() needs to restore
        self.set_fen(fen)

    # -- reading the position ------------------------------------------------

    @property
    def all_occupied(self):
        return self.occupied[WHITE] | self.occupied[BLACK]

    def piece_on(self, sq, colour):
        """Which kind of `colour` piece is on `sq`, or -1 if none."""
        mask = 1 << sq
        for kind in range(6):
            if self.pieces[piece_index(colour, kind)] & mask:
                return kind
        return -1

    def king_square(self, colour):
        return lowest_square(self.pieces[piece_index(colour, KING)])

    def is_attacked(self, sq, by_colour):
        """Could `by_colour` capture something standing on `sq`?

        Note the trick on the pawn line: instead of asking "which squares do
        their pawns attack", we ask "if a pawn of OUR colour stood here,
        which squares would it attack" and check for their pawns there.
        Attacks are symmetric, so that is the same question backwards, and it
        saves generating every pawn attack.
        """
        base = by_colour * 6
        occupied = self.all_occupied

        if PAWN_ATTACKS[1 - by_colour][sq] & self.pieces[base + PAWN]:
            return True
        if KNIGHT_ATTACKS[sq] & self.pieces[base + KNIGHT]:
            return True
        if KING_ATTACKS[sq] & self.pieces[base + KING]:
            return True
        rooks_queens = self.pieces[base + ROOK] | self.pieces[base + QUEEN]
        if sliding_attacks(sq, occupied, ROOK_DIRECTIONS) & rooks_queens:
            return True
        bishops_queens = self.pieces[base + BISHOP] | self.pieces[base + QUEEN]
        if sliding_attacks(sq, occupied, BISHOP_DIRECTIONS) & bishops_queens:
            return True
        return False

    def in_check(self, colour=None):
        if colour is None:
            colour = self.side
        return self.is_attacked(self.king_square(colour), 1 - colour)

    # -- changing the position ----------------------------------------------

    def _add(self, piece, sq):
        """Put `piece` on `sq`, updating the bitboards and the Zobrist key."""
        bit = 1 << sq
        self.pieces[piece] |= bit
        self.occupied[piece // 6] |= bit
        self.key ^= ZOBRIST_PIECE[piece][sq]

    def _remove(self, piece, sq):
        """Take `piece` off `sq`. Identical XOR, because XOR undoes itself."""
        bit = 1 << sq
        self.pieces[piece] &= ~bit
        self.occupied[piece // 6] &= ~bit
        self.key ^= ZOBRIST_PIECE[piece][sq]

    def push(self, move):
        """Play `move`. Returns False, having undone it, if it was illegal.

        This is the "pseudo-legal then verify" design. `generate_moves` does
        not check whether a move leaves our own king in check -- pins and
        check evasions are never special-cased anywhere. Instead we play the
        move, look at whether our king is attacked, and take it back if so.

        A legal-move generator would be faster. This is ~15 lines instead of
        ~150 and it cannot be subtly wrong.
        """
        frm, to, promotion, flag = move
        us, them = self.side, 1 - self.side
        base, enemy_base = us * 6, them * 6

        moved = self.piece_on(frm, us)
        if flag == EN_PASSANT:
            captured = PAWN            # the victim is not on `to`
        else:
            captured = self.piece_on(to, them)

        # Everything pop() cannot recompute.
        self.undo_stack.append((move, moved, captured, self.castling,
                                self.ep_square, self.halfmove_clock, self.key))

        # The old en passant file is part of the key; clear it before we set
        # the new one.
        if self.ep_square >= 0:
            self.key ^= ZOBRIST_EP_FILE[self.ep_square % 8]

        if captured >= 0:
            victim_square = to
            if flag == EN_PASSANT:
                # The captured pawn is beside our destination, not on it.
                victim_square = to - 8 if us == WHITE else to + 8
            self._remove(enemy_base + captured, victim_square)

        self._remove(base + moved, frm)
        if promotion:
            self._add(base + promotion, to)
        else:
            self._add(base + moved, to)

        if flag == CASTLE:
            # The king has already moved; now move the rook over it. `to`
            # identifies which of the four castles this is.
            rook_from, rook_to = {6: (7, 5), 2: (0, 3),
                                  62: (63, 61), 58: (56, 59)}[to]
            self._remove(base + ROOK, rook_from)
            self._add(base + ROOK, rook_to)

        # Castling rights: see CASTLING_MASK above.
        self.key ^= ZOBRIST_CASTLING[self.castling]
        self.castling &= CASTLING_MASK[frm] & CASTLING_MASK[to]
        self.key ^= ZOBRIST_CASTLING[self.castling]

        # Only a double pawn push creates an en passant target.
        if flag == DOUBLE_PAWN_PUSH:
            self.ep_square = (frm + to) // 2
            self.key ^= ZOBRIST_EP_FILE[self.ep_square % 8]
        else:
            self.ep_square = -1

        # The fifty-move rule counts moves since the last "irreversible" one.
        if moved == PAWN or captured >= 0:
            self.halfmove_clock = 0
        else:
            self.halfmove_clock += 1

        self.side = them
        self.key ^= ZOBRIST_BLACK_TO_MOVE

        # The legality test, and the only reason this returns a bool.
        if self.is_attacked(self.king_square(us), them):
            self.pop()
            return False
        return True

    def pop(self):
        """Undo the last push. Exactly reverses it, in reverse order."""
        move, moved, captured, castling, ep_square, clock, key = \
            self.undo_stack.pop()
        frm, to, promotion, flag = move

        self.side = 1 - self.side          # back to the mover
        us, them = self.side, 1 - self.side
        base, enemy_base = us * 6, them * 6

        if flag == CASTLE:
            rook_from, rook_to = {6: (7, 5), 2: (0, 3),
                                  62: (63, 61), 58: (56, 59)}[to]
            self._remove(base + ROOK, rook_to)
            self._add(base + ROOK, rook_from)

        if promotion:
            self._remove(base + promotion, to)
        else:
            self._remove(base + moved, to)
        self._add(base + moved, frm)

        if captured >= 0:
            victim_square = to
            if flag == EN_PASSANT:
                victim_square = to - 8 if us == WHITE else to + 8
            self._add(enemy_base + captured, victim_square)

        # The key was maintained by the _add/_remove calls above, but we have
        # the exact old value saved, so just restore it. Cheaper, and it
        # cannot drift over millions of moves.
        self.castling = castling
        self.ep_square = ep_square
        self.halfmove_clock = clock
        self.key = key

    # -- move generation ----------------------------------------------------

    def generate_moves(self):
        """Every pseudo-legal move for the side to move.

        "Pseudo-legal" = correct for the piece, but might leave our king in
        check. `push()` filters those out. Castling is the one exception: it
        is generated fully legal, because "may not castle through check" is
        cheapest to check right here.
        """
        moves = []
        us, them = self.side, 1 - self.side
        base = us * 6
        own = self.occupied[us]
        enemy = self.occupied[them]
        occupied = own | enemy

        self._generate_pawn_moves(moves, us, them, enemy, occupied)

        # Knights, bishops, rooks, queens, king: find the target squares, drop
        # the ones with our own pieces on them, emit the rest.
        for kind in (KNIGHT, BISHOP, ROOK, QUEEN, KING):
            for frm in squares_of(self.pieces[base + kind]):
                if kind == KNIGHT:
                    targets = KNIGHT_ATTACKS[frm]
                elif kind == KING:
                    targets = KING_ATTACKS[frm]
                elif kind == BISHOP:
                    targets = sliding_attacks(frm, occupied, BISHOP_DIRECTIONS)
                elif kind == ROOK:
                    targets = sliding_attacks(frm, occupied, ROOK_DIRECTIONS)
                else:
                    targets = (sliding_attacks(frm, occupied, ROOK_DIRECTIONS)
                               | sliding_attacks(frm, occupied,
                                                 BISHOP_DIRECTIONS))
                for to in squares_of(targets & ~own):
                    moves.append(make_move(frm, to))

        self._generate_castling(moves, us, them, occupied)
        return moves

    def _generate_pawn_moves(self, moves, us, them, enemy, occupied):
        """Pawns: the fiddliest piece. Four ways to move, plus promotion."""
        pawns = self.pieces[us * 6 + PAWN]
        if us == WHITE:
            forward = 8                 # one rank up
            start_rank, promo_rank = 1, 7
        else:
            forward = -8
            start_rank, promo_rank = 6, 0

        for frm in squares_of(pawns):
            rank = frm // 8

            # 1. one square forward, if empty
            one = frm + forward
            if not occupied >> one & 1:
                if one // 8 == promo_rank:
                    # Promotion is four moves, not one. Queen first because
                    # it is nearly always best, and move ordering matters.
                    for promotion in (QUEEN, ROOK, BISHOP, KNIGHT):
                        moves.append(make_move(frm, one, promotion))
                else:
                    moves.append(make_move(frm, one))

                # 2. two squares, only from the starting rank and only if
                #    BOTH squares are empty (hence the nesting)
                if rank == start_rank:
                    two = one + forward
                    if not occupied >> two & 1:
                        moves.append(make_move(frm, two, 0,
                                               DOUBLE_PAWN_PUSH))

            # 3. captures, including promotion-by-capture
            for to in squares_of(PAWN_ATTACKS[us][frm] & enemy):
                if to // 8 == promo_rank:
                    for promotion in (QUEEN, ROOK, BISHOP, KNIGHT):
                        moves.append(make_move(frm, to, promotion))
                else:
                    moves.append(make_move(frm, to))

            # 4. en passant: capture a pawn that just double-pushed past us.
            #    The target square is empty, which is why it needs its own flag.
            if self.ep_square >= 0:
                if PAWN_ATTACKS[us][frm] >> self.ep_square & 1:
                    moves.append(make_move(frm, self.ep_square, 0, EN_PASSANT))

    def _generate_castling(self, moves, us, them, occupied):
        """Castling, generated fully legal.

        Three conditions: we still have the right, the squares between king
        and rook are empty, and the king does not start, cross, or land on an
        attacked square. That last one is why this is checked here rather
        than left to push(): push() only looks at where the king ENDS UP.
        """
        if us == WHITE:
            options = ((CASTLE_WK, 0x60, (4, 5, 6), 6),
                       (CASTLE_WQ, 0x0E, (4, 3, 2), 2))
        else:
            options = ((CASTLE_BK, 0x6000000000000000, (60, 61, 62), 62),
                       (CASTLE_BQ, 0x0E00000000000000, (60, 59, 58), 58))

        for right, must_be_empty, king_path, king_to in options:
            if not self.castling & right:
                continue
            if occupied & must_be_empty:
                continue
            if any(self.is_attacked(sq, them) for sq in king_path):
                continue
            moves.append(make_move(king_path[0], king_to, 0, CASTLE))

    def legal_moves(self):
        """Only the moves that are actually legal. Used at the root and for
        detecting checkmate; the search uses push()'s return value instead so
        it does not pay for this twice."""
        result = []
        for move in self.generate_moves():
            if self.push(move):
                self.pop()
                result.append(move)
        return result

    # -- evaluation ---------------------------------------------------------

    def evaluate(self):
        """How good is this position, in centipawns, for the side to move?

        Positive means the side to move is better. That sign convention is
        what lets the search negate scores as it recurses (see engine.py).

        Two evaluations live in this repo and both return the same units:

          `nnue.evaluate`      a trained network. This is what plays.
          `evaluate_tables`    material and piece-square bonuses, which is
                               what the network replaced, and the fallback
                               when nnue.npz or numpy is missing.

        Keeping the old one is not sentiment. It is the baseline the network
        is measured against, and it is what the search falls back to if the
        weights will not load.
        """
        if nnue is not None and nnue.AVAILABLE:
            return nnue.evaluate(self)
        return self.evaluate_tables()

    def evaluate_tables(self):
        """The hand written evaluation: material plus a bonus per square.

        The tables are written from white's point of view, so we sum them all
        and flip the sign at the end if it is black's turn.
        """
        # Pick the king table by how much material is left. Anything below
        # roughly "queen + rook each gone" counts as an endgame.
        non_pawn_material = 0
        for colour in (WHITE, BLACK):
            for kind in (KNIGHT, BISHOP, ROOK, QUEEN):
                bb = self.pieces[piece_index(colour, kind)]
                non_pawn_material += PIECE_VALUE[kind] * bin(bb).count("1")
        table = ENDGAME_SCORE if non_pawn_material < 2400 else MIDGAME_SCORE

        score = 0
        for piece in range(12):
            piece_table = table[piece]
            for sq in squares_of(self.pieces[piece]):
                score += piece_table[sq]

        return score if self.side == WHITE else -score

    # -- FEN ---------------------------------------------------------------

    def set_fen(self, fen):
        """Load a position from FEN, the standard text format.

        'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1' is
        board / side to move / castling rights / en passant / halfmove clock
        / fullmove number.
        """
        self.pieces = [0] * 12
        self.occupied = [0, 0]
        self.undo_stack = []

        parts = fen.split()
        # Ranks come highest-first in FEN, so row 0 of the text is rank 8.
        for row, text in enumerate(parts[0].split("/")):
            rank = 7 - row
            file = 0
            for ch in text:
                if ch.isdigit():
                    file += int(ch)        # a run of empty squares
                else:
                    piece = "PNBRQKpnbrqk".index(ch)
                    bit = 1 << (rank * 8 + file)
                    self.pieces[piece] |= bit
                    self.occupied[piece // 6] |= bit
                    file += 1

        self.side = WHITE if parts[1] == "w" else BLACK

        self.castling = 0
        rights = parts[2] if len(parts) > 2 else "-"
        for ch, bit in (("K", CASTLE_WK), ("Q", CASTLE_WQ),
                        ("k", CASTLE_BK), ("q", CASTLE_BQ)):
            if ch in rights:
                self.castling |= bit

        ep = parts[3] if len(parts) > 3 else "-"
        if ep == "-":
            self.ep_square = -1
        else:
            self.ep_square = (ord(ep[0]) - ord("a")) + (int(ep[1]) - 1) * 8

        self.halfmove_clock = int(parts[4]) if len(parts) > 4 else 0
        self._rebuild_key()

    def to_fen(self):
        """The position back out as FEN. The inverse of `set_fen`.

        Never used by the search; handy for pasting a position into another
        tool while debugging.
        """
        rows = []
        for rank in range(7, -1, -1):
            row, empty = "", 0
            for file in range(8):
                sq = rank * 8 + file
                for piece in range(12):
                    if self.pieces[piece] >> sq & 1:
                        if empty:
                            row += str(empty)
                            empty = 0
                        row += "PNBRQKpnbrqk"[piece]
                        break
                else:
                    empty += 1
            rows.append(row + (str(empty) if empty else ""))

        rights = "".join(ch for ch, bit in (("K", CASTLE_WK), ("Q", CASTLE_WQ),
                                            ("k", CASTLE_BK), ("q", CASTLE_BQ))
                         if self.castling & bit) or "-"
        ep = "-" if self.ep_square < 0 else square_name(self.ep_square)
        return (f"{'/'.join(rows)} {'w' if self.side == WHITE else 'b'} "
                f"{rights} {ep} {self.halfmove_clock} 1")

    def _rebuild_key(self):
        """Compute the Zobrist key from scratch. Only needed after set_fen --
        push/pop maintain it incrementally."""
        key = 0
        for piece in range(12):
            for sq in squares_of(self.pieces[piece]):
                key ^= ZOBRIST_PIECE[piece][sq]
        key ^= ZOBRIST_CASTLING[self.castling]
        if self.ep_square >= 0:
            key ^= ZOBRIST_EP_FILE[self.ep_square % 8]
        if self.side == BLACK:
            key ^= ZOBRIST_BLACK_TO_MOVE
        self.key = key

    def __str__(self):
        """Print the board, for debugging. White at the bottom."""
        rows = []
        for rank in range(7, -1, -1):
            row = []
            for file in range(8):
                sq = rank * 8 + file
                for piece in range(12):
                    if self.pieces[piece] >> sq & 1:
                        row.append("PNBRQKpnbrqk"[piece])
                        break
                else:
                    row.append(".")
            rows.append(f"{rank + 1} " + " ".join(row))
        return "\n".join(rows) + "\n  a b c d e f g h"
