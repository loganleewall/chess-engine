"""The search: given a position and a time budget, which move is best?

WHAT IS HERE
------------
1. negamax + alpha-beta      the search itself
2. iterative deepening       search depth 1, then 2, then 3... until time runs out
3. quiescence                at depth 0, keep going until no captures are left
4. transposition table       remember positions we already searched
5. move ordering             try likely-best moves first, so alpha-beta can cut

THERE IS NO TREE OBJECT. Read `_negamax` and notice that it calls itself, and
that `board.push()` / `board.pop()` mutate ONE board in place. The search tree
exists only as the chain of active function calls -- a node is a stack frame,
and it is gone the moment the function returns. Nothing is ever allocated to
represent it.
"""

import time

from board import KING, PAWN, PIECE_VALUE, move_to_uci

# Score for being checkmated. We use "MATE - ply" so that a mate found sooner
# scores higher than one found later, which makes the engine actually deliver
# mate instead of shuffling around in a won position.
MATE = 30000

# Any score above this must be a mate score rather than a normal evaluation.
MATE_THRESHOLD = MATE - 1000

INFINITY = 31000

# Transposition table entry kinds. See the comment in `_negamax` where they
# are used -- this is the subtlest idea in the file.
EXACT, LOWER_BOUND, UPPER_BOUND = 0, 1, 2


class Engine:
    def __init__(self):
        # Zobrist key -> (depth, score, kind, best_move).
        # A dict handles collisions itself, which is why there is no "is this
        # slot really my position?" check.
        self.transposition_table = {}

        self.nodes = 0                # how many positions we looked at
        self.deadline = 0.0           # time.monotonic() value to stop at
        self.stopped = False          # set when we run out of time
        self.position_history = []    # Zobrist keys of the actual game so far

    # -- the public entry point ---------------------------------------------

    def search(self, board, seconds, max_depth=64):
        """Find the best move, spending about `seconds` on it.

        ITERATIVE DEEPENING: we search to depth 1, then depth 2, then depth 3,
        and so on until the clock runs out, keeping the answer from the last
        depth we FINISHED. Two reasons this is not wasteful:

          1. We always have a complete answer ready when time runs out. If we
             just launched a depth-10 search and got interrupted, we would
             have nothing.
          2. Each shallow pass fills the transposition table, so the next,
             deeper pass tries good moves first and cuts off much faster.
             Searching 1,2,...,10 is genuinely FASTER than searching straight
             to 10, because of better move ordering.
        """
        self.nodes = 0
        self.stopped = False
        self.deadline = time.monotonic() + seconds
        start = time.monotonic()

        legal = board.legal_moves()
        if not legal:
            return None, 0, 0             # checkmate or stalemate

        best_move = legal[0]              # always have something to return
        best_score = 0
        depth_reached = 0

        for depth in range(1, max_depth + 1):
            move, score = self._search_root(board, depth, legal)

            # If the clock ran out mid-iteration, the numbers are incomplete
            # and we throw them away, keeping the previous depth's answer.
            if self.stopped:
                break

            best_move, best_score, depth_reached = move, score, depth

            # Found a forced mate we can actually reach within this depth --
            # searching deeper cannot improve on it.
            if abs(score) > MATE_THRESHOLD and MATE - abs(score) <= depth:
                break

            # Don't start a depth we cannot finish. The next depth typically
            # costs about as much as everything so far, so if we are already
            # past halfway there is no point beginning it.
            if time.monotonic() - start > seconds * 0.5:
                break

        return best_move, best_score, depth_reached

    def _search_root(self, board, depth, legal):
        """One full-depth pass over the root moves.

        This is separate from `_negamax` only so that "which move is best" is
        visible in one place.
        """
        alpha, beta = -INFINITY, INFINITY
        best_move, best_score = legal[0], -INFINITY

        # Try the move the table liked last time first -- at the root that is
        # usually the previous iteration's best, which is our best guess.
        entry = self.transposition_table.get(board.key)
        ordered = self._order_moves(board, legal, entry[3] if entry else None)

        for move in ordered:
            board.push(move)
            self.position_history.append(board.key)

            # The minus sign is the whole trick -- see `_negamax`.
            score = -self._negamax(board, depth - 1, -beta, -alpha, 1)

            self.position_history.pop()
            board.pop()

            if self.stopped:
                return best_move, best_score

            if score > best_score:
                best_score, best_move = score, move
            if score > alpha:
                alpha = score               # we have a new floor to beat

        self.transposition_table[board.key] = (depth, best_score, EXACT,
                                               best_move)
        return best_move, best_score

    # -- the recursion ------------------------------------------------------

    def _negamax(self, board, depth, alpha, beta, ply):
        """Score this position for the side to move, searching `depth` deeper.

        NEGAMAX: one function scores every position from the point of view of
        whoever is to move, and the caller negates:

            score = -self._negamax(...)

        That works because chess is zero-sum: a position worth +200 to me is
        worth -200 to you. Without it you would need separate "maximising" and
        "minimising" functions (that is textbook minimax); this is the same
        algorithm written once.

        ALPHA-BETA: `alpha` is the best score we have already proved we can
        get somewhere else. `beta` is the best the OPPONENT has already proved
        they can get. If we find a move here scoring >= beta, the opponent
        will simply avoid this whole position, so its exact value is
        irrelevant -- we stop looking. That is a "cutoff", and it is what
        makes searching to depth 14 possible at all.
        """
        self.nodes += 1

        # Check the clock occasionally. `time.monotonic()` is far too slow to
        # call every node, so we do it every 2048.
        if self.nodes % 2048 == 0 and time.monotonic() >= self.deadline:
            self.stopped = True
        if self.stopped:
            return 0

        # Draws. Checked before anything else because a draw ends the game
        # regardless of how good the position looks.
        if ply > 0:
            if self._is_repetition(board) or board.halfmove_clock >= 100:
                return 0

        # --- transposition table lookup -----------------------------------
        #
        # We may have searched this exact position before, via a different
        # move order. What we stored depends on how that search ended:
        #
        #   EXACT       we searched every move: the score is the true value.
        #   LOWER_BOUND we hit a cutoff, so the true score is AT LEAST this.
        #   UPPER_BOUND nothing beat alpha, so the true score is AT MOST this.
        #
        # A bound is only reusable if it still answers the current question,
        # which is what the two `if` tests below decide.
        tt_move = None
        entry = self.transposition_table.get(board.key)
        if entry is not None:
            tt_depth, tt_score, tt_kind, tt_move = entry
            if tt_depth >= depth and ply > 0:
                score = self._score_from_table(tt_score, ply)
                if tt_kind == EXACT:
                    return score
                if tt_kind == LOWER_BOUND and score >= beta:
                    return score
                if tt_kind == UPPER_BOUND and score <= alpha:
                    return score

        if depth <= 0:
            return self._quiesce(board, alpha, beta, ply)

        moves = board.generate_moves()
        alpha_at_entry = alpha            # needed to classify our TT entry
        best_score = -INFINITY
        best_move = None
        legal_count = 0

        for move in self._order_moves(board, moves, tt_move):
            if not board.push(move):      # illegal: left our king in check
                continue
            legal_count += 1
            self.position_history.append(board.key)

            score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1)

            self.position_history.pop()
            board.pop()

            if self.stopped:
                return 0

            if score > best_score:
                best_score, best_move = score, move
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break                     # cutoff: opponent avoids this line

        # No legal moves at all: checkmate if we are in check, else stalemate.
        # `-MATE + ply` makes being mated later less bad than being mated now,
        # which is what makes a losing engine fight instead of giving up.
        if legal_count == 0:
            return -MATE + ply if board.in_check() else 0

        # --- store what we learned ----------------------------------------
        if best_score <= alpha_at_entry:
            kind = UPPER_BOUND            # never beat alpha: it is a ceiling
        elif best_score >= beta:
            kind = LOWER_BOUND            # we cut off: it is a floor
        else:
            kind = EXACT                  # searched everything

        existing = self.transposition_table.get(board.key)
        if existing is None or existing[0] <= depth:
            self.transposition_table[board.key] = (
                depth, self._score_to_table(best_score, ply), kind, best_move)

        return best_score

    def _quiesce(self, board, alpha, beta, ply):
        """Keep searching captures until the position is quiet.

        THE HORIZON PROBLEM, which this exists to fix: suppose depth runs out
        right after we capture a defended pawn with our queen. `evaluate()`
        counts a won pawn and reports +100 -- it has no idea our queen is
        about to be taken. Stopping only at quiet positions fixes this.

        STAND-PAT: unlike the main search, we are not obliged to capture. So
        the static evaluation acts as a floor: if it already beats beta, we
        can return immediately, because the opponent will avoid this position
        whether or not a good capture exists.

        The one exception is being in check, where we ARE obliged to respond,
        so stand-pat is not available and every legal move gets searched.
        """
        self.nodes += 1
        if self.nodes % 2048 == 0 and time.monotonic() >= self.deadline:
            self.stopped = True
        if self.stopped:
            return 0

        in_check = board.in_check()

        if in_check:
            stand_pat = -INFINITY         # no "do nothing" option when in check
        else:
            stand_pat = board.evaluate()
            if stand_pat >= beta:
                return stand_pat
            if stand_pat > alpha:
                alpha = stand_pat

        if ply > 60:                      # hard stop; long capture chains exist
            return board.evaluate()

        moves = board.generate_moves()
        if not in_check:
            # Captures and promotions only. Everything else is "quiet" and
            # by definition does not change the tactical picture.
            moves = [m for m in moves if self._is_capture(board, m) or m[2]]

        legal_count = 0
        for move in self._order_moves(board, moves, None):
            if not board.push(move):
                continue
            legal_count += 1
            score = -self._quiesce(board, -beta, -alpha, ply + 1)
            board.pop()

            if self.stopped:
                return 0
            if score >= beta:
                return score
            if score > alpha:
                alpha = score

        # Only meaningful when in check, because that is the only case where
        # we generated ALL moves. With captures only, "no moves" just means
        # "nothing to capture", not checkmate.
        if in_check and legal_count == 0:
            return -MATE + ply

        return alpha

    # -- helpers -----------------------------------------------------------

    def _is_capture(self, board, move):
        """Does `move` take something? En passant counts (flag 2)."""
        return bool(board.occupied[1 - board.side] >> move[1] & 1) \
            or move[3] == 2

    def _order_moves(self, board, moves, tt_move):
        """Sort moves best-guess-first.

        This matters enormously. Alpha-beta only cuts off when it finds a good
        move EARLY -- with perfect ordering it examines roughly the square
        root of the positions it would otherwise need. Ordering is the single
        biggest contributor to how deep the engine can see.

        Our priorities:
          1. the move the transposition table liked here before
          2. captures, best first, by MVV-LVA
          3. promotions
          4. everything else, in generation order

        MVV-LVA = "Most Valuable Victim, Least Valuable Attacker": take the
        biggest piece you can with the smallest piece you can. Capturing a
        queen with a pawn is the dream; capturing a pawn with a queen usually
        is not. Multiplying the victim's value by 16 makes the victim dominate
        so the attacker only breaks ties.

        Killer moves and a history table are the natural next heuristics to
        add here. Both are pure ordering improvements -- they change no
        results, only how fast cutoffs are found.
        """
        them = 1 - board.side

        def priority(move):
            frm, to, promotion, flag = move
            if move == tt_move:
                return 3_000_000
            if self._is_capture(board, move):
                victim = PAWN if flag == 2 else board.piece_on(to, them)
                attacker = board.piece_on(frm, board.side)
                if victim < 0:
                    victim = PAWN
                if attacker < 0:
                    attacker = KING
                return 2_000_000 + PIECE_VALUE[victim] * 16 \
                    - PIECE_VALUE[attacker]
            if promotion:
                return 1_000_000 + promotion
            return 0

        return sorted(moves, key=priority, reverse=True)

    def _is_repetition(self, board):
        """Has this exact position occurred earlier in the game or search?

        Only positions with the SAME SIDE TO MOVE can be repetitions, and the
        Zobrist key includes the side to move -- so we step back through the
        history two plies at a time. Checking every ply would compare our
        positions against the opponent's and never match anything.

        (Comparing adjacent plies is an easy bug to write: it matches nothing,
        and the engine draws a completely won endgame by repetition while
        being blind to it.)

        We also stop at the last irreversible move: a capture or pawn move
        makes every earlier position unreachable, and `halfmove_clock` counts
        exactly how far back that was.
        """
        history = self.position_history
        key = board.key
        stop = max(0, len(history) - 1 - board.halfmove_clock)
        # -1 is the current position; -3 is the last time it was our turn.
        for i in range(len(history) - 3, stop - 1, -2):
            if history[i] == key:
                return True
        return False

    # Mate scores are stored as "distance from HERE", but read back at a
    # different distance from the root, so they need converting both ways.
    # Without this the engine reports mates that do not exist.
    def _score_to_table(self, score, ply):
        if score > MATE_THRESHOLD:
            return score + ply
        if score < -MATE_THRESHOLD:
            return score - ply
        return score

    def _score_from_table(self, score, ply):
        if score > MATE_THRESHOLD:
            return score - ply
        if score < -MATE_THRESHOLD:
            return score + ply
        return score


def time_for_move(time_left_ms, increment_ms=500, moves_to_go=28,
                  reserve_ms=1500):
    """How many seconds to spend on this move.

    Running out of time loses the game outright, so we keep a `reserve_ms`
    cushion that is NEVER spent -- it absorbs delays we cannot see or measure
    (the harness's own network round trip and legality checking).

    Of what is left, we take a 28th (a rough guess at how many moves remain)
    plus most of the increment we get back each move, and never more than a
    third of the clock on any single move.
    """
    spare = max(time_left_ms - reserve_ms, 0)
    target = spare / moves_to_go + increment_ms * 0.8
    hard_cap = spare * 0.33
    return max(0.02, min(target, hard_cap) / 1000.0)
