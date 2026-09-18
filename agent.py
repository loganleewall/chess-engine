"""The competition entry point: the one function the harness calls.

The simplified mirror of the full engine's `agent.py`: 78 lines of code
there, 67 here.

THE HARNESS'S ENTIRE API IS ONE FUNCTION:

    get_move(fen, time_left_ms) -> "e2e4"

It imports this module once per game, then calls `get_move` for every move we
have to play. We never see the opponent's moves directly -- just a fresh FEN
each turn, which is why the repetition bookkeeping below is fiddlier than you
would expect.

THIS FILE'S REAL JOB IS DAMAGE CONTROL. Three things score as an instant loss
no matter how well the engine plays, and each one gets a specific defence:

    returning an illegal move   ->  re-check the answer against a fresh
                                    legal-move list before returning it
    crashing                    ->  catch every exception, return any legal move
    running out of time         ->  time_for_move() keeps a reserve it never spends

The real agent.py does two more things this one does not need: it pins numpy's
and numba's thread pools to one thread BEFORE importing them (they size
themselves from the host's core count at import, and would thrash on a single
dedicated core), and it runs a throwaway search at import to force numba to
compile everything inside the 90-second startup budget rather than on the game
clock. Neither applies here -- pure Python has no compile step.
"""

import sys

from board import Board, move_to_uci
from engine import Engine, time_for_move

_engine = Engine()
_board = Board()

# Every position this game has passed through, as Zobrist keys, for detecting
# repetitions. Bounded because a repetition can never reach back past the last
# capture or pawn move -- 100 plies at the very most.
_history = []
_HISTORY_LIMIT = 200

_last_fullmove_number = 0


def _log(message):
    """Log to stderr, never stdout: stdout may be the harness's own channel."""
    print(message, file=sys.stderr, flush=True)


def _choose_move(fen, time_left_ms):
    """The normal path. Anything that goes wrong here lands in `_fallback`."""
    global _last_fullmove_number

    # Detect the harness reusing this process for a NEW game. The fullmove
    # number only ever goes up within a game, so if it drops, our repetition
    # history belongs to a game that is over.
    parts = fen.split()
    fullmove_number = int(parts[5]) if len(parts) > 5 else 1
    if fullmove_number < _last_fullmove_number:
        _history.clear()
    _last_fullmove_number = fullmove_number

    _board.set_fen(fen)

    if not _history or _history[-1] != _board.key:
        _history.append(_board.key)
    del _history[:-_HISTORY_LIMIT]

    # Hand the search the history EXCLUDING the current position, which it
    # appends itself as it descends.
    _engine.position_history = list(_history[:-1])

    seconds = time_for_move(time_left_ms)
    move, score, depth = _engine.search(_board, seconds)

    if move is None:                      # no legal moves: game is already over
        raise ValueError("no legal moves")

    # PARANOIA, AND THE MOST IMPORTANT FOUR LINES IN THE FILE.
    # An illegal move forfeits the game instantly, so we never take the
    # search's word for it. Regenerate from a clean position and confirm.
    _board.set_fen(fen)
    legal = _board.legal_moves()
    if move not in legal:
        _log(f"search returned an illegal move; playing {move_to_uci(legal[0])}")
        move = legal[0]

    # Record the position our own move produces. The harness only asks us every
    # OTHER ply, so without this the history would have a hole at each of our
    # moves and the two-ply-step repetition check would be built on a list with
    # gaps in it.
    _board.set_fen(fen)
    _board.push(move)
    _history.append(_board.key)

    _log(f"depth {depth} score {score} nodes {_engine.nodes} "
         f"move {move_to_uci(move)}")
    return move_to_uci(move)


def _fallback(fen):
    """Any legal move beats crashing: a crash is a loss, a weak move is not.

    Deliberately uses nothing but `board.py`, so a bug in the search cannot
    reach it. Prefers a mate, then a capture, then whatever is first.
    """
    board = Board(fen)
    legal = board.legal_moves()
    if not legal:
        return "0000"                     # the null move; the game is over

    for move in legal:                    # is anything mate?
        board.push(move)
        mated = board.in_check() and not board.legal_moves()
        board.pop()
        if mated:
            return move_to_uci(move)

    for move in legal:                    # otherwise grab material
        if board.occupied[1 - board.side] >> move[1] & 1:
            return move_to_uci(move)

    return move_to_uci(legal[0])


def get_move(fen: str, time_left_ms: int) -> str:
    """What the harness calls. Must always return a legal UCI move string."""
    try:
        return _choose_move(fen, time_left_ms)
    except Exception as exc:              # broad on purpose: a crash is a loss
        _log(f"search failed ({exc!r}); falling back")
        return _fallback(fen)


if __name__ == "__main__":
    # A smoke test of the real API: three positions, one obvious answer each.
    CASES = [
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
         "opening move"),
        ("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", "mate in 1: expect a1a8"),
        ("7k/8/8/3qQ3/8/8/8/7K w - - 0 1", "free queen: expect e5d5"),
    ]
    for fen, description in CASES:
        print(f"{description:<32} -> {get_move(fen, 10_000)}")
