"""Play a game against the engine in the terminal.

    python3 human.py                 # you are White, engine thinks for 1s
    python3 human.py black           # you are Black
    python3 human.py white 3.0       # give the engine 3s per move

Moves are typed in UCI: the square you are moving from, then the square you
are moving to, plus a piece letter when you promote. So "e2e4", "g8f6",
"a7a8q". Three words do something else instead of moving:

    board   reprint the position
    moves   list every legal move
    quit    stop the game

Nothing here is used by the competition harness. It exists so a person can
sit down and confirm in two minutes that the engine really plays chess.
"""

import sys

from board import WHITE, Board, move_to_uci
from engine import Engine


def read_human_move(board):
    """Ask until the answer is a legal move. Returns None if the player quits.

    We build the legal moves once and index them by their UCI text, so
    checking the input is a dictionary lookup rather than any parsing. That
    also means an unreachable move and a typo fail the same way, which is the
    behaviour you want.
    """
    legal = {move_to_uci(move): move for move in board.legal_moves()}

    while True:
        try:
            text = input("your move > ").strip().lower()
        except EOFError:
            return None

        if text in ("quit", "exit", "q"):
            return None
        if text == "board":
            print(board)
            continue
        if text == "moves":
            print("  " + " ".join(sorted(legal)))
            continue
        if text in legal:
            return legal[text]

        print(f"  '{text}' is not legal here. Type 'moves' for the list.")


def verdict(board):
    """A one-line result if the game is over, otherwise None."""
    if not board.legal_moves():
        if board.in_check():
            loser = "White" if board.side == WHITE else "Black"
            winner = "Black" if board.side == WHITE else "White"
            return f"checkmate, {winner} wins ({loser} has no move)"
        return "stalemate, draw"
    if board.halfmove_clock >= 100:
        return "fifty-move rule, draw"
    return None


def main():
    human_side = WHITE
    seconds = 1.0
    if len(sys.argv) > 1 and sys.argv[1].lower().startswith("b"):
        human_side = 1 - WHITE
    if len(sys.argv) > 2:
        seconds = float(sys.argv[2])

    board = Board()
    engine = Engine()
    # The search needs the game's earlier positions to spot a repetition, and
    # a fresh game has exactly one.
    engine.position_history = [board.key]

    print(f"\nYou are {'White' if human_side == WHITE else 'Black'}. "
          f"The engine gets {seconds:g}s a move.\n")
    print(board)

    while True:
        result = verdict(board)
        if result:
            print(f"\n{result}")
            return

        if board.side == human_side:
            move = read_human_move(board)
            if move is None:
                print("\ngame abandoned")
                return
        else:
            move, score, depth = engine.search(board, seconds)
            # Score is from the mover's point of view, so a positive number
            # here means the engine thinks it is ahead.
            print(f"\nengine plays {move_to_uci(move)}   "
                  f"depth {depth}  score {score}  {engine.nodes:,} nodes")

        board.push(move)
        engine.position_history.append(board.key)
        print()
        print(board)
        print()


if __name__ == "__main__":
    main()
