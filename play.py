"""Watch the simplified engine play itself. Useful for convincing yourself it
really works, and for seeing the search's output move by move.

    python3 play.py            # 0.2s per move, 60 moves
    python3 play.py 1.0 200    # 1s per move, up to 200 moves
"""

import sys

from board import Board, move_to_uci
from engine import Engine


def main():
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 0.2
    max_moves = int(sys.argv[2]) if len(sys.argv) > 2 else 60

    board = Board()
    engine = Engine()
    engine.position_history = [board.key]

    for move_number in range(1, max_moves + 1):
        move, score, depth = engine.search(board, seconds)

        if move is None:
            if board.in_check():
                winner = "Black" if board.side == 0 else "White"
                print(f"\ncheckmate -- {winner} wins")
            else:
                print("\nstalemate -- draw")
            break

        if board.halfmove_clock >= 100:
            print("\nfifty-move rule -- draw")
            break

        side = "White" if board.side == 0 else "Black"
        # Score is always from the mover's point of view, so a positive number
        # means whoever just moved thinks they are ahead.
        print(f"{move_number:>3}. {side:<5} {move_to_uci(move):<6} "
              f"depth {depth:<3} score {score:>6}  "
              f"{engine.nodes:>8,} nodes")

        board.push(move)
        engine.position_history.append(board.key)
    else:
        print(f"\nstopped after {max_moves} moves")

    print()
    print(board)


if __name__ == "__main__":
    main()
