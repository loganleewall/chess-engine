"""
WHAT IS BEING LEARNED
---------------------
A number for a quiet position, in the units the search wants.

The labels come from Stockfish, via the Lichess evaluation dump: 395 million
positions that somebody already spent the compute to evaluate deeply. We are
not learning to play chess. We are learning to predict a strong engine's
verdict on a position, quickly, so that our own search can call it millions of
times a second.

"""
import argparse
import glob
import time

import numpy as np
import torch
import torch.nn as nn

# Quantization scales. These are a contract with nnue.py: change one here
# and the weights stop being readable.
QA, QB, SCALE = 255, 64, 400

NUM_ZONES = 16
FEATURES = NUM_ZONES * 768          # 12,288
OUT_BUCKETS = 8
WEIGHT_CLIP = 1.98                  # keeps round(w * QA) inside int16


def king_zone_table():
    """Which of 16 zones each king square belongs to, from its own side's view.

    Rank 0 here is the king's own home rank, so this table is read after the
    square has already been mirrored for black.

    The split is uneven on purpose, because king position matters much more in
    some places than others. The home rank gets six zones, since castled short,
    castled long and still in the centre are genuinely different worlds. The
    two ranks above it get three each. A king that has walked further up the
    board than that is either in an endgame or in trouble, and only which half
    of the board it is on still matters, so those get two each.
    """
    table = np.zeros(64, dtype=np.int64)
    for square in range(64):
        rank, file = divmod(square, 8)
        if rank == 0:
            zone = (0, 0, 1, 2, 3, 4, 5, 5)[file]
        elif rank == 1:
            zone = 6 + (0 if file <= 2 else 1 if file <= 4 else 2)
        elif rank == 2:
            zone = 9 + (0 if file <= 2 else 1 if file <= 4 else 2)
        elif rank <= 4:
            zone = 12 + (0 if file <= 3 else 1)
        else:
            zone = 14 + (0 if file <= 3 else 1)
        table[square] = zone
    return table


KING_ZONE = king_zone_table()


def feature_indices(piece, square, zones):
    """Piece lists -> the input indices that are set, for both perspectives.

    `piece` and `square` are (batch, 32), one slot per piece. A position with
    fewer than 32 pieces fills its leftover slots with piece -1, and whatever
    square sits beside a -1 is ignored. Returns two (batch, 32) index tensors,
    one per perspective, padded with FEATURES so the embedding's padding row
    (which is held at zero) absorbs them.

    This is the batched twin of `feature_indices` in nnue.py. Same arithmetic,
    same result; that file does one position at a time in plain Python, this
    one does 16,384 at once on the GPU.
    """
   
    padded = piece < 0
    colour = torch.where(padded, 0, piece // 6) #white or black
    kind = torch.where(padded, 0, piece % 6) #which piece

    out = []
    for perspective in (0, 1):              # 0 = White's view, 1 = Black's
        own_king = 5 if perspective == 0 else 11
        # Exactly one square holds our king, so masking and summing finds it.
        king_square = (square * (piece == own_king)).sum(1)

        # Black sees the board upside down. XOR with 56 flips the rank and
        # keeps the file (e8 <-> e1, a7 <-> a2); for White it is XOR 0, a no-op.
        # The king's square is flipped too, because the zone table is written
        # from the king's own side of the board.
        flip = 56 * perspective
        zone = zones[king_square ^ flip]

        # One index per piece, built from four parts. Each part is multiplied
        # by the size of everything after it, so every combination lands on its
        # own number, 0..12,287:
        #
        #   zone          which of 16 zones MY king is in        x 768
        #   mine/theirs   colour ^ perspective, 0 means mine     x 384
        #   kind          pawn .. king                           x 64
        #   square        0..63, flipped for Black               x 1
        #
        # `zone` is one number per position; [:, None] repeats it across all 32
        # of that position's pieces.
        index = (zone[:, None] * 768
                 + (colour ^ perspective) * 384
                 + kind * 64
                 + (square ^ flip))

        # Empty slots point at the extra all-zero row, so they add nothing.
        out.append(torch.where(padded, FEATURES, index))
    return out[0], out[1]


def output_bucket(piece):
    """Which of the 8 linear output heads, by how many pieces are left.

    An endgame and a middlegame do not share a scale: a one pawn edge with
    queens on is worth much less than the same pawn in a king and pawn ending.
    Giving each piece count band its own output row lets the network say so
    without having to encode it in the hidden layer.
    """
    count = (piece >= 0).sum(1)
    return ((count - 2) // 4).clamp(0, OUT_BUCKETS - 1) 


class Net(nn.Module):
    """(16 x 768 -> hidden) x 2 -> 8 x 1."""

    def __init__(self, hidden):
        super().__init__()
        # The first layer is an Embedding, not a Linear. Mathematically it is
        # the same thing (a 12,288 x hidden matrix), but the input is a list of
        # at most 32 set indices rather than a 12,288-long vector of mostly
        # zeros, so summing the named rows is the whole first layer.
        self.ft = nn.Embedding(FEATURES + 1, hidden, padding_idx=FEATURES) #embedding
        self.ft_b = nn.Parameter(torch.zeros(hidden)) #bias
        self.out_w = nn.Parameter(torch.zeros(OUT_BUCKETS, 2 * hidden)) #weight table
        self.out_b = nn.Parameter(torch.zeros(OUT_BUCKETS))

        nn.init.normal_(self.ft.weight, std=0.05)
        nn.init.normal_(self.out_w, std=0.05)
        with torch.no_grad():
            self.ft.weight[FEATURES] = 0        # the padding row stays zero

    def forward(self, white_idx, black_idx, side_to_move, bucket):
        white = self.ft(white_idx).sum(1) + self.ft_b
        black = self.ft(black_idx).sum(1) + self.ft_b

        # Order the two accumulators as "mine" then "theirs" rather than
        # "white" then "black", so one set of output weights serves both sides.
        white_moves = (side_to_move == 0)[:, None]
        mine = torch.where(white_moves, white, black)
        theirs = torch.where(white_moves, black, white)

        # SCReLU: clamp to [0, 1], then square. The clamp is a clipped ReLU,
        # the integer version's clamp to [0, QA]; together with the square it
        # is the network's only nonlinearity. The square lets pairs of pieces
        # interact, and it costs one multiply.
        hidden = torch.cat([mine, theirs], 1).clamp(0, 1)
        return (hidden * hidden * self.out_w[bucket]).sum(1) + self.out_b[bucket]

    def clip_weights(self):
        """Keep every weight inside the range int16 quantization can hold."""
        with torch.no_grad():
            self.ft.weight.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)
            self.out_w.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)


class Shard:
    """One data file, held as CPU tensors and expanded per batch on the device.

    Only one shard is in memory at a time. That is the whole reason the
    training set can be 96 million positions on a laptop: the shards live on
    disk and the loop walks them.
    """

    def __init__(self, path):
        data = np.load(path)
        self.piece = torch.from_numpy(data["piece"])      # (n, 32) int8
        self.square = torch.from_numpy(data["square"])    # (n, 32) int8
        self.stm = torch.from_numpy(data["stm"])          # (n,)    int8
        # Labels are stored from White's point of view; the network always
        # answers from the mover's, so flip the sign when Black is to move.
        cp_white = torch.from_numpy(data["cp"].astype(np.float32))
        self.cp = torch.where(self.stm == 0, cp_white, -cp_white)
        self.target = torch.sigmoid(self.cp / SCALE)
        self.n = len(self.target)

    def batch(self, index, device, zones):
        piece = self.piece[index].to(device).long()
        square = self.square[index].to(device).long()
        white_idx, black_idx = feature_indices(piece, square, zones)
        return (white_idx, black_idx, self.stm[index].to(device).long(),
                output_bucket(piece), self.target[index].to(device))


def quantize(net):
    """Round the float weights to the integers the engine reads.

    Each scale is chosen so the arithmetic downstream cannot overflow:
    inputs and the accumulator at QA=255, output weights at QB=64, and the
    output bias at QA*QB because it is added after both have been applied.
    """
    with torch.no_grad():
        ft_w = torch.round(net.ft.weight[:FEATURES].cpu() * QA).to(torch.int16)
        ft_b = torch.round(net.ft_b.cpu() * QA).to(torch.int16)
        out_w = torch.round(net.out_w.cpu() * QB).to(torch.int16)
        out_b = torch.round(net.out_b.cpu() * QA * QB).to(torch.int32)
    return ft_w.numpy(), ft_b.numpy(), out_w.numpy(), out_b.numpy()


def integer_eval(piece, square, stm, ft_w, ft_b, out_w, out_b):
    """The engine's integer forward pass, in numpy, for checking the export.

    This must agree with `nnue.evaluate`. It exists so that the mismatch, if
    there is one, is caught here rather than in a rated game three days later.
    """
    piece = torch.from_numpy(piece.astype(np.int64))
    square = torch.from_numpy(square.astype(np.int64))
    white_idx, black_idx = feature_indices(piece, square,
                                           torch.from_numpy(KING_ZONE))
    hidden = ft_w.shape[1]

    table = np.vstack([ft_w.astype(np.int64), np.zeros((1, hidden), np.int64)])
    white = table[white_idx.numpy()].sum(1) + ft_b.astype(np.int64)
    black = table[black_idx.numpy()].sum(1) + ft_b.astype(np.int64)

    white_moves = (stm == 0)[:, None]
    mine = np.where(white_moves, white, black)
    theirs = np.where(white_moves, black, white)

    activated = np.clip(np.concatenate([mine, theirs], 1), 0, QA)
    bucket = output_bucket(piece).numpy()
    total = (activated * activated * out_w.astype(np.int64)[bucket]).sum(1)
    return (total // QA + out_b.astype(np.int64)[bucket]) * SCALE // (QA * QB)


def main():
    parser = argparse.ArgumentParser(description="Train the evaluation network.")
    parser.add_argument("--out", default="mynet.npz", help="where to write the weights")
    parser.add_argument("--data", default="data/sample_train.npz",
                        help="glob of training shards")
    parser.add_argument("--val", default="data/sample_val.npz",
                        help="held out shard, never trained on")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        args.device = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(args.device)
    torch.manual_seed(0)

    shards = sorted(f for f in glob.glob(args.data) if f != args.val)
    if not shards:
        raise SystemExit(f"no training shards matched {args.data!r}")
    validation = Shard(args.val)

    zones = torch.from_numpy(KING_ZONE).to(device)
    net = Net(args.hidden).to(device)
    optimiser = torch.optim.Adam(net.parameters(), lr=args.lr)

    steps_per_epoch = sum(Shard(f).n // args.batch for f in shards)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=args.lr,
        total_steps=max(steps_per_epoch * args.epochs, 4),
        pct_start=0.15, final_div_factor=30)

    print(f"{len(shards)} shard(s), {steps_per_epoch * args.batch:,} positions "
          f"per epoch, {validation.n:,} held out")
    print(f"hidden {args.hidden}, batch {args.batch}, {args.epochs} epochs, "
          f"device {device}\n")

    def validation_loss():
        net.eval()
        total = 0.0
        with torch.no_grad():
            for start in range(0, validation.n, args.batch):
                index = torch.arange(start, min(start + args.batch, validation.n))
                *inputs, target = validation.batch(index, device, zones)
                predicted = torch.sigmoid(net(*inputs))
                total += ((predicted - target) ** 2).sum().item()
        net.train()
        return total / validation.n

    print(f"  before training   val {validation_loss():.5f}")
    started = time.monotonic()

    for epoch in range(1, args.epochs + 1):
        running, steps = 0.0, 0
        for path in shards:
            shard = Shard(path)
            order = torch.randperm(shard.n)
            for step in range(shard.n // args.batch):
                index = order[step * args.batch:(step + 1) * args.batch]
                *inputs, target = shard.batch(index, device, zones)

                predicted = torch.sigmoid(net(*inputs))
                loss = ((predicted - target) ** 2).mean()

                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                optimiser.step()
                schedule.step()
                net.clip_weights()      # after the step, so it always holds

                running += loss.item()
                steps += 1
            del shard                   # one shard resident at a time

        print(f"  epoch {epoch:>3}        train {running / max(steps, 1):.5f}   "
              f"val {validation_loss():.5f}   {time.monotonic() - started:4.0f}s")

    # --- export, and prove the integers still agree with the floats --------
    ft_w, ft_b, out_w, out_b = quantize(net)

    sample = torch.arange(0, min(20000, validation.n))
    with torch.no_grad():
        *inputs, _ = validation.batch(sample, device, zones)
        float_cp = (net(*inputs) * SCALE).cpu().numpy()
    i = sample.numpy()
    int_cp = integer_eval(validation.piece[i].numpy(), validation.square[i].numpy(),
                          validation.stm[i].numpy(), ft_w, ft_b, out_w, out_b)

    error = np.abs(float_cp - int_cp)
    truth = validation.cp[i].numpy()
    print()
    print(f"  quantization error   mean {error.mean():.1f} cp, max {error.max():.0f} cp")
    print(f"  vs stockfish         mean {np.abs(int_cp - truth).mean():.0f} cp "
          f"(predicting 0 would be {np.abs(truth).mean():.0f} cp)")

    np.savez(args.out, ft_w=ft_w, ft_b=ft_b, out_w=out_w, out_b=out_b,
             hidden=np.int32(args.hidden), out_buckets=np.int32(OUT_BUCKETS),
             num_kb=np.int32(NUM_ZONES), king_bucket=KING_ZONE.astype(np.int64))
    print(f"\n  wrote {args.out}")
    print(f"  to use it:  cp {args.out} nnue.npz")


if __name__ == "__main__":
    main()
