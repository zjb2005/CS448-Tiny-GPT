# version 2 of the training script. same data, same model code (model.py), but:
#
#   1. bigger model: 6 layers, 384 dim, 6 heads  (~10.8M params, v1 had ~1.9M)
#      v1 never overfit (train and val loss stayed together the whole time),
#      so the model was too small, not the data
#   2. cosine learning rate with warmup instead of one sudden drop at 80%
#   3. keeps the checkpoint with the best val loss, in case the bigger model
#      starts to overfit near the end
#   4. saves a checkpoint along the way, so if colab disconnects at minute 30
#      we don't lose everything
#
# takes about 35 min on a colab T4. after the first 100 steps it prints an
# estimate of how long the whole thing will take.
#
# usage:
#   python train_v2.py          full run (GPU)
#   python train_v2.py quick    tiny run to check the code works
#
# writes NEW files so it doesn't overwrite version 1:
#   ckpt_v2.pt, loss_log_v2.csv, loss_curve_v2.png, samples_v2.txt

import os
import sys
import math
import time
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import TinyGPT

# ---------------- settings ----------------
batch_size = 64
block_size = 256
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.1
max_lr = 1e-3
min_lr = 1e-4
warmup = 200          # steps where lr slowly goes up from 0
max_iters = 5000
eval_every = 250
eval_batches = 40
sample_at = [0, 500, 1000, 2500]

if len(sys.argv) > 1 and sys.argv[1] == "quick":
    print("quick mode, tiny settings")
    batch_size = 16
    block_size = 64
    n_embd = 64
    n_head = 4
    n_layer = 2
    warmup = 20
    max_iters = 300
    eval_every = 50
    eval_batches = 10
    sample_at = [0]

torch.manual_seed(1337)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)


def get_lr(it):
    # warmup: go from 0 up to max_lr in a straight line.
    # starting with a big lr on a fresh random model can blow things up
    if it < warmup:
        return max_lr * (it + 1) / warmup
    # after that, follow half a cosine wave from max_lr down to min_lr
    progress = (it - warmup) / (max_iters - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ---------------- data (same as v1) ----------------
f = open("data/stories.txt")
text = f.read()
f.close()
print("characters in dataset:", len(text))

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {}
itos = {}
for i, ch in enumerate(chars):
    stoi[ch] = i
    itos[i] = ch
print("vocab size:", vocab_size)


def encode(s):
    return [stoi[c] for c in s]


def decode(ids):
    return "".join([itos[i] for i in ids])


data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
print("train chars:", len(train_data), " val chars:", len(val_data))


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size - 1, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ["train", "val"]:
        total = 0.0
        for k in range(eval_batches):
            x, y = get_batch(split)
            _, loss = model(x, y)
            total += loss.item()
        out[split] = total / eval_batches
    model.train()
    return out


def write_sample(step):
    model.eval()
    start = torch.tensor([encode("Once upon a time")], dtype=torch.long, device=device)
    out = model.generate(start, 300, temperature=0.8)
    model.train()
    s = decode(out[0].tolist())
    f = open("samples_v2.txt", "a")
    f.write("=" * 20 + " step " + str(step) + " " + "=" * 20 + "\n")
    f.write(s + "\n\n")
    f.close()
    print("--- sample at step", step, "---")
    print(s)
    print("---")


config = {
    "vocab_size": vocab_size,
    "block_size": block_size,
    "n_embd": n_embd,
    "n_head": n_head,
    "n_layer": n_layer,
    "dropout": dropout,
}


def save(path, step, val_loss):
    torch.save({"model": model.state_dict(), "chars": chars, "config": config,
                "step": step, "val_loss": val_loss}, path)


# ---------------- model ----------------
model = TinyGPT(vocab_size, block_size, n_embd, n_head, n_layer, dropout).to(device)
n_params = 0
for p in model.parameters():
    n_params += p.numel()
print("number of parameters:", n_params)

opt = torch.optim.AdamW(model.parameters(), lr=max_lr)

open("samples_v2.txt", "w").close()
log = open("loss_log_v2.csv", "w")
log.write("step,train_loss,val_loss,lr\n")

steps_logged = []
train_hist = []
val_hist = []
lr_hist = []
best_val = 999.0
best_step = 0
t0 = time.time()

for it in range(max_iters + 1):

    lr = get_lr(it)
    for g in opt.param_groups:
        g["lr"] = lr

    if it == 100:
        per_step = (time.time() - t0) / 100
        print("about %.2f s per step, whole run should take around %.0f minutes"
              % (per_step, per_step * max_iters * 1.1 / 60))

    if it % eval_every == 0 or it == max_iters:
        losses = estimate_loss()
        msg = "step %5d | train loss %.4f | val loss %.4f | lr %.6f | %.0fs" % (
            it, losses["train"], losses["val"], lr, time.time() - t0)
        if losses["val"] < best_val:
            best_val = losses["val"]
            best_step = it
            save("ckpt_v2.pt", it, best_val)
            msg += "  (new best, saved)"
        print(msg)
        log.write("%d,%.4f,%.4f,%.6f\n" % (it, losses["train"], losses["val"], lr))
        log.flush()
        steps_logged.append(it)
        train_hist.append(losses["train"])
        val_hist.append(losses["val"])
        lr_hist.append(lr)

    if it in sample_at:
        write_sample(it)

    if it == max_iters:
        break

    x, y = get_batch("train")
    logits, loss = model(x, y)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

log.close()
print("training took %.1f minutes" % ((time.time() - t0) / 60))
print("best val loss %.4f at step %d, that one is in ckpt_v2.pt" % (best_val, best_step))

# the last sample should come from the model we actually kept, not the last step
ckpt = torch.load("ckpt_v2.pt", map_location=device)
model.load_state_dict(ckpt["model"])
write_sample("best (step %d)" % best_step)

# ---------------- plots ----------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(steps_logged, train_hist, label="v2 train")
axes[0].plot(steps_logged, val_hist, label="v2 val")

# if the v1 log is still around, draw it too so we can compare
if os.path.exists("loss_log.csv"):
    s1 = []
    v1 = []
    lines = open("loss_log.csv").read().strip().split("\n")[1:]
    for ln in lines:
        parts = ln.split(",")
        s1.append(int(parts[0]))
        v1.append(float(parts[2]))
    axes[0].plot(s1, v1, "--", color="gray", label="v1 val")

axes[0].set_xlabel("step")
axes[0].set_ylabel("cross entropy loss")
axes[0].set_ylim(0.5, 2.5)     # the first point is ~4.5 and squishes everything else
axes[0].set_title("loss")
axes[0].legend()

axes[1].plot(steps_logged, lr_hist, color="green")
axes[1].set_xlabel("step")
axes[1].set_ylabel("learning rate")
axes[1].set_title("learning rate (warmup + cosine)")

plt.tight_layout()
plt.savefig("loss_curve_v2.png", dpi=150)
print("saved loss_curve_v2.png")
