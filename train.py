# step 2 + 3: build the tiny model and pretrain it with next character prediction
#
# usage:
#   python train.py          full run, meant for a GPU (colab T4 takes a few minutes)
#   python train.py quick    much smaller run, just to check that everything works on a laptop
#
# saves:
#   ckpt.pt          the trained model + the character list (chat.py needs this)
#   loss_log.csv     train/val loss over time
#   loss_curve.png   plot of the above
#   samples.txt      what the model writes at different points during training

import sys
import time
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import TinyGPT

# ---------------- settings ----------------
batch_size = 64
block_size = 256      # how many characters the model sees at once
n_embd = 192
n_head = 6
n_layer = 4
dropout = 0.1
lr = 1e-3
max_iters = 6000
eval_every = 250
eval_batches = 40
sample_at = [0, 250, 1000, 3000]    # also writes a sample at the very end

if len(sys.argv) > 1 and sys.argv[1] == "quick":
    print("quick mode, tiny settings")
    batch_size = 16
    block_size = 64
    n_embd = 64
    n_head = 4
    n_layer = 2
    max_iters = 600
    eval_every = 100
    eval_batches = 10
    sample_at = [0, 300]

torch.manual_seed(1337)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

# ---------------- data ----------------
f = open("data/stories.txt")
text = f.read()
f.close()
print("characters in dataset:", len(text))

# the vocabulary is just every character that shows up
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
val_data = data[n:]     # last 10% is never trained on, just used to check for overfitting
print("train chars:", len(train_data), " val chars:", len(val_data))


def get_batch(split):
    d = train_data if split == "train" else val_data
    # pick random starting points, y is just x shifted over by one character
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
    f = open("samples.txt", "a")
    f.write("=" * 20 + " step " + str(step) + " " + "=" * 20 + "\n")
    f.write(s + "\n\n")
    f.close()
    print("--- sample at step", step, "---")
    print(s)
    print("---")


# ---------------- model ----------------
model = TinyGPT(vocab_size, block_size, n_embd, n_head, n_layer, dropout).to(device)
n_params = 0
for p in model.parameters():
    n_params += p.numel()
print("number of parameters:", n_params)

opt = torch.optim.AdamW(model.parameters(), lr=lr)

# start fresh log files every run
open("samples.txt", "w").close()
log = open("loss_log.csv", "w")
log.write("step,train_loss,val_loss\n")

steps_logged = []
train_hist = []
val_hist = []
t0 = time.time()

for it in range(max_iters + 1):

    # lower the learning rate for the last part of training, helps it settle
    if it == int(max_iters * 0.8):
        for g in opt.param_groups:
            g["lr"] = lr / 10
        print("lowering learning rate to", lr / 10)

    if it % eval_every == 0 or it == max_iters:
        losses = estimate_loss()
        print("step %5d | train loss %.4f | val loss %.4f | %.0fs"
              % (it, losses["train"], losses["val"], time.time() - t0))
        log.write("%d,%.4f,%.4f\n" % (it, losses["train"], losses["val"]))
        log.flush()
        steps_logged.append(it)
        train_hist.append(losses["train"])
        val_hist.append(losses["val"])

    if it in sample_at or it == max_iters:
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

# ---------------- save ----------------
config = {
    "vocab_size": vocab_size,
    "block_size": block_size,
    "n_embd": n_embd,
    "n_head": n_head,
    "n_layer": n_layer,
    "dropout": dropout,
}
torch.save({"model": model.state_dict(), "chars": chars, "config": config}, "ckpt.pt")
print("saved ckpt.pt")

plt.figure(figsize=(6, 4))
plt.plot(steps_logged, train_hist, label="train")
plt.plot(steps_logged, val_hist, label="val")
plt.xlabel("step")
plt.ylabel("cross entropy loss")
plt.title("training a tiny GPT on TinyStories")
plt.legend()
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=150)
print("saved loss_curve.png")
