# step 4: talk to the trained model
# this is what we run for the demo. works on a laptop cpu too, the model is tiny.
#
#   python chat.py              uses ckpt.pt (version 1)
#   python chat.py ckpt_v2.pt   uses the bigger version 2 model
#
# type a prompt and it keeps writing from there.
# commands:
#   temp 0.5   -> change temperature (lower = safer and more boring, higher = more random)
#   len 500    -> change how many characters it writes
#   quit       -> exit

import sys
import torch
from model import TinyGPT

device = "cuda" if torch.cuda.is_available() else "cpu"

ckpt_path = "ckpt.pt"
if len(sys.argv) > 1:
    ckpt_path = sys.argv[1]
print("loading", ckpt_path)
ckpt = torch.load(ckpt_path, map_location=device)
cfg = ckpt["config"]
chars = ckpt["chars"]
stoi = {}
itos = {}
for i, ch in enumerate(chars):
    stoi[ch] = i
    itos[i] = ch

model = TinyGPT(cfg["vocab_size"], cfg["block_size"], cfg["n_embd"],
                cfg["n_head"], cfg["n_layer"], cfg["dropout"])
model.load_state_dict(ckpt["model"])
model.to(device)
model.eval()

n_params = 0
for p in model.parameters():
    n_params += p.numel()
print("loaded model with", n_params, "parameters, vocab of", len(chars), "characters")

temperature = 0.8
length = 400

print("type a prompt. commands: 'temp 0.5', 'len 500', 'quit'")
while True:
    try:
        prompt = input("\nprompt> ")
    except EOFError:
        break

    if prompt.strip() == "quit":
        break
    if prompt.startswith("temp "):
        temperature = float(prompt.split()[1])
        print("temperature is now", temperature)
        continue
    if prompt.startswith("len "):
        length = int(prompt.split()[1])
        print("will write", length, "characters")
        continue

    # the model only knows characters it saw during training
    ids = []
    for c in prompt:
        if c in stoi:
            ids.append(stoi[c])
        else:
            print("(skipping character the model has never seen: %r)" % c)
    if len(ids) == 0:
        ids = [stoi["\n"]]    # empty prompt, just let it start from a newline

    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(x, length, temperature=temperature, top_k=20)
    text = "".join([itos[i] for i in out[0].tolist()])
    print()
    print(text)
