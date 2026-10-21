# a little web page for talking to the model, nicer than the terminal for the demo
#
#   pip install gradio
#   python web_demo.py           -> open http://127.0.0.1:7860 in the browser
#   python web_demo.py share     -> also gives a public link (use this on colab)
#   python web_demo.py ckpt_v2.pt        -> use the bigger version 2 model
#   python web_demo.py ckpt_v2.pt share  -> both
#
# the text shows up a few characters at a time while the model is writing,
# so you can actually watch it generate

import sys
import torch
import gradio as gr
from model import TinyGPT

device = "cuda" if torch.cuda.is_available() else "cpu"

ckpt_path = "ckpt.pt"
share = False
for a in sys.argv[1:]:
    if a.endswith(".pt"):
        ckpt_path = a
    if a == "share":
        share = True
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


def write_story(prompt, temperature, length, stop_at_end):
    ids = []
    for c in prompt:
        if c in stoi:
            ids.append(stoi[c])
    if len(ids) == 0:
        ids = [stoi["\n"]]
    x = torch.tensor([ids], dtype=torch.long, device=device)

    text = "".join([itos[i] for i in ids])
    for step in range(int(length)):
        x = model.generate(x, 1, temperature=temperature, top_k=20)
        text += itos[x[0, -1].item()]

        # stories in the data are separated by a blank line, so when the model
        # writes one after "The end." it's about to start a brand new story
        if stop_at_end and len(text) > len(prompt) + 50 and text.endswith("\n\n"):
            break

        if step % 5 == 0:
            yield text
    yield text.strip()


demo = gr.Interface(
    fn=write_story,
    inputs=[
        gr.Textbox(label="Start of the story", value="Once upon a time", lines=2),
        gr.Slider(0.1, 1.5, value=0.8, step=0.1, label="Temperature (low = safe, high = wild)"),
        gr.Slider(50, 1000, value=400, step=50, label="How many characters to write"),
        gr.Checkbox(value=True, label="Stop when the story ends"),
    ],
    outputs=gr.Textbox(label="What the model wrote", lines=14),
    examples=[
        ["Once upon a time", 0.8, 400, True],
        ["The little dog", 0.8, 400, True],
        ["Lily wanted to", 0.8, 400, True],
        ["One day, Tom went to the park", 0.3, 400, True],
        ["One day, Tom went to the park", 1.3, 400, True],
        ["What is 2+2?", 0.8, 300, True],
    ],
    title="Tiny Story GPT",
    description="A %.1f million parameter character-level GPT we trained from scratch on TinyStories. "
                "It only knows how to continue stories, it was never taught to answer questions."
                % (n_params / 1e6),
)

demo.launch(share=share)
