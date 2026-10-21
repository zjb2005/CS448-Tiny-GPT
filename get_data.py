# step 1: download the dataset and turn it into one big text file
#
# dataset: TinyStories  (https://huggingface.co/datasets/roneneldan/TinyStories)
# short stories written in simple english, the kind of thing a 3-4 year old
# would understand. the vocabulary is tiny so even a really small model
# can learn to write something that looks like english.
#
# the full train split is ~2GB which is way more than we need, so we only
# use their validation split (~22k stories) and make our own train/val split
# from it later in train.py

import os
from datasets import load_dataset

print("downloading TinyStories (validation split)...")
ds = load_dataset("roneneldan/TinyStories", split="validation")
print("got", len(ds), "stories")

# the stories have fancy quotes and dashes in them. turn those into plain
# ascii so we don't waste vocab slots on 5 different kinds of apostrophe
fix = {
    "’": "'", "‘": "'",
    "“": '"', "”": '"',
    "—": "-", "–": "-",
    "…": "...",
}

stories = []
too_short = 0
for row in ds:
    s = row["text"].strip()
    for k in fix:
        s = s.replace(k, fix[k])

    # keep only normal printable characters + newlines
    keep = []
    for ch in s:
        if ch == "\n" or (32 <= ord(ch) < 127):
            keep.append(ch)
    s = "".join(keep)

    if len(s) < 100:
        too_short += 1
        continue
    stories.append(s)

print("kept", len(stories), "stories, dropped", too_short, "that were too short or empty")

# blank line between stories so the model can see where one ends
text = "\n\n".join(stories)

if not os.path.exists("data"):
    os.mkdir("data")
f = open("data/stories.txt", "w")
f.write(text)
f.close()

chars = sorted(set(text))
print("total characters:", len(text))
print("unique characters:", len(chars))
print("".join(chars))
print()
print("first story looks like this:")
print(stories[0])
