# Telegram channel to image dataset

`tg_dataset.py` turns the photos of a Telegram channel into an image dataset with one caption file per image, ready for captioning tools such as TagGUI and for trainers such as Ostris AI Toolkit.

## Why

Many channels post old photographs with a short editor's caption: the place, the year, the people. That caption is good context for a vision-language model that writes the final training caption, but it arrives full of noise: channel signatures, "subscribe" calls, promo posts, reader questions ("Узнали место?"), long essays, and letters swapped for lookalike Latin characters to stop copying. Albums carry the caption on one photo only. This script removes the noise and pairs each photo with a clean caption.

## How

1. In Telegram Desktop, export the channel history with photos, format "Machine-readable JSON". You get a folder with `result.json` and `photos/`.
2. Run the script on that folder. It needs only Python 3, no extra packages.

```
python tg_dataset.py D:\data\ChatExport
python tg_dataset.py D:\data\ChatExport --dry-run
```

The dataset goes to `D:\data\ChatExport\dataset`: each photo is copied as `<message id>.jpg`, with its caption in `<message id>.txt`, and `manifest.jsonl` records the original text and every decision for each photo. `--dry-run` writes nothing and prints statistics and samples, so you can check the result first. If the photos folder was renamed, for example after watermark removal, the script finds the only `photos*` folder, or you can pass `--photos <folder>`.

A second run into the same folder rewrites every `.txt` and copies only the images that are not there yet. It deletes nothing.

## What it does to a caption

- Removes the channel footer: the channel name, links, and "subscribe" or "we are in MAX" lines.
- Drops sentences that talk about the channel or other posts, and reader questions. A caption that is only a question is kept as it is.
- Maps lookalike Latin and Greek letters in Cyrillic words back to Cyrillic, and removes emoji.
- Copies an album's caption to every photo of the album.
- Keeps whole paragraphs up to 80 words, and ends at a sentence end.
- Writes the caption as one line.

Promo posts (ads, giveaways, cross-promotion) cannot be recognized reliably by rules. They are listed by message id per channel in `PROMO_IDS` in the script. For a new channel, run `--dry-run`, check the posts it flags, and add their ids.
