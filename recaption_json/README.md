# recaption_json

Adds a second caption to every image of a dataset: a short prompt in the form a user types into an image model, made by a local LLM from the long caption the image already has. The long caption stays in `<stem>.txt`; the short prompt goes to `<stem>.short` next to it. A trainer that reads any sidecar extension as a caption (Ostris AI Toolkit) can then train the same images on both, so the model learns the short prompts users write as well as the long reference captions.

```
D:\photos\street\0001.jpg
D:\photos\street\0001.txt     A black-and-white photograph from the 1960s in Leningrad of a GAZ-21 Volga parked on a city street. Two men in dark coats ...
D:\photos\street\0001.short   A GAZ-21 Volga parked on a city street. 1960s, Leningrad, black-and-white photo.
```

The short prompt is written the way a brief but competent user writes one: it only describes what the picture shows (subject, era, color mode or medium, often a place), with the long caption's own words and facts and nothing added. It is never a request ("Generate a ...", "I need a ...", "please").

## Install

Run `install.bat`. It puts `httpx` into the shared venv of this repository (`..\venv`), creating the venv when it is missing.

The LLM runs elsewhere, in any server with an OpenAI-compatible chat endpoint: KoboldCpp, llama.cpp server, LM Studio or a cloud API. The default endpoint is KoboldCpp's, `http://localhost:5001/v1/`. Load a text or vision-language model there before you start; this tool sends text only. Reasoning is switched off in each request.

## Usage

`run.bat` passes every argument to `recaption.py`; from a shell with the venv active, `python recaption.py` takes the same arguments.

Write a `.short` for every captioned image of a dataset, ten captions per request:

```
run.bat D:\photos
```

See what would be done first: the number of items per style and the first request, with no request sent and no file written:

```
run.bat D:\photos --dry-run
```

Try the first 20 items, then look at the `.short` files before the full run:

```
run.bat D:\photos --limit 20
```

Continue a run that was stopped (Ctrl+C, a crash, the LLM server going down): start the same command again. Items that have a `.short` are skipped:

```
run.bat D:\photos
```

Write every `.short` again, for example after a change to the prompt:

```
run.bat D:\photos --overwrite
```

Redo some items only: delete their `.short` files and run without `--overwrite`.

Keep the log next to the dataset instead of in the current folder:

```
run.bat D:\photos --log D:\photos_recaption_log.csv
```

More tag lists, fewer plain phrases:

```
run.bat D:\photos --styles A=35,B=50,C=15
```

Another server or model, one caption per request (for a small model that loses track in a batch):

```
run.bat D:\photos --endpoint http://192.168.1.20:1234/v1/ --model qwen3-14b --batch 1
```

Long captions in another sidecar extension:

```
run.bat D:\photos --ext .caption
```

A run prints one line per batch and a summary:

```
236 items in D:\photos, 0 with a .short already, 236 to do (A 93, B 84, C 59), batch 10
model koboldcpp/gemma-4-31B-it, log D:\photos_recaption_log.csv
[ 10/236] 9 ok, 1 retried, 0 failed, 1.1 s/item
[ 20/236] 10 ok, 0 retried, 0 failed, 0.9 s/item
...
236 items in 262 s (1.1 s/item): 205 ok on the first try, 31 after a retry, 0 failed, 0 skipped
```

The exit code is 0 when every item was processed (some may have failed; see the log) and 1 when the run stopped early.

## Options

| option | default | what it does |
|---|---|---|
| `--ext` | `.txt` | the extension of the long captions |
| `--endpoint` | `http://localhost:5001/v1/` | the OpenAI-compatible API |
| `--model` | `auto` | the model name; `auto` takes the first model the endpoint lists |
| `--batch` | `10` | items per request; `1` asks for each item on its own |
| `--styles` | `A=40,B=35,C=25` | the share of each style; `C=0` turns a style off |
| `--limit` | all | at most this many items without a `.short` |
| `--overwrite` | off | write a new `.short` where one exists |
| `--dry-run` | off | count the items per style and print the first request; no requests, no files |
| `--log` | `recaption_log.csv` | the CSV log, appended to; a relative path is relative to the current folder |
| `--temperature` | `0.3` | the sampling temperature |

## Items and sidecars

The dataset folder is searched recursively. An item is every `<stem>.txt` (or `--ext`) with an image of the same stem next to it (`.jpg`, `.jpeg`, `.png`, `.webp`); a caption without an image is left alone, and `_controls` mask folders are not searched. An item with a non-empty `<stem>.short` is skipped unless `--overwrite` is given.

`<stem>.short` holds the short prompt only: one line of UTF-8 text with no BOM and no line break at the end. The `.txt` file is never changed. The long captions may be UTF-8 or Windows-1251.

## Styles

Every item gets one style letter from a hash of its file name, so a rerun gives the same letter, and over a dataset the letters follow the `--styles` shares. The styles are defined in `prompt/recaption.txt`:

| style | form | example |
|---|---|---|
| A | subject phrase, then tags with era, place and color mode; 8 to 20 words | `A GAZ-21 Volga parked on a city street. 1960s, Soviet Union, black-and-white photo.` |
| B | 6 to 12 lowercase comma-separated tags, era and color mode last | `gaz-21 volga, parked cars, ornate stone building, pedestrians, soviet street, 1960s, black and white photo` |
| C | one plain descriptive phrase; 8 to 20 words | `old black and white photo of kids riding pedal cars in a park, Soviet Union 1950s` |

There is no style D: a request style ("Generate a 1960s photo of ...") was dropped, because a short prompt only describes the picture. There is no style E: a Russian style was dropped, because every short prompt is in English; Cyrillic appears only in a sign's text quoted as the long caption gives it.

## Checks

Every short prompt the model returns is checked before it is written:

- length: A and C 8 to 20 words; B 6 to 12 comma-separated tags of at most 5 words each;
- English only: no Cyrillic outside quotes (a quoted sign's text may keep its alphabet);
- an era: a decade (`1960s`, `60s`, `mid-1960s`), a year, `20th century`, or `Soviet era`;
- a color mode that matches the opening of the long caption: a black-and-white form for "A black-and-white ...", a sepia or black-and-white form for a sepia photograph, a color form for a color photograph, `film still` or `movie still` for a film still. A painting, an illustration or a drawing names its medium instead (`painting`, `illustration`, `drawing`) and never says photo;
- no color words when the long caption begins "A black-and-white" (light and dark are allowed, and place names such as Red Square are not colors);
- a concept name from the prompt's list only when the long caption meets its condition;
- no template opening ("A color photograph from ...") and none of: this image, this photograph, appears to, seems, likely, possibly, probably;
- no request or instruction: no generate, create, render, give me, gimme, show me, make me, do a, I need, I want, please, can you anywhere, and no imperative verb such as make, draw or show as the first word.

An item that fails a check, or that a batch answer leaves out, is asked for again on its own with the reason added to the request, at most twice. An item that still fails gets no `.short`; its reason is in the log, and the next run tries it again. A request with one item may be answered with the bare short prompt instead of JSON; it is taken and checked like any other.

## The log

One CSV row per item, appended on every run (UTF-8 with a BOM, so Excel shows Cyrillic):

| column | meaning |
|---|---|
| `path` | the long caption, relative to the dataset folder |
| `style` | the style letter |
| `status` | `ok` on the first try, `retried` (passed after a retry), `failed` (no `.short` written), `skipped` (a `.short` exists) |
| `attempts` | requests for this item, 1 to 3 |
| `reason` | for `failed`, why its last answer was rejected; for `retried`, `first try: ` and why its first answer was rejected; empty otherwise |
| `words` | words in the short prompt |
| `concepts` | the verified concept names in the short prompt |
| `mode` | `batch` or `single`, of the first try |
| `seconds` | time spent on the item; a batch's time is shared out between its items |

To see what failed, filter the log on `status` = `failed`; to see which rule the model breaks most, filter on `retried` and sort by `reason`. The summary at the end of a run counts both, by reason.

## Batch or single

`compare_batch.py` asks for the first 100 items of a dataset twice, one per request and ten per request, first try only and without writing any `.short`, and writes `compare_batch.md` next to the script: seconds per item, the first-try pass rate overall and per style, the mean word count per style, the share of items with a verified concept name, cross-item leakage in batch mode (a short prompt that names a concept, vehicle model, place or year its own long caption does not support but another long caption of the same batch does), and 20 items side by side. The raw answers go to `compare_batch.json`. From the `recaption_json` folder:

```
..\venv\Scripts\python.exe compare_batch.py D:\photos
..\venv\Scripts\python.exe compare_batch.py D:\photos --batch 5 --items 50
..\venv\Scripts\python.exe compare_batch.py --report-only
```

On the first dataset (a 31B model in KoboldCpp) ten per request passed 92 % on the first try against 86 % one per request, at 0.96 against 1.14 seconds per item, with no leakage; hence the default `--batch 10`. A request of one item is hardly slower because KoboldCpp keeps the system prompt in its cache. Run the comparison again for another model before you rely on batches.

## AI Toolkit

Two dataset entries on the same folder, one per caption extension:

```yaml
datasets:
  - folder_path: <root>
    caption_ext: txt
    num_repeats: 1
    caption_dropout_rate: 0.05
  - folder_path: <root>
    caption_ext: short
    num_repeats: 1
```

## The prompts

`prompt/recaption.txt` is the system prompt of this tool: the facts rules, the styles, and the concept names with their conditions. The concept table in `recaption.py` (`CONCEPTS`) mirrors that list; change both together. `prompt/prompt.txt` is a captioning prompt for a later tool and is not used here. Both files are read as UTF-8, or as Windows-1251 when they are not UTF-8.
