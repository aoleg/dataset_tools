# recaption_json

Adds a second caption to every image of a dataset: a short prompt in the form a user types into an image model, made by a local LLM from the long caption the image already has. The long caption stays in `<stem>.txt`; the short prompt goes to `<stem>.short` next to it. A trainer that reads any sidecar extension as a caption (Ostris AI Toolkit) can then train the same images on both, so the model learns the short prompts users write as well as the long reference captions.

## Install

Run `install.bat`. It puts `httpx` into the shared venv of this repository (`..\venv`). The LLM runs elsewhere: any OpenAI-compatible chat endpoint (KoboldCpp, llama.cpp server, LM Studio), by default `http://localhost:5001/v1/`.

## Use

```
run.bat <dataset>
run.bat <dataset> --dry-run
run.bat <dataset> --batch 1 --limit 50 --log D:\logs\recaption_log.csv
```

The dataset folder is searched recursively. An item is every `<stem>.txt` with an image of the same stem next to it (`.jpg`, `.jpeg`, `.png`, `.webp`). An item that has a non-empty `<stem>.short` is skipped, so a stopped run continues where it stopped when you start it again; `--overwrite` writes every item again.

| option | default | what it does |
|---|---|---|
| `--ext` | `.txt` | the extension of the long captions |
| `--endpoint` | `http://localhost:5001/v1/` | the OpenAI-compatible API |
| `--model` | `auto` | the model name; `auto` takes the first model the endpoint lists |
| `--batch` | `10` | items per request; `1` asks for each item on its own |
| `--styles` | `A=30,B=25,C=20,D=15,E=10` | the share of each style; `E=0` turns the Russian style off |
| `--limit` | all | at most this many items without a `.short` |
| `--overwrite` | off | write a new `.short` where one exists |
| `--dry-run` | off | count the items per style and print the first request; no requests, no files |
| `--log` | `recaption_log.csv` | the CSV log, appended to; a relative path is relative to the current folder |
| `--temperature` | `0.3` | the sampling temperature |

## The sidecar

`<stem>.short` holds the short prompt only: one line of UTF-8 text with no BOM and no line break at the end. The `.txt` file is never changed.

## Styles

Every item gets one style letter from a hash of its file name, so a rerun gives the same letter and the shares follow `--styles`. The styles are defined in `prompt/recaption.txt`: A is a subject phrase with trailing tags, B a comma-separated tag list, C a casual sentence, D a request ("Generate a ... photo of ..."), E style A or C in Russian.

## Checks

Every short prompt the model returns is checked before it is written:

- length: A, C and D 8 to 20 words; B 6 to 12 comma-separated tags of at most 5 words each; E 6 to 20 words, at least 80 % of the letters Cyrillic;
- an era: a decade (`1960s`, `60s`, `mid-1960s`), a year, `20th century`, or `Soviet era`; for E also the Russian forms (`1960-е`, `1960-х`);
- a color mode that matches the opening of the long caption: a black-and-white form for "A black-and-white ..." (and sepia), a color form for a color photograph, `film still` or `movie still` for a film still; Russian forms for E. A painting, an illustration or a drawing has no color mode in the prompt, so nothing is required for it;
- no color words when the long caption begins "A black-and-white" (light and dark are allowed, and place names such as Red Square are not colors);
- a concept name from the prompt's list only when the long caption meets its condition;
- no template opening ("A color photograph from ...") and none of: this image, this photograph, appears to, seems, likely, possibly, probably.

An item that fails a check, or that a batch answer leaves out, is asked for again on its own with the reason added to the request, at most twice. An item that still fails gets no `.short`; its reason is in the log.

## The log

One CSV row per item: `path`, `style`, `status` (`ok` on the first try, `retried`, `failed`, `skipped`), `attempts`, `reason`, `words`, `concepts` (the verified concept names in the short prompt), `mode` (`batch` or `single`, of the first try) and `seconds` (a batch's time is shared out between its items). A progress line is printed per batch and a summary at the end.

## Batch or single

`compare_batch.py <dataset>` asks for the first 100 items twice, one per request and ten per request, first try only and without writing any `.short`, and writes `compare_batch.md`: seconds per item, the first-try pass rate overall and per style, the mean word count per style, the share of items with a verified concept name, and cross-item leakage in batch mode (a short prompt that names a concept, vehicle model, place or year its own long caption does not support but another long caption of the same batch does). `--batch 5` compares another size and `--report-only` rebuilds the report from `compare_batch.json`.

On the first dataset (a 31B model in KoboldCpp) ten per request passed 92 % on the first try against 86 % one per request, at 0.96 against 1.14 seconds per item, with no leakage; hence the default `--batch 10`. A request of one item is hardly slower because KoboldCpp keeps the system prompt in its cache.

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

`prompt/recaption.txt` is the system prompt of this tool. `prompt/prompt.txt` is a captioning prompt for a later tool and is not used here. Both files are read as UTF-8, or as Windows-1251 when they are not UTF-8.
