# Judge

`judge.py` keeps only the images of a dataset that meet a condition you state in words, such as "the photograph shows a bus made in the USSR" or "the caption or the picture is about Soviet history", and moves the rest aside. It reads the captions with a local LLM, scores every image with SigLIP 2 against prompts made from the condition, and shows only the doubtful images to a local vision model, one at a time, so a folder of tens of thousands of images costs hours, not days.

It generalises `scripts\unrelated.py`, whose topic is fixed (the Soviet Union), to any condition, and adds the images. Everything it prints, and its `--help`, is in Russian: it was written for a partner who runs it through a local agent, after [vk_dataset](../vk_dataset/README.md).

## Install

Run `install.bat`. Judge installs nothing of its own: it needs what [classify](../classify/README.md) installs (torch with CUDA, transformers, numpy, Pillow) and classify's SigLIP 2 model, so run `classify\install.bat` first, or the `install.bat` in the repository root. `install.bat` checks that classify's snapshot in `classify\models\siglip2-so400m-patch16-naflex` holds both halves of the model, the image tower that classify uses and the text tower that judge needs, and downloads the snapshot through classify into that same folder when it is missing. There is one copy of the model on disk.

The LLM and the vision model run on any OpenAI-compatible server: KoboldCpp at `http://localhost:5001/v1/` is the default, as for `unrelated.py`; Ollama is `http://127.0.0.1:11434/v1/`. The environment variable `JUDGE_ENDPOINT`, when set, replaces the default, so `set JUDGE_ENDPOINT=http://127.0.0.1:11434/v1/` spares `--endpoint` on a machine that runs Ollama. Without `--model` the first model the server lists is used. One model that reads both text and images, such as a Qwen vision model, serves all three layers.

## Usage

```
run.bat <folder> --condition "<text>" [--images] [--no-captions] [--vlm-band LO HI] [--vlm-max N]
        [--positives "<prompt>" ...] [--negatives "<prompt>" ...] [--endpoint URL] [--model NAME]
        [--batch 10] [--timeout 600] [--sheets] [--limit N] [--dry-run] [--rejudge]
run.bat <folder> --undo
```

Examples:

```
run.bat <base>\soviet_buses\<group folder> --condition "на фото советский автобус или троллейбус" --endpoint http://127.0.0.1:11434/v1/ --dry-run
run.bat <base>\soviet_buses\<group folder> --condition "на фото советский автобус или троллейбус" --images --sheets --endpoint http://127.0.0.1:11434/v1/
run.bat <base>\soviet_buses\<group folder> --undo
```

`<folder>` is scanned with all its subfolders. An item is every image, with its caption when a `.txt` of the same name lies next to it. Folders whose names start with a dot, the output folders of the dataset tools (`_embeddings`, `_duplicates`, `_unrelated` and the like) and folders that hold the `_dataset_tools.json` mark are skipped. `run.bat` pauses at the end when started by double-click or when the tool fails; set `NOPAUSE=1` to prevent this.

## The three layers

The layers run one after the other over the whole folder, cheapest first.

1. **Captions** (on by default, `--no-captions` turns them off). Every caption goes to the LLM in batches of `--batch`, with the condition, and comes back YES, NO or INDETERMINATE: the batching, the one-letter reply and its parsing of `unrelated.py`. A batch whose reply cannot be read is asked once more; a caption still unanswered counts as INDETERMINATE.
2. **SigLIP 2** (`--images`). Every image is embedded once with classify's encoder (its decode pool and its cache in `<folder>\_embeddings`, which classify shares), and scored against the prompts: the condition itself plus three short positive and three negative English prompts that the LLM writes once from the condition. The score is the softmax probability of the positive prompts against all of them, between 0 and 1. `--positives` and `--negatives` replace the written prompts (one prompt per flag, repeatable); the condition always stays among the positives.
3. **The VLM** (`--images`). The images scored inside the band `--vlm-band LO HI` (0.35 and 0.65 by default) go to the vision model one by one, downscaled to 1024 pixels on the long side, with the condition and the caption, and the model answers with one letter: Y, N or I. The band is visited by rank, nearest to 0.5 first, up to `--vlm-max` images (300 by default); a reply that is not one of the letters counts as I.

How they combine for one image:

| caption | image score | result |
|---|---|---|
| YES | any | accepted |
| NO | under HI, or `--images` off | rejected |
| NO | at or above HI | to the VLM |
| INDETERMINATE or none | `--images` off | stays, undecided |
| INDETERMINATE or none | at or above HI | accepted |
| INDETERMINATE or none | at or below LO | rejected |
| INDETERMINATE or none | between LO and HI | to the VLM |
| to the VLM | | Y accepted, N rejected, anything else stays; over `--vlm-max` stays |

Only a rejected image moves. An image that stays undecided is never moved.

SigLIP's scores are steep: an image that clearly shows the condition or clearly does not lands near 1 or near 0, so the band holds the genuinely doubtful images. Read the band size in the report before you raise `--vlm-max`: a vision model on a consumer GPU sees about 100 to 250 images per hour, while SigLIP scores tens of images per second.

## The GPU

The layers never share the card. The captions run first; then the SigLIP pass over every image, after which the encoder is deleted and the CUDA cache emptied; then the VLM band. Ollama keeps a model in video memory for some minutes after its last answer, so before the SigLIP pass judge asks Ollama to unload it (Ollama's own `/api/generate` with an empty prompt and `keep_alive` 0); the VLM layer loads it again, which takes a few seconds. KoboldCpp, LM Studio and llama.cpp cannot be asked: there, make sure that SigLIP 2 (about 2.3 GB of weights, plus a batch of 64 images) fits beside the loaded model, or stop the server during the pass. Keep one model in Ollama for the whole run and other GPU tools stopped: a 27B vision model takes most of a 24 GB card.

## Output

```
<folder>\                        the accepted and the undecided images stay here
<folder>\_embeddings\            classify's embedding cache, shared with classify
<folder>_rejected\               the rejected images, each with every file of its name, under their relative paths
  _dataset_tools.json            the mark that keeps the dataset tools from scanning this folder
  _judge\
    verdicts.csv                 one row per image
    prompts.json                 the condition, the prompts and where they came from, the band, the cap
    moves.jsonl                  the undo log
    report.txt                   the report of the last run
    accepted.jpg, rejected.jpg   contact sheets of up to 24 accepted and rejected images (--sheets)
    band.jpg                     the images the VLM saw or the cap left out, nearest to 0.5 first (--sheets)
```

`verdicts.csv` (UTF-8 with BOM) has `path`, `caption_verdict` (YES, NO, INDETERMINATE, UNANSWERED or empty), `image_score`, `vlm_verdict` (Y, N, I or empty), `final` (YES, NO or INDETERMINATE), `reason`, `moved` (1, `undone` or empty) and `caption`. The reasons: `caption_yes`, `caption_no`, `image_high`, `image_low`, `vlm_yes`, `vlm_no`, `vlm_other`, `vlm_cap` (in the band, left out by `--vlm-max`), `undecided` (no caption verdict and no image layer), `no_score` (the image could not be read).

The contact sheets come from vk_dataset's sheet module (`vk_dataset\vkds\sheet.py`, next to this folder); each tile is captioned with its score, its letter and the start of its caption.

The report at the end counts each layer's verdicts and the final ones, gives the band size, how many images the VLM saw and how many the cap left, the rates (images per second for SigLIP, seconds per image for the VLM), and the folders written. The same lines go to `_judge\report.txt`, overwritten by every run, so whoever did not watch the run (a person, or an agent after a person ran it in a terminal) can read its outcome.

## Running again

A rerun reads `verdicts.csv`. The images already accepted or rejected are skipped; a rejected image that a `--dry-run` left in place is moved now, without asking any model again. The other images reuse what is stored for them, a caption verdict, a score or a VLM answer, and only the missing layers run, so an interrupted run continues and a rerun with a larger `--vlm-max` shows the vision model only the rest of the band. New images in the folder are judged in full.

A rerun with another condition is refused while `verdicts.csv` holds verdicts for the first one: `--rejudge` judges everything afresh, `--undo` first puts the moved files back. The prompts of a condition are kept in `prompts.json` and reused, so the scores of two runs agree.

`--dry-run` runs every layer and writes `verdicts.csv`, `prompts.json` and the sheets, and moves nothing. `--undo` moves every file of `moves.jsonl` back to its place (a file whose old place is taken again is left and reported) and marks the rows `undone`, so a later run does not move them again unless `--rejudge`.

## Where it fits

For a [vk_dataset](../vk_dataset/README.md) download, judge with a condition replaces step 4 of the sequence there: with the captions alone (the default) it is `scripts\unrelated.py` with your condition in place of the fixed topic, and with `--images` it also judges the photos without captions. `unrelated.py` stays for its fixed topic. Run it after `deduplicate`, on the raw group folder, before the pastvu import. The Russian manual of the partner who runs vk_dataset through a local agent, `vk_dataset\partner\README.md`, walks through that scenario step by step: the person starts judge in a terminal, the agent reads `report.txt` afterwards.

Judge is agnostic about the condition. The era of a photograph for a historic collection stays the business of the pastvu import and the classify samples folder.

## Options

| option | default | meaning |
|---|---|---|
| `--condition` | | the condition, in any language |
| `--captions`, `--no-captions` | on | the caption layer |
| `--images` | off | the SigLIP 2 and VLM layers |
| `--vlm-band LO HI` | 0.35 0.65 | at or above HI accepts, at or below LO rejects, between goes to the VLM |
| `--vlm-max N` | 300 | how many images of the band the VLM sees |
| `--positives`, `--negatives` | written by the LLM | SigLIP prompts, one per flag |
| `--endpoint` | `JUDGE_ENDPOINT`, else `http://localhost:5001/v1/` | the OpenAI-compatible server |
| `--model` | the first one listed | the model on the server |
| `--batch` | 10 | captions per request |
| `--timeout` | 600 | seconds to wait for one reply |
| `--sheets` | off | the contact sheets |
| `--limit N` | all | only the first N images, for a trial |
| `--dry-run` | off | judge everything, move nothing |
| `--rejudge` | off | judge again what `verdicts.csv` already decided |
| `--undo` | | put the moved files back |
| `--check-models` | | check the SigLIP 2 snapshot (used by `install.bat`) |

Exit codes: `0` done, `1` the model server did not answer, `2` a refusal (no folder, no condition, another condition on the same folder, the model missing), `130` interrupted.

## Project files

| file | purpose |
|---|---|
| `judge.py` | the tool |
| `run.bat` | runs it from the shared venv, offline for the models |
| `install.bat` | checks classify's dependencies and its SigLIP 2 snapshot |
| `requirements.txt` | empty: classify brings everything |
