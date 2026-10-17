@echo off
:: ---------------------------------------------------------------------------
:: Sample query against coyo-hd-11m-llavanext (CaptionEmporium).
::
:: Every option is listed, so edit this file instead of memorising them.
:: "*" means "not set" and switches that filter off.
::
:: Terms are substrings and every term must match. "|" separates alternatives
:: ("lighthouse|beacon"), "+" stands for a space ("stone+bridge").
:: Exclusions are whole words; any one of them drops the row.
::
:: --caption searches the one-sentence LLaVA caption. The other text columns
:: can be searched too: the long caption, the original alt text, and the two
:: tag lists (use full tag names there, e.g. "baseball_cap").
::
:: --caption-col picks which column becomes the .txt caption on download:
:: caption_llava, caption_llava_short, caption, tags_open_images, tags_booru.
:: ---------------------------------------------------------------------------

call "%~dp0..\run.bat" --dataset coyo-hd-11m ^
  --caption "lighthouse" ^
  --caption_llava "*" ^
  --caption_original "*" ^
  --tags_open_images "*" ^
  --tags_booru "*" ^
  --exclude "*" ^
  --exclude-caption_llava "*" ^
  --exclude-caption_original "*" ^
  --exclude-tags_open_images "*" ^
  --exclude-tags_booru "*" ^
  --faces "*" ^
  --min-width "*" ^
  --min-height "*" ^
  --min-aesthetic "*" ^
  --max-watermark "*" ^
  --max-nsfw "*" ^
  --caption-col "caption_llava_short" ^
  --limit "100" ^
  --start "0"

pause
