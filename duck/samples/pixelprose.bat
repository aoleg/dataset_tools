@echo off
:: ---------------------------------------------------------------------------
:: Sample query against PixelProse (tomg-group-umd/pixelprose).
::
:: Every option is listed, so edit this file instead of memorising them.
:: "*" means "not set" and switches that filter off.
::
:: Terms are substrings and every term must match. "|" separates alternatives
:: ("lighthouse|beacon"), "+" stands for a space ("stone+bridge").
:: Exclusions are whole words; any one of them drops the row.
::
:: --watermark-class, per the dataset's own documentation:
::     0 = image WITH a watermark
::     1 = image without watermark          <-- "no watermark, no text"
::     2 = image without watermark but WITH text
:: This sample uses 1. Use "1-2" to allow text but still bar watermarks.
::
:: --min-width / --min-height filter the real image size (original_width and
:: original_height), not the resized copy that was fed to the captioning model.
::
:: --split picks the source: all, cc12m, commonpool, redcaps. For high-resolution
:: work redcaps is densest (82% of it is >= 1024x1024) and commonpool is
:: thinnest (3.8%).
:: ---------------------------------------------------------------------------

call "%~dp0..\run.bat" --dataset pixelprose ^
  --split "all" ^
  --caption "lighthouse" ^
  --original_caption "*" ^
  --exclude "*" ^
  --exclude-original_caption "*" ^
  --watermark-class "1" ^
  --max-watermark-score "*" ^
  --min-aesthetic "*" ^
  --min-width "*" ^
  --min-height "*" ^
  --caption-col "vlm_caption" ^
  --strip-preamble "yes" ^
  --limit "100" ^
  --start "0"

pause
