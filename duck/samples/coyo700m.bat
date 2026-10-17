@echo off
:: ---------------------------------------------------------------------------
:: Sample query against COYO-700M (kakaobrain/coyo-700m).
::
:: Every option is listed, so edit this file instead of memorising them.
:: "*" means "not set" and switches that filter off.
::
:: Terms are substrings and every term must match. "|" separates alternatives
:: ("lighthouse|beacon"), "+" stands for a space ("stone+bridge").
:: Exclusions are whole words; any one of them drops the row.
::
:: COYO-700M is 126 GB. A full scan takes minutes; try --count or --quick-test
:: first to size the query.
:: ---------------------------------------------------------------------------

call "%~dp0..\run.bat" --dataset coyo700m ^
  --caption "lighthouse" ^
  --exclude "*" ^
  --faces "*" ^
  --min-width "1024" ^
  --min-height "1024" ^
  --min-aesthetic "*" ^
  --max-watermark "*" ^
  --max-nsfw "*" ^
  --min-clip "*" ^
  --limit "100" ^
  --start "0"

pause
