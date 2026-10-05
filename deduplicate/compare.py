#!/usr/bin/env python3
"""
Show which files dedup.py judged copies of which, for a check by eye.

Reads <folder>/_duplicates/plan.json (the same plan is in every folder of a
run) and writes to <folder>/_duplicates/compare/:

  pairs.txt           one block per group: the kept file, then every moved
                      copy with where it is now, the match kind and its size
  NNNN_<kept>.jpg     one sheet per group: the kept copy and every moved copy
                      side by side, labelled

Usage:    python compare.py <folder> [--skip-exact] [--only hash|features]
                            [--height N] [--out DIR] [--review]
  --skip-exact   leave out groups whose moved copies are all byte-identical to
                 the kept copy; there is nothing to look at there
  --only KIND    only groups with at least one moved copy of this match kind;
                 "features" is the kind worth checking first (a border, crop,
                 frame or footer differed)
  --height N     tile height in pixels (default 360)
  --out DIR      write somewhere else than <folder>/_duplicates/compare

Nothing is moved or changed. Moved copies are read from where they are now.
"""
import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

Image.MAX_IMAGE_PIXELS = None


def tile(path: Path, label: str, height: int) -> Image.Image:
    try:
        im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        w, h = im.size
        im = im.resize((max(1, int(w * height / h)), height), Image.LANCZOS)
    except Exception as e:  # noqa: BLE001 - a missing or unreadable file gets a grey tile
        im = Image.new("RGB", (height * 3 // 4, height), (90, 90, 90))
        label += f"  [{type(e).__name__}]"
    out = Image.new("RGB", (max(im.width, 220), height + 34), "white")
    out.paste(im, (0, 34))
    d = ImageDraw.Draw(out)
    y = 2
    for line in (label[:48], label[48:96]):
        if line:
            d.text((3, y), line, fill="black")
            y += 15
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="a folder of the run; its _duplicates/plan.json is read")
    ap.add_argument("--skip-exact", action="store_true")
    ap.add_argument("--only", choices=["hash", "features"])
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--out")
    ap.add_argument("--review", action="store_true", help="then open the review tool (review.py) on the groups")
    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except AttributeError:
            pass

    folder = Path(args.folder).resolve()
    plan_path = folder / "_duplicates" / "plan.json"
    if not plan_path.is_file():
        print(f"no plan: {plan_path}")
        return 1
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    out = Path(args.out).resolve() if args.out else folder / "_duplicates" / "compare"
    out.mkdir(parents=True, exist_ok=True)

    groups = []
    for g in plan["groups"]:
        moves = g.get("move") or g.get("members") or []
        if not moves:
            continue
        kinds = {m.get("match") for m in moves}
        if args.skip_exact and all(m.get("exact") for m in moves):
            continue
        if args.only and args.only not in kinds:
            continue
        groups.append((g, moves))
    # feature matches first: they are the ones to check
    order = {"features": 0, "through other copies": 1, "hash": 2, "exact": 3}
    groups.sort(key=lambda gm: (min(order.get(m.get("match"), 9) for m in gm[1]), gm[0]["keep"]["path"].casefold()))

    lines = [f"plan written {plan.get('written')}, match {plan.get('match')}; {len(groups)} group(s) listed, "
             f"{len(plan['groups'])} in the plan", ""]
    for n, (g, moves) in enumerate(groups, 1):
        k = g["keep"]
        what = "kept" if "move" in g else "best copy (group left for review)"
        lines.append(f"[{n:04d}] {what}: {k['path']}")
        lines.append(f"       {k.get('w')}x{k.get('h')}  tier {k.get('tier')}  score {k.get('score')}  reason: {g.get('reason')}"
                     + (f"  REVIEW: {g['review']}" if "review" in g else ""))
        tiles = [tile(Path(k["path"]), f"KEPT {Path(k['path']).name} {k.get('w')}x{k.get('h')} t{k.get('tier')} {k.get('score')}", args.height)]
        for m in moves:
            now = Path(m.get("to") or m["path"])
            src = now if now.exists() else Path(m["path"])
            kind = m.get("match", "?") + (" (identical)" if m.get("exact") else "")
            lines.append(f"       copy: {m['path']}")
            lines.append(f"             now at: {m.get('to', '(not moved)')}")
            lines.append(f"             {m.get('w')}x{m.get('h')}  tier {m.get('tier')}  score {m.get('score')}  match: {kind}"
                         + (f"  phash {m.get('phash_distance')} dhash {m.get('dhash_distance')}" if "phash_distance" in m else ""))
            tiles.append(tile(src, f"{'moved' if 'move' in g else 'copy'} {Path(m['path']).name} {m.get('w')}x{m.get('h')} t{m.get('tier')} {m.get('score')} {kind}", args.height))
        lines.append("")
        W = sum(t.width for t in tiles) + 8 * (len(tiles) - 1)
        sheet = Image.new("RGB", (W, args.height + 34), "red")
        x = 0
        for t in tiles:
            sheet.paste(t, (x, 0))
            x += t.width + 8
        if sheet.width > 3000:
            sheet = sheet.resize((3000, int(sheet.height * 3000 / sheet.width)), Image.LANCZOS)
        stem = "".join(c if c.isalnum() or c in "-_." else "_" for c in Path(k["path"]).stem)[:40]
        sheet.save(out / f"{n:04d}_{stem}.jpg", quality=85)
    (out / "pairs.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"{len(groups)} group(s): {out / 'pairs.txt'} and {len(groups)} sheet(s) in {out}")
    if args.review:
        import subprocess
        return subprocess.call([sys.executable, str(Path(__file__).with_name("review.py")), str(folder)])
    return 0


if __name__ == "__main__":
    sys.exit(main())
