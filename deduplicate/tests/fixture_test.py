"""Fixture test for the sorted-folder rules of dedup.py.

Builds a sorted and an unsorted tree from synthetic pictures (no real dataset
files), runs dedup.py with --sorted, checks every case of the rules, runs again
to confirm that nothing moves, undoes and compares both trees byte for byte with
the originals; then the same for --sorted-copies one. Prints ALL PASS or the
failed checks, exit code 1 on failure.

Usage: python tests/fixture_test.py [fixture dir]      (default: tests/_fixture)

The fixture dir is deleted and rebuilt on every run. Run it with the venv
python, which has Pillow, numpy, imagehash and OpenCV.
"""
import hashlib, json, os, random, shutil, subprocess, sys
from pathlib import Path
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
DEDUP = str(HERE.parent / "dedup.py")
PY = sys.executable
FX = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE / "_fixture"
S, U = FX / "sorted", FX / "unsorted"


def picture(seed, side=1100):
    """A textured synthetic picture: noise background and many shapes, so ORB has keypoints."""
    rnd = random.Random(seed)
    im = Image.new("RGB", (side, side), tuple(rnd.randrange(30, 230) for _ in range(3)))
    dr = ImageDraw.Draw(im)
    # a few big shapes keep the hashes of different pictures apart; many small
    # shapes with corners give ORB its keypoints
    for n, (lo, hi) in ((6, (side // 4, side * 3 // 4)), (90, (15, 120))):
        for _ in range(n):
            x0, y0 = rnd.randrange(-50, side), rnd.randrange(-50, side)
            x1, y1 = x0 + rnd.randrange(lo, hi), y0 + rnd.randrange(lo, hi)
            col = tuple(rnd.randrange(256) for _ in range(3))
            if rnd.random() < 0.6:
                dr.rectangle((x0, y0, x1, y1), fill=col, outline=(0, 0, 0), width=3)
            else:
                dr.ellipse((x0, y0, x1, y1), fill=col, outline=(255, 255, 255), width=3)
    for _ in range(40):
        dr.line([(rnd.randrange(side), rnd.randrange(side)) for _ in range(3)],
                fill=tuple(rnd.randrange(256) for _ in range(3)), width=rnd.randrange(2, 9))
    return im


def save(im, path, side=None, quality=92, border=0.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    if side:
        im = im.resize((side, side), Image.LANCZOS)
    if border:
        b = int(im.width * border)
        canvas = Image.new("RGB", (im.width + 2 * b, im.height + 2 * b), "white")
        canvas.paste(im, (b, b))
        im = canvas
    if path.suffix == ".png":
        im.save(path)
    else:
        im.save(path, quality=quality)


def txt(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build():
    if FX.exists():
        shutil.rmtree(FX)
    P = {n: picture(n) for n in range(1, 15)}
    # 1. slot 1.jpg (P1) and an unrelated 1.png (P9) share a/1.txt; U x.png is a better P1
    save(P[1], S / "a/1.jpg", 600); save(P[9], S / "a/1.png", 600); txt(S / "a/1.txt", "slot one")
    save(P[1], U / "x.png", 1100)
    # 2. same extension, slot caption untouched
    save(P[2], S / "b/2.jpg", 600); txt(S / "b/2.txt", "slot two")
    save(P[2], U / "y.jpg", 1100)
    # 3. slot without caption, incoming with caption
    save(P[3], S / "c/3.jpg", 600)
    save(P[3], U / "z.jpg", 1100); txt(U / "z.txt", "incoming three")
    # 4. both have captions: sorted wins, incoming caption parked
    save(P[4], S / "d/4.jpg", 600); txt(S / "d/4.txt", "slot four")
    save(P[4], U / "w.jpg", 1100); txt(U / "w.txt", "incoming four")
    # 5. same bucket, same bytes re-encoded: within margin or tie, no promotion
    save(P[5], S / "e/5.jpg", 1100, quality=92)
    save(P[5], U / "v.jpg", 1100, quality=93)
    # 6. incoming has a wide border: feature match, review, nothing moves
    save(P[6], S / "f/6.jpg", 600)
    save(P[6], U / "u.jpg", 1100, border=0.12)     # hash distances 18/20: feature match only
    # 12. byte-identical: sorted kept
    save(P[7], S / "g/7.jpg", 800); shutil.copy2(S / "g/7.jpg", U / "t.jpg")
    # 13. three unsorted copies, sorted is best
    save(P[8], S / "h/8.jpg", 1100)
    save(P[8], U / "s1.jpg", 600); save(P[8], U / "s2.jpg", 800); save(P[8], U / "s3.jpg", 400)
    # 11. two slots 9.jpg (P10) and 9.png (P11) in one directory, shared 9.txt, both lose to PNGs
    save(P[10], S / "k/9.jpg", 600); save(P[11], S / "k/9.png", 600); txt(S / "k/9.txt", "slot nine")
    save(P[10], U / "p.png", 1100); save(P[11], U / "q.png", 1100)
    # 7. same picture in two sorted folders, one better; m has no caption, n has one
    save(P[12], S / "m/7a.jpg", 600)
    save(P[12], S / "n/7b.jpg", 1100); txt(S / "n/7b.txt", "cat n")
    # 9. same picture twice in one sorted folder
    save(P[13], S / "o/9a.jpg", 600); save(P[13], S / "o/9b.jpg", 1100)
    # 10. unsorted keeper better than sorted copies in two folders, all captioned
    save(P[14], S / "r/10a.jpg", 600); txt(S / "r/10a.txt", "cat r")
    save(P[14], S / "s/10b.jpg", 800); txt(S / "s/10b.txt", "cat s")
    save(P[14], U / "g.jpg", 1100); txt(U / "g.txt", "incoming g")


def snapshot(root):
    out = {}
    for p in root.rglob("*"):
        if p.is_file() and "_duplicates" not in p.parts:
            out[p.relative_to(root).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def run(*extra):
    cmd = [PY, DEDUP, str(U), "--sorted", str(S), "--workers", "2", *extra]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(r.stdout[-2500:])
    if r.returncode:
        print("STDERR:", r.stderr[-3000:])
    run.stdout = r.stdout
    return r.returncode


def check(label, conds):
    bad = [c for c, ok in conds.items() if not ok]
    print(f"[{'PASS' if not bad else 'FAIL'}] {label}" + ("" if not bad else ": " + "; ".join(bad)))
    return not bad


def ex(p):
    return p.exists()


def rd(p):
    return p.read_text(encoding="utf-8") if p.exists() else None


build()
before = {"S": snapshot(S), "U": snapshot(U)}
print("=== dry run"); assert run("--dry-run") == 0
plan = json.load(open(U / "_duplicates/plan.json", encoding="utf-8"))
for g in plan["groups"]:
    print("   ", os.path.basename(g["keep"]["path"]), "|", g.get("reason"), "|", g.get("review", ""),
          "| others", [os.path.basename(m["path"]) for m in g.get("move", g.get("members", []))],
          "| to", os.path.basename(g["promote"]["to"]) if "promote" in g else "")
print("=== run"); assert run() == 0
ok = True
ok &= check("1 rename on collision", {
    "1.jpg moved out": not ex(S / "a/1.jpg") and ex(S / "_duplicates/a/1.jpg"),
    "moved-out copy has a caption copy": rd(S / "_duplicates/a/1.txt") == "slot one",
    "x.png became 1 (1).png": ex(S / "a/1 (1).png") and not ex(U / "x.png"),
    "1.png untouched": ex(S / "a/1.png"),
    "1.txt untouched": rd(S / "a/1.txt") == "slot one",
    "1 (1).txt is a copy": rd(S / "a/1 (1).txt") == "slot one"})
ok &= check("2 same extension", {
    "y.jpg is now 2.jpg": ex(S / "b/2.jpg") and not ex(U / "y.jpg"),
    "2.jpg bytes are y's": before["U"]["y.jpg"] == hashlib.sha256((S / "b/2.jpg").read_bytes()).hexdigest(),
    "2.txt untouched": rd(S / "b/2.txt") == "slot two",
    "old 2.jpg in _duplicates": ex(S / "_duplicates/b/2.jpg"),
    "moved-out copy has a caption copy": rd(S / "_duplicates/b/2.txt") == "slot two"})
ok &= check("3 incoming caption fills empty slot", {
    "3.jpg is z": ex(S / "c/3.jpg") and not ex(U / "z.jpg"),
    "3.txt is incoming": rd(S / "c/3.txt") == "incoming three",
    "z.txt gone from unsorted": not ex(U / "z.txt")})
ok &= check("4 sorted caption wins, incoming parked", {
    "4.jpg is w": not ex(U / "w.jpg") and ex(S / "d/4.jpg"),
    "4.txt untouched": rd(S / "d/4.txt") == "slot four",
    "w.txt parked": rd(U / "_duplicates/w.txt") == "incoming four" and not ex(U / "w.txt")})
ok &= check("5 within margin", {
    "5.jpg stays": ex(S / "e/5.jpg") and before["S"]["e/5.jpg"] == hashlib.sha256((S / "e/5.jpg").read_bytes()).hexdigest(),
    "v.jpg moved out": not ex(U / "v.jpg") and ex(U / "_duplicates/v.jpg")})
ok &= check("6 feature match is review", {
    "6.jpg stays": ex(S / "f/6.jpg"), "u.jpg stays": ex(U / "u.jpg"),
    "group is in the plan as review": any(g.get("review", "").startswith("review: framing differs")
                                          and any(m["path"].endswith("u.jpg") for m in g["members"] + [g["keep"]])
                                          for g in plan["groups"])})
ok &= check("12 identical", {"7.jpg stays": ex(S / "g/7.jpg"), "t.jpg moved out": ex(U / "_duplicates/t.jpg")})
ok &= check("13 three unsorted, sorted best", {
    "8.jpg stays": ex(S / "h/8.jpg"),
    "s1 s2 s3 out": all(ex(U / f"_duplicates/{n}.jpg") for n in ("s1", "s2", "s3"))})
ok &= check("11 two slots in one directory", {
    "9.jpg and 9.png moved out": ex(S / "_duplicates/k/9.jpg") and ex(S / "_duplicates/k/9.png"),
    "p.png -> 9.png": before["U"]["p.png"] == (hashlib.sha256((S / "k/9.png").read_bytes()).hexdigest() if ex(S / "k/9.png") else None),
    "q.png -> 9 (1).png": before["U"]["q.png"] == (hashlib.sha256((S / "k/9 (1).png").read_bytes()).hexdigest() if ex(S / "k/9 (1).png") else None),
    "9.txt untouched": rd(S / "k/9.txt") == "slot nine",
    "9 (1).txt copy": rd(S / "k/9 (1).txt") == "slot nine"})
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
ok &= check("7 slot sync, policy keep", {
    "both slots exist": ex(S / "m/7a.jpg") and ex(S / "n/7b.jpg"),
    "m got n's bytes": sha(S / "m/7a.jpg") == before["S"]["n/7b.jpg"],
    "n untouched": sha(S / "n/7b.jpg") == before["S"]["n/7b.jpg"] and rd(S / "n/7b.txt") == "cat n",
    "old m copy in _duplicates": sha(S / "_duplicates/m/7a.jpg") == before["S"]["m/7a.jpg"],
    "m got a copy of n's caption": rd(S / "m/7a.txt") == "cat n"})
ok &= check("9 same folder twice", {"9b stays": ex(S / "o/9b.jpg"), "9a out": ex(S / "_duplicates/o/9a.jpg") and not ex(S / "o/9a.jpg")})
ok &= check("10 promotion into two slots", {
    "g moved into s/10b.jpg (best slot)": sha(S / "s/10b.jpg") == before["U"]["g.jpg"] and not ex(U / "g.jpg"),
    "r/10a.jpg is a copy of g": sha(S / "r/10a.jpg") == before["U"]["g.jpg"],
    "both captions untouched": rd(S / "r/10a.txt") == "cat r" and rd(S / "s/10b.txt") == "cat s",
    "g.txt parked": rd(U / "_duplicates/g.txt") == "incoming g",
    "old slot images in _duplicates": ex(S / "_duplicates/r/10a.jpg") and ex(S / "_duplicates/s/10b.jpg")})
# plan names equal real names
planned = {g["promote"]["to"] for g in plan["groups"] if "promote" in g}
planned |= {pc["to"] for g in plan["groups"] for pc in g.get("sync", [])}
ok &= check("dry-run names equal real names", {"all planned targets exist": all(Path(p).exists() for p in planned),
                                               "count": len(planned) == 9})
print("=== second run (expect nothing to move)"); assert run("--dry-run") == 0
plan2 = json.load(open(U / "_duplicates/plan.json", encoding="utf-8"))
ok &= check("15 second run", {"no moves": sum(len(g.get("move", [])) for g in plan2["groups"]) == 0,
                              "promoted and synced files come from the cache": "hashing 0 new" in run.stdout,
                              "plan has a summary": plan2["summary"]["promotions"] == 0})
print("=== undo"); rc = run("--undo")
after = {"S": snapshot(S), "U": snapshot(U)}
ok &= check("14 undo restores both trees", {"undo exit 0": rc == 0, "sorted identical": after["S"] == before["S"],
                                            "unsorted identical": after["U"] == before["U"]})
if after["S"] != before["S"] or after["U"] != before["U"]:
    for k in ("S", "U"):
        print(k, "only before:", sorted(set(before[k]) - set(after[k])), "only after:", sorted(set(after[k]) - set(before[k])),
              "changed:", sorted(x for x in before[k] if x in after[k] and before[k][x] != after[k][x]))
print("=== policy one: run"); assert run("--sorted-copies", "one") == 0
ok &= check("8 policy one", {
    "n/7b.jpg stays": sha(S / "n/7b.jpg") == before["S"]["n/7b.jpg"],
    "m/7a.jpg out": not ex(S / "m/7a.jpg") and ex(S / "_duplicates/m/7a.jpg"),
    "no caption invented for m": not ex(S / "m/7a.txt")})
ok &= check("10 policy one", {
    "g in s/10b.jpg": sha(S / "s/10b.jpg") == before["U"]["g.jpg"],
    "r/10a.jpg and its caption out": not ex(S / "r/10a.jpg") and ex(S / "_duplicates/r/10a.jpg")
                                       and rd(S / "_duplicates/r/10a.txt") == "cat r" and not ex(S / "r/10a.txt"),
    "s/10b.txt untouched": rd(S / "s/10b.txt") == "cat s"})
print("=== policy one: undo"); rc = run("--sorted-copies", "one", "--undo")
after = {"S": snapshot(S), "U": snapshot(U)}
ok &= check("14b undo after policy one", {"undo exit 0": rc == 0, "sorted identical": after["S"] == before["S"],
                                          "unsorted identical": after["U"] == before["U"]})
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
