#!/usr/bin/env python3
"""
Review the groups of the last dedup.py run full screen, and change the kept copy.

Reads <folder>/_duplicates/plan.json and shows one group per screen: the kept
copy and every moved copy, drawn from the files themselves, scaled to fit.
Click a copy to choose it as the kept one. When you move on to another group,
the chosen copy returns to its place and the old kept copy moves to
_duplicates. Press B after the click to keep both instead: the copy returns
and the kept copy stays. Every move is appended to moves.jsonl, so --undo of
the run still puts everything back. Your position and decisions are saved in
<first folder>/_duplicates/compare/review.json, and the next launch resumes.

Keys: Right or Space next group, Left previous, Home and End first and last,
click chooses a copy, B keeps both, U clears the choice, 1-9 show one copy
alone (again to return), H hides and shows the help, Esc exits and discards
the choice on the current group.

A group with a promotion or a slot sync (sorted folders) is shown but locked.
A missing file is shown as a grey tile and cannot be chosen.

Usage:    python review.py <folder>        (any folder of the run)
          run.bat ... --review             opens it after a run
          compare.bat <folder> --review    opens it after the sheets are built
"""
import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dedup  # noqa: E402

STATE_NAME = "review.json"
ORDER = {"features": 0, "through other copies": 1, "hash": 2, "exact": 3}
KIND_LABEL = {0: "feature match: a border, crop, frame or footer differs", 1: "matched through other copies",
              2: "hash match", 3: "identical files", 9: "left for review"}
HELP = [
    "Right / Space: next group     Left: previous     Home / End: first / last     Click: choose a copy as the kept one",
    "B: keep both (the chosen copy returns, the kept one stays)     U: clear the choice     1-9: one copy full screen, again to return",
    "H: hide / show this help     Esc: exit.   A choice applies when you leave the group; Esc discards the choice on this group.",
]


# ---------------------------------------------------------------------------
# Engine (no window): the groups, their current state, and the moves
# ---------------------------------------------------------------------------

class Member:
    __slots__ = ("orig", "cur", "info")

    def __init__(self, orig: Path, cur: Path, info: dict):
        self.orig, self.cur, self.info = orig, cur, info

    @property
    def exists(self) -> bool:
        return self.cur.exists()


def group_id(entry) -> str:
    paths = sorted([entry["keep"]["path"]] + [m["path"] for m in entry.get("move", []) + entry.get("members", [])])
    return hashlib.sha1("\n".join(paths).encode("utf-8")).hexdigest()[:16]


class Group:
    def __init__(self, entry: dict):
        self.entry, self.id = entry, group_id(entry)
        k = entry["keep"]
        self.members = [Member(Path(k["path"]), Path(k["path"]), k)]
        for m in entry.get("move", []):
            self.members.append(Member(Path(m["path"]), Path(m.get("to") or m["path"]), m))
        for m in entry.get("members", []):
            self.members.append(Member(Path(m["path"]), Path(m["path"]), m))
        self.keeper = self.members[0].orig
        self.kept = set()
        self.touched = False
        self.locked = "promote" in entry or "sync" in entry
        self.note = entry.get("review") or entry.get("note")
        kinds = [ORDER.get(m.get("match"), 9) for m in entry.get("move", []) + entry.get("members", [])]
        self.kind = min(kinds) if kinds else 9

    def member(self, orig: Path):
        return next(m for m in self.members if m.orig == orig)

    def to_state(self) -> dict:
        return {"keeper": str(self.keeper), "kept": sorted(str(p) for p in self.kept),
                "current": {str(m.orig): str(m.cur) for m in self.members}}

    def from_state(self, st: dict) -> None:
        self.keeper = Path(st["keeper"])
        self.kept = {Path(p) for p in st.get("kept", [])}
        for m in self.members:
            if str(m.orig) in st.get("current", {}):
                m.cur = Path(st["current"][str(m.orig)])
        self.touched = True


class Review:
    def __init__(self, folder: Path):
        folder = folder.resolve()
        self.plan_path = folder / dedup.OUT_DIRNAME / dedup.PLAN_NAME
        if not self.plan_path.is_file():
            raise SystemExit(f"no plan: {self.plan_path}\nRun the tool first; the review reads the plan of the last run.")
        self.plan = json.loads(self.plan_path.read_text(encoding="utf-8"))
        self.roots = [Path(p) for p in self.plan.get("folders", []) + self.plan.get("sorted_folders", [])]
        first = self.roots[0] if self.roots else folder
        self.state_path = first / dedup.OUT_DIRNAME / "compare" / STATE_NAME
        self.groups = [Group(e) for e in self.plan["groups"] if e.get("move") or e.get("members")]
        self.groups.sort(key=lambda g: (g.kind, str(g.members[0].orig).casefold()))
        self.position = 0
        self.load_state()

    # --- state -------------------------------------------------------------
    def load_state(self) -> None:
        try:
            st = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        by_id = {g.id: g for g in self.groups}
        for gid, s in st.get("groups", {}).items():
            if gid in by_id:
                by_id[gid].from_state(s)
        for i, g in enumerate(self.groups):
            if g.id == st.get("position"):
                self.position = i

    def save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        st = {"plan_written": self.plan.get("written"), "saved": datetime.now().isoformat(timespec="seconds"),
              "position": self.groups[self.position].id if self.groups else None,
              "groups": {g.id: g.to_state() for g in self.groups if g.touched}}
        dedup.write_json(self.state_path, st)

    # --- moves -------------------------------------------------------------
    def root_of(self, p: Path) -> Path:
        for r in self.roots:
            if p.is_relative_to(r):
                return r
        return p.parent

    def log(self, root: Path, **entry) -> None:
        (root / dedup.OUT_DIRNAME).mkdir(parents=True, exist_ok=True)
        with open(root / dedup.OUT_DIRNAME / dedup.MOVES_NAME, "a", encoding="utf-8") as f:
            f.write(json.dumps({"time": datetime.now().isoformat(timespec="seconds"), **entry}, ensure_ascii=False) + "\n")

    def move(self, m: Member, target: Path, keeper: Path, reason: str) -> bool:
        """Move a member's image, and its caption, from where it is to target; log both."""
        if m.cur == target or not m.cur.exists():
            return False
        if target.exists():
            target = dedup.free_dest(target, target.with_suffix(".txt"))
        root = self.root_of(m.orig)
        src = m.cur
        dedup.move_file(src, target)
        self.log(root, **{"from": str(src), "to": str(target)}, kept=str(keeper), reason=reason)
        cap = src.with_suffix(".txt")
        if cap.exists() and not target.with_suffix(".txt").exists():
            cdst = target.with_suffix(".txt")
            shared = any(p.is_file() and p.suffix.lower() in dedup.IMAGE_EXTS and p.stem.casefold() == src.stem.casefold()
                         for p in src.parent.iterdir())
            if shared:                       # another image of the same stem stays with the caption
                shutil.copy2(cap, cdst)
                self.log(root, action="copy", **{"from": str(cap), "to": str(cdst)}, sha256=dedup.file_sha256(cdst),
                         reason="review: caption of a moved copy, shared with another image")
            else:
                dedup.move_file(cap, cdst)
                self.log(root, **{"from": str(cap), "to": str(cdst)}, kept=str(keeper),
                         reason="review: caption of a moved copy")
        m.cur = target
        return True

    def dup_target(self, m: Member) -> Path:
        root = self.root_of(m.orig)
        rel = m.orig.relative_to(root) if m.orig.is_relative_to(root) else Path(m.orig.name)
        out = root / dedup.OUT_DIRNAME / rel
        return dedup.free_dest(out, out.with_suffix(".txt"))

    def apply(self, g: Group, keeper: Path, kept_add=()) -> int:
        """Bring the group to: keeper and kept members at their original paths,
        every other member in _duplicates. Returns the number of files moved."""
        if g.locked:
            return 0
        old = g.keeper
        g.kept |= set(kept_add)
        g.kept.discard(keeper)
        g.keeper = keeper
        g.touched = True
        n = 0
        for m in g.members:                   # out first, so the places are free
            if m.orig != keeper and m.orig not in g.kept and m.cur == m.orig:
                reason = "review: replaced as the kept copy" if m.orig == old else "review: moved out"
                n += self.move(m, self.dup_target(m), keeper, reason)
        for m in g.members:                   # then home
            if (m.orig == keeper or m.orig in g.kept) and m.cur != m.orig:
                reason = "review: chosen as the kept copy" if m.orig == keeper else "review: kept too"
                n += self.move(m, m.orig, keeper, reason)
        self.save_state()
        return n


# ---------------------------------------------------------------------------
# Window
# ---------------------------------------------------------------------------

def run_ui(rv: Review, snapshot: str | None = None) -> int:
    import tkinter as tk
    from tkinter import font as tkfont
    from PIL import Image, ImageOps, ImageTk

    if not rv.groups:
        print("the plan has no groups to review")
        return 0

    class App:
        def __init__(self):
            self.root = tk.Tk()
            self.root.title("Deduplicate: review")
            self.root.attributes("-fullscreen", True)
            self.root.configure(bg="black")
            self.W, self.H = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
            self.canvas = tk.Canvas(self.root, bg="black", highlightthickness=0, cursor="hand2")
            self.canvas.pack(fill=tk.BOTH, expand=True)
            size = max(11, round(self.H / 120))      # Tk is not DPI-aware: size the text from the screen
            self.font = tkfont.Font(family="Segoe UI", size=size)
            self.bold = tkfont.Font(family="Segoe UI", size=size + 1, weight="bold")
            self.small = tkfont.Font(family="Segoe UI", size=size - 1)
            self.line = int(size * 1.7)
            self.show_help = True
            self.pending = None            # orig path of the clicked copy
            self.pending_both = False
            self.zoom = None               # member index shown alone
            self.message = ""
            self.photos, self.tiles, self.cache = [], [], {}
            self.canvas.bind("<Button-1>", self.on_click)
            self.root.bind("<Key>", self.on_key)
            self.render()

        # --- helpers ---
        @property
        def group(self):
            return rv.groups[rv.position]

        def load(self, path: Path, box):
            key = (str(path), box)
            if key in self.cache:
                return self.cache[key]
            try:
                with Image.open(path) as im:
                    im.draft("RGB", (box[0] * 2, box[1] * 2))
                    im = ImageOps.exif_transpose(im).convert("RGB")
                    # fill the cell, up or down, so copies of different sizes compare at one size;
                    # the label states the real pixel size
                    scale = min(box[0] / im.width, box[1] / im.height)
                    size = (max(1, int(im.width * scale)), max(1, int(im.height * scale)))
                    im = im.resize(size, Image.LANCZOS if scale < 1 else Image.BICUBIC)
                    photo = ImageTk.PhotoImage(im)
            except Exception:  # noqa: BLE001
                photo = None
            self.cache[key] = photo
            return photo

        def label_for(self, m: Member, g: Group):
            info = m.info
            if not m.exists:
                state = "MISSING"
            elif m.orig == g.keeper:
                state = "KEPT"
            elif m.orig == self.pending:
                state = "CHOSEN, keeps both" if self.pending_both else "CHOSEN as the kept copy"
            elif m.orig in g.kept:
                state = "kept too"
            elif m.cur != m.orig:
                state = "moved to _duplicates"
            else:
                state = "in place"
            root = rv.root_of(m.orig)
            line1 = f"{state}:  {m.orig.name}"
            line2 = (f"{info.get('w')}x{info.get('h')}   bucket {info.get('tier')}   score {info.get('score')}"
                     + (f"   {info['match']}" if info.get("match") else "") + f"   [{root.name}]")
            return line1, line2

        def frame_colour(self, m: Member, g: Group):
            if not m.exists:
                return "#555555", 2
            if m.orig == self.pending:
                return "#ffd400", 5
            if m.orig == g.keeper:
                return "#2ecc40", 5
            if m.orig in g.kept:
                return "#39c0ff", 4
            return "#444444", 1

        # --- drawing ---
        def render(self):
            c = self.canvas
            c.delete("all")
            self.photos, self.tiles = [], []
            g = self.group
            y = 6
            if self.show_help:
                for line in HELP:
                    c.create_text(12, y, anchor="nw", text=line, fill="#bbbbbb", font=self.small)
                    y += self.line
                y += 4
            head = f"Group {rv.position + 1} of {len(rv.groups)}   ·   {KIND_LABEL.get(g.kind, '')}   ·   {g.entry.get('reason', '')}"
            c.create_text(12, y, anchor="nw", text=head, fill="white", font=self.bold)
            y += self.line + 4
            if g.locked:
                c.create_text(12, y, anchor="nw", fill="#ff9955", font=self.font,
                              text="Locked: this group has a promotion or a slot sync; undo the run to change it.")
                y += self.line
            elif g.note:
                c.create_text(12, y, anchor="nw", text=f"Note: {g.note}", fill="#ff9955", font=self.font)
                y += self.line
            top, bottom = y + 6, self.H - self.line - 12
            if self.zoom is not None and self.zoom < len(g.members):
                self.draw_tile(g.members[self.zoom], g, 12, top, self.W - 24, bottom - top, self.zoom)
            else:
                n = len(g.members)
                cols = n if n <= 4 else (4 if n <= 8 else math.ceil(math.sqrt(n)))
                rows = math.ceil(n / cols)
                gap = 14
                cell_w = (self.W - gap * (cols + 1)) / cols
                cell_h = (bottom - top - gap * (rows + 1)) / rows
                for i, m in enumerate(g.members):
                    r, col = divmod(i, cols)
                    x0 = gap + col * (cell_w + gap)
                    y0 = top + gap + r * (cell_h + gap)
                    self.draw_tile(m, g, x0, y0, cell_w, cell_h, i)
            # status line
            if self.message:
                status, colour = self.message, "#ff6666"
            elif g.locked:
                status, colour = "No actions on a locked group.", "#999999"
            elif self.pending is not None:
                name, kname = self.pending.name, g.keeper.name
                status = (f"On leaving: {name} returns to its place and {kname} stays (both kept)." if self.pending_both
                          else f"On leaving: {name} becomes the kept copy and {kname} moves to _duplicates.")
                colour = "#ffd400"
            else:
                status, colour = "Click a copy to choose it as the kept one.", "#999999"
            c.create_text(12, self.H - self.line - 4, anchor="nw", text=status, fill=colour, font=self.font)

        def draw_tile(self, m: Member, g: Group, x0, y0, w, h, index):
            c = self.canvas
            img_h = h - 2 * self.line - 8
            colour, width = self.frame_colour(m, g)
            photo = self.load(m.cur, (int(w - 12), int(img_h - 12))) if m.exists else None
            if photo is not None:
                pw, ph = photo.width(), photo.height()
                px, py = x0 + (w - pw) / 2, y0 + (img_h - ph) / 2
                c.create_image(px, py, anchor="nw", image=photo)
                self.photos.append(photo)
                c.create_rectangle(px - 3, py - 3, px + pw + 3, py + ph + 3, outline=colour, width=width)
            else:
                c.create_rectangle(x0 + 6, y0 + 6, x0 + w - 6, y0 + img_h - 6, fill="#2a2a2a", outline=colour, width=width)
                c.create_text(x0 + w / 2, y0 + img_h / 2, text="file not found" if not m.exists else "cannot read",
                              fill="#888888", font=self.font)
            l1, l2 = self.label_for(m, g)
            c.create_text(x0 + w / 2, y0 + img_h + 4, anchor="n", text=f"{index + 1}.  {l1}", fill=colour if colour != "#444444" else "#dddddd",
                          font=self.font, width=w)
            c.create_text(x0 + w / 2, y0 + img_h + 4 + self.line, anchor="n", text=l2, fill="#aaaaaa", font=self.small, width=w)
            self.tiles.append((x0, y0, x0 + w, y0 + h, index))

        # --- actions ---
        def on_click(self, ev):
            g = self.group
            if g.locked:
                return
            for x0, y0, x1, y1, i in self.tiles:
                if x0 <= ev.x <= x1 and y0 <= ev.y <= y1:
                    m = g.members[i]
                    if not m.exists:
                        return
                    if m.orig == g.keeper:
                        self.pending, self.pending_both = None, False
                    else:
                        self.pending, self.pending_both = m.orig, False
                    self.message = ""
                    self.render()
                    return

        def apply_pending(self):
            g = self.group
            if self.pending is None or g.locked:
                return
            try:
                if self.pending_both:
                    rv.apply(g, g.keeper, {self.pending})
                else:
                    rv.apply(g, self.pending)
            except Exception as e:  # noqa: BLE001
                self.message = f"could not apply: {e}"
            self.pending, self.pending_both = None, False

        def goto(self, index):
            self.apply_pending()
            rv.position = max(0, min(len(rv.groups) - 1, index))
            self.zoom = None
            self.cache = {}
            rv.save_state()
            self.render()

        def on_key(self, ev):
            k = ev.keysym
            if k in ("Right", "space", "Next"):
                self.goto(rv.position + 1)
            elif k in ("Left", "Prior"):
                self.goto(rv.position - 1)
            elif k == "Home":
                self.goto(0)
            elif k == "End":
                self.goto(len(rv.groups) - 1)
            elif k == "Escape":
                if self.zoom is not None:
                    self.zoom = None
                    self.render()
                else:
                    rv.save_state()
                    self.root.destroy()
            elif k.lower() == "h":
                self.show_help = not self.show_help
                self.render()
            elif k.lower() == "b":
                if self.pending is not None:
                    self.pending_both = not self.pending_both
                    self.render()
            elif k.lower() == "u":
                self.pending, self.pending_both = None, False
                self.message = ""
                self.render()
            elif k.isdigit() and k != "0":
                i = int(k) - 1
                if i < len(self.group.members):
                    self.zoom = None if self.zoom == i else i
                    self.render()

    app = App()
    if snapshot:                           # render the first screen to a file and exit (for tests and docs)
        from PIL import ImageGrab
        app.root.update()
        app.root.after(400, lambda: (ImageGrab.grab().save(snapshot), app.root.destroy()))
    app.root.mainloop()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="a folder of the run; its _duplicates/plan.json is read")
    ap.add_argument("--snapshot", metavar="FILE", help="save a screenshot of the first screen to FILE and exit")
    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except AttributeError:
            pass
    rv = Review(Path(args.folder))
    print(f"{len(rv.groups)} group(s); resuming at {rv.position + 1}; state in {rv.state_path}")
    return run_ui(rv, args.snapshot)


if __name__ == "__main__":
    sys.exit(main())
