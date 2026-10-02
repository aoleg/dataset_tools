#!/usr/bin/env python3
"""Unit tests for k2prep's bucket and geometry logic.

Plain asserts, no pytest dependency:

    python test_k2prep.py

Covers SPEC.md section 4.2 (the bucket list must match musubi-tuner exactly),
section 4.3 (the per-tier family table), and the pure-function half of the
acceptance criteria in section 12.
"""

from collections import defaultdict

import k2prep as k


def test_bucket_list_sizes():
    assert len(k.generate_buckets(1024)) == 65
    assert len(k.generate_buckets(768)) == 49
    assert len(k.generate_buckets(512)) == 33


def test_musubi_exact_dimensions():
    """4:3 at 1024 is 1184x880. Rounding sqrt(area * AR) to a multiple of 16
    gives 1168x880, which is not a bucket and would make the trainer re-crop."""
    buckets = k.generate_buckets(1024)
    assert (1184, 880) in buckets
    assert (1168, 880) not in buckets


def test_family_table():
    """The per-tier dimensions in SPEC.md section 4.3, verbatim."""
    expected = {
        "9:16": [(768, 1360), (576, 1024), (384, 672)],
        "2:3":  [(832, 1248), (624, 944), (416, 624)],
        "4:5":  [(912, 1136), (688, 848), (448, 576)],
        "1:1":  [(1024, 1024), (768, 768), (512, 512)],
        "5:4":  [(1136, 912), (848, 688), (560, 464)],
        "3:2":  [(1248, 832), (944, 624), (624, 416)],
        "16:9": [(1360, 768), (1024, 576), (672, 384)],
    }
    for family, rows in expected.items():
        for tier, want in zip(k.TIERS, rows):
            assert k.bucket_for(tier, family) == want, (family, tier)


def test_every_bucket_is_a_real_bucket():
    """Acceptance criterion 3, at the source: every dimension this tool can emit
    is present in the tier's generated list."""
    for tier in k.TIERS:
        buckets = set(k.generate_buckets(tier))
        for family in k.AR_FAMILIES:
            assert k.bucket_for(tier, family) in buckets


def test_family_assignment():
    assert k.assign_family(1.5) == "3:2"
    assert k.assign_family(1.0) == "1:1"
    assert k.assign_family(4.0) == "16:9"        # 21:9+ is cropped hard, on purpose
    assert k.assign_family(640 / 480) == "5:4"
    assert k.assign_family(1080 / 2340) == "9:16"


def test_panorama_demotes_on_post_crop_area():
    """Acceptance criterion 5. 2400x600 is 1.44 MP raw but 0.64 MP after the
    16:9 crop, so it belongs in 768, not 1024."""
    tier, bucket, crop = k.assign_tier(2400, 600, k.assign_family(4.0))
    assert tier == 768
    assert bucket == (1024, 576)
    assert crop == (1067, 600)


def test_small_camera_image_lands_in_512():
    """Acceptance criterion 6."""
    tier, bucket, crop = k.assign_tier(640, 480, k.assign_family(640 / 480))
    assert tier == 512
    assert bucket == (560, 464)


def test_thumbnail_is_too_small():
    """Acceptance criterion 7."""
    assert k.assign_tier(400, 300, k.assign_family(400 / 300)) is None


def test_upscale_tolerance_boundary():
    """A source exactly at the tolerance limit fits; one pixel under does not."""
    bw, bh = k.bucket_for(1024, "1:1")
    limit = (bw * bh) / k.UPSCALE_TOLERANCE ** 2  # 792,874.1
    assert 891 * 891 >= limit > 890 * 890
    assert k.assign_tier(891, 891, "1:1")[0] == 1024
    assert k.assign_tier(890, 890, "1:1")[0] == 768


def test_crop_dims_minimal():
    assert k.crop_dims(6000, 4000, 1248 / 832) == (6000, 4000)   # already 3:2
    assert k.crop_dims(2400, 600, 1024 / 576) == (1067, 600)     # trim width
    assert k.crop_dims(1000, 2000, 1.0) == (1000, 1000)          # trim height


def test_crop_box_anchoring():
    """Horizontal centred always; vertical biased to 1/3 for portrait targets."""
    left, top, right, bottom = k.crop_box(1000, 2000, 1.0)       # square target
    assert (left, right) == (0, 1000)
    assert (top, bottom) == (500, 1500)                          # centred

    left, top, right, bottom = k.crop_box(1000, 2000, 768 / 1360)  # portrait
    cw = right - left
    ch = bottom - top
    assert (cw, ch) == k.crop_dims(1000, 2000, 768 / 1360)
    assert left == (1000 - cw) // 2
    assert top == int((2000 - ch) * (1 / 3))                     # above centre

    # Never outside the source.
    for w, h in ((3, 5000), (5000, 3), (1, 1), (17, 19)):
        for family in k.AR_FAMILIES:
            bw, bh = k.bucket_for(1024, family)
            l, t, r, b = k.crop_box(w, h, bw / bh)
            assert 0 <= l < r <= w and 0 <= t < b <= h, (w, h, family)


def test_jpeg_quality_estimator():
    """An exact IJG-scaled table must round-trip to its own quality."""
    for q in (30, 50, 75, 85, 90, 95, 97):
        assert k.estimate_jpeg_quality(k._ijg_scaled_table(q)) == q
    # A wildly non-standard table is rejected rather than guessed at.
    assert k.estimate_jpeg_quality([1] * 32 + [255] * 32) is None
    assert k.estimate_jpeg_quality([16] * 63) is None            # wrong length


def test_score_bands_are_absolute():
    assert k.score_q(100) == 10 and k.score_q(95) == 9 and k.score_q(54) == 1
    assert k.score_q(None) is None
    assert k.score_b(1.00) == 10 and k.score_b(1.20) == 7 and k.score_b(3.0) == 1
    assert k.score_d(0.10) == 10 and k.score_d(0.026) == 6 and k.score_d(0.0) == 1


def test_needed_area_matches_spec_table():
    """SPEC.md section 4.1, minimum post-crop area per tier (square family)."""
    assert k.needed_area(1024, "1:1") == 792874
    assert k.needed_area(768, "1:1") == 445991
    assert k.needed_area(512, "1:1") == 198218


# ---------------------------------------------------------------------------
# Rendered-image scoring
# ---------------------------------------------------------------------------

def _block_grid(w, h, period, amplitude=12.0, base=110.0):
    """A synthetic image with a vertical edge every `period` pixels, which is
    what a JPEG block grid looks like to the metric."""
    import numpy as np
    from PIL import Image
    x = np.arange(w)
    step = np.floor(x / period) * amplitude
    step = step - step.mean()
    field = np.tile(base + step, (h, 1))
    return Image.fromarray(np.clip(field, 0, 255).astype("uint8"), "L")


def test_block_detector_finds_a_known_period():
    """The detector must lock on to a non-integer period, since an 8px source
    block lands every 8*scale output pixels."""
    for period in (4.0, 5.7, 8.0, 11.3):
        img = _block_grid(600, 64, period)
        strong = k.block_period_energy(img, period)
        assert strong is not None and strong > 0.10, (period, strong)
        # ... and must not fire on a period the image does not contain
        wrong = k.block_period_energy(img, period * 1.7)
        assert wrong < strong / 2, (period, wrong, strong)


def test_block_detector_reports_nothing_below_the_resolution_limit():
    """A heavy downscale puts the grid past Nyquist. Returning None there is the
    correct answer, not a gap: the artifacts really are gone."""
    img = _block_grid(600, 64, 8.0)
    assert k.block_period_energy(img, 2.0) is None
    assert k.block_period_energy(img, k.MIN_BLOCK_PERIOD - 0.1) is None
    assert k.block_period_energy(img, k.MIN_BLOCK_PERIOD) is not None


def test_flat_image_scores_clean():
    import numpy as np
    from PIL import Image
    noise = (np.random.default_rng(0).normal(128, 20, (400, 400))
             .clip(0, 255).astype("uint8"))
    img = Image.fromarray(noise, "L")
    assert k.block_period_energy(img, 8.0) < 0.05


def test_rendered_bands_are_separate_from_source_bands():
    """They measure different pixels and must not be silently interchangeable."""
    assert k.D_RENDERED_BANDS != k.D_BANDS
    assert k.score_d_rendered(0.140) == 10 and k.score_d_rendered(0.005) == 1
    assert k.score_b_rendered(0.005) == 10 and k.score_b_rendered(0.5) == 1
    # monotone, and every band reachable
    assert [k.score_d_rendered(lo) for lo, _s in k.D_RENDERED_BANDS] == list(range(10, 1, -1))
    assert [k.score_b_rendered(hi) for hi, _s in k.B_RENDERED_BANDS] == list(range(10, 1, -1))


# ---------------------------------------------------------------------------
# --sort
# ---------------------------------------------------------------------------

def _sorted_geometry(w, h):
    """What analyse_for_sort would decide, without touching a file."""
    from pathlib import Path
    res = k.Result(path=Path("x.jpg"), name="x.jpg")
    res.src_w, res.src_h = w, h
    res.src_ar = w / h
    res.family = k.assign_family(res.src_ar)
    target = k.bucket_for(k.SORT_TIER, res.family)
    cw, ch = k.crop_dims(w, h, target[0] / target[1])
    res.crop = (cw, ch)
    res.bucket = target if cw * ch > target[0] * target[1] else (cw, ch)
    return res, target


def test_sort_always_judges_at_the_1024_tier():
    """A small image would be a 512-tier image in the dataset pipeline, but its
    sort score must still be measured against 1024 or scores from different
    folders cannot be compared."""
    for w, h in ((6000, 4000), (2400, 600), (640, 480), (400, 300)):
        res, target = _sorted_geometry(w, h)
        assert target == k.bucket_for(1024, res.family)


def test_sort_downscales_large_sources():
    res, target = _sorted_geometry(6000, 4000)
    assert res.bucket == target                     # rendered down to 1024
    assert res.bucket != res.crop


def test_sort_never_upscales_a_small_source():
    """'Already at 1024 or less' is scored as it is; upscaling would invent
    detail and flatter the result."""
    for w, h in ((800, 600), (400, 300), (640, 480)):
        res, target = _sorted_geometry(w, h)
        assert res.bucket == res.crop, (w, h)
        assert res.bucket[0] <= target[0] and res.bucket[1] <= target[1]
        # scored at scale 1.0, so the block period is the source's own 8px grid
        assert 8.0 * (res.bucket[0] / res.crop[0]) == 8.0


def test_sort_score_maps_to_its_folder():
    from pathlib import Path
    prep = Path("/tmp/_prep")
    for score in range(1, 11):
        assert k.sort_dir_for(prep, score).name == f"{k.SORT_DIR_PREFIX}{score}"


def test_fine_score_floors_to_the_band_score():
    """The continuous value exists to rank within a band; it must never
    disagree with the integer score the rest of the tool uses."""
    import math
    for i in range(1, 1200):
        v = i / 2000
        assert math.floor(k.fine_score(v, k.D_RENDERED_BANDS, False)) == \
            k.score_d_rendered(v), v
        assert math.floor(k.fine_score(v, k.B_RENDERED_BANDS, True)) == \
            k.score_b_rendered(v), v


def test_fine_score_is_monotone():
    d = [k.fine_score(v, k.D_RENDERED_BANDS, False)
         for v in (0.001, 0.02, 0.06, 0.14, 0.19, 0.23, 0.5, 5.0)]
    assert all(a < b for a, b in zip(d, d[1:])), d      # more detail is better
    b = [k.fine_score(v, k.B_RENDERED_BANDS, True)
         for v in (0.001, 0.009, 0.02, 0.05, 0.08, 0.3, 3.0)]
    assert all(a > b2 for a, b2 in zip(b, b[1:])), b    # more blocking is worse


def _fake(name, fine):
    from pathlib import Path
    r = k.Result(path=Path(name), name=name)
    r.scored = True
    r.composite_fine = fine
    r.composite = int(fine)
    return r


def test_every_tier_is_populated_however_uniform_the_folder():
    """The whole point of --sort N: a folder of uniformly excellent images
    still has a best third and a worst third."""
    for spread in (0.001, 0.05, 3.0):
        images = [_fake(f"i{i}.jpg", 10.0 + i * spread / 40) for i in range(40)]
        for n in range(2, 11):
            for r in images:
                r.sort_tier = 0
            tiers = k.plan_quality_tiers(images, n)
            assert len(tiers) == n
            assert all(t.count > 0 for t in tiers), (spread, n,
                                                     [t.count for t in tiers])
            assert sum(t.count for t in tiers) == len(images)


def test_tiers_are_ordered_best_first():
    images = [_fake(f"i{i}.jpg", 1.0 + i * 0.25) for i in range(36)]
    k.plan_quality_tiers(images, 4)
    by_tier = defaultdict(list)
    for r in images:
        by_tier[r.sort_tier].append(r.composite_fine)
    for tier in range(1, 4):
        assert min(by_tier[tier]) >= max(by_tier[tier + 1]), by_tier


def test_a_lone_outlier_does_not_take_a_tier_to_itself():
    """One superb image among ordinary ones must not push everything else down
    a grade - the failure mode absolute bands already have."""
    images = [_fake("star.jpg", 10.9)] + [_fake(f"i{i}.jpg", 3.0 + i * 0.01)
                                          for i in range(29)]
    tiers = k.plan_quality_tiers(images, 3)
    assert tiers[0].count > 1, tiers[0].count
    star = next(r for r in images if r.name == "star.jpg")
    assert star.sort_tier == 1


def test_cuts_prefer_a_real_gap_near_the_ideal_position():
    """30 images, N=3, with a chasm at 12 rather than the ideal 10: the cut
    should slide to the gap instead of splitting a tight cluster."""
    images = ([_fake(f"a{i}.jpg", 9.0 + i * 0.01) for i in range(12)] +
              [_fake(f"b{i}.jpg", 5.0 + i * 0.01) for i in range(18)])
    tiers = k.plan_quality_tiers(images, 3)
    assert tiers[0].count == 12, [t.count for t in tiers]
    assert tiers[0].gap_below > 3.0


def test_fewer_images_than_tiers_leaves_tiers_empty():
    images = [_fake("a.jpg", 9.0), _fake("b.jpg", 4.0)]
    tiers = k.plan_quality_tiers(images, 5)
    assert [t.count for t in tiers] == [1, 1, 0, 0, 0]
    assert images[0].sort_tier == 1 and images[1].sort_tier == 2


def test_sort_tier_count_is_validated():
    import contextlib, io
    for bad in ("0", "1", "11", "-3", "abc"):
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                k.parse_args(["f", "--sort", bad])
            except SystemExit as exc:
                assert exc.code != 0
            else:
                raise AssertionError(f"--sort {bad} was accepted")
    assert k.parse_args(["f", "--sort"]).sort == k.SORT_ABSOLUTE
    for good in range(2, 11):
        assert k.parse_args(["f", "--sort", str(good)]).sort == good


# ---------------------------------------------------------------------------
# --vl
# ---------------------------------------------------------------------------

def test_vl_reply_parsing_survives_how_models_actually_answer():
    crit = ["sharpness", "composition"]
    for reply in ('{"sharpness": 7, "composition": 4}',
                  '```json\n{"sharpness": 7, "composition": 4}\n```',
                  'Sure! Here you go:\n{"sharpness": 7, "composition": 4}\nHope that helps.',
                  '{"Sharpness": 7.0, "COMPOSITION": 4}'):
        assert k.parse_vl_reply(reply, crit) == {"sharpness": 7, "composition": 4}, reply
    # bare numbers, in order
    assert k.parse_vl_reply("7 4", crit) == {"sharpness": 7, "composition": 4}
    # out of range is clamped, not accepted blindly
    assert k.parse_vl_reply('{"sharpness": 99, "composition": -5}', crit) == \
        {"sharpness": 10, "composition": 1}


def test_vl_reply_parsing_refuses_to_guess():
    crit = ["sharpness", "composition"]
    for reply in ("I cannot rate this image.", "", "sharpness is good"):
        try:
            k.parse_vl_reply(reply, crit)
        except ValueError:
            pass
        else:
            raise AssertionError(f"guessed a score from {reply!r}")


def test_blend_weights_the_model_higher_without_letting_it_win():
    both = k.blend_scores(10, 10)
    great_criteria_broken_image = k.blend_scores(10, 2)
    great_image_wrong_subject = k.blend_scores(2, 10)
    mediocre_both = k.blend_scores(7, 7)

    assert abs(both - 10) < 1e-9
    # the model counts for more...
    assert great_criteria_broken_image > great_image_wrong_subject
    # ...but a technically broken image is still marked down, hard enough that
    # a merely decent one beats it
    assert great_criteria_broken_image < both
    assert great_criteria_broken_image < mediocre_both
    # and it is not overwhelming: a perfect critique cannot rescue a 1
    assert k.blend_scores(10, 1) < mediocre_both


def test_blend_is_monotone_in_both_inputs():
    for tech in (1, 4, 7, 10):
        vals = [k.blend_scores(vl, tech) for vl in range(1, 11)]
        assert all(a < b for a, b in zip(vals, vals[1:])), tech
    for vl in (1, 4, 7, 10):
        vals = [k.blend_scores(vl, tech) for tech in range(1, 11)]
        assert all(a < b for a, b in zip(vals, vals[1:])), vl


def test_env_parsing():
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "x.env"
        p.write_text('# comment\n\nA=1\n B = "two" \nC=\nD=\'three\'\nnot a pair\n',
                     encoding="utf-8")
        assert k.parse_env_file(p) == {"A": "1", "B": "two", "D": "three"}


def test_vl_endpoint_normalisation_and_signature():
    cfg = k.VLConfig("http://localhost:8080/v1", "m", 10.0, ["a", "b"])
    assert cfg.chat_url.endswith("/v1/chat/completions")
    assert cfg.models_url.endswith("/v1/models")
    # the cache identity must move when the question does
    other = k.VLConfig("http://localhost:8080/v1", "m", 10.0, ["a", "c"])
    model2 = k.VLConfig("http://localhost:8080/v1", "n", 10.0, ["a", "b"])
    assert cfg.signature() != other.signature()
    assert cfg.signature() != model2.signature()
    # ...but not when something irrelevant does
    assert cfg.signature() == k.VLConfig("http://elsewhere/v1", "m", 99.0,
                                         ["a", "b"]).signature()


def test_vl_prompt_asks_for_high_is_good_and_names_every_criterion():
    cfg = k.VLConfig("http://x/v1", "m", 10.0, ["sharpness", "composition"])
    msgs = k.build_vl_messages(cfg, "AAAA")
    text = msgs[-1]["content"][0]["text"]
    assert "10 is excellent, 1 is unusable" in text     # never inverted
    for c in cfg.criteria:
        assert f"- {c}" in text
    assert msgs[-1]["content"][1]["image_url"]["url"].startswith(
        "data:image/jpeg;base64,")


def test_vl_requires_sort():
    import contextlib, io
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            k.parse_args(["f", "--vl", "sharpness"])
        except SystemExit as exc:
            assert exc.code != 0
        else:
            raise AssertionError("--vl was accepted without --sort")
    args = k.parse_args(["f", "--sort", "3", "--vl", " sharpness , Composition ,, sharpness "])
    assert args.vl_criteria == ["sharpness", "Composition"]   # trimmed, deduped


def test_thread_defaults_differ_for_local_work_and_the_model():
    plain = k.parse_args(["f"])
    assert (plain.local_threads, plain.vl_threads) == (4, 1)
    explicit = k.parse_args(["f", "--threads", "8"])
    assert (explicit.local_threads, explicit.vl_threads) == (8, 8)


def test_move_requires_sort():
    import contextlib, io
    for argv in (["f", "--move"], ["f", "--move", "--threshold", "5"]):
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                k.parse_args(argv)
            except SystemExit as exc:
                assert exc.code != 0
            else:
                raise AssertionError("--move was accepted without --sort")
    assert k.parse_args(["f", "--sort", "--move"]).move is True


# ---------------------------------------------------------------------------
# Bucket merging
# ---------------------------------------------------------------------------

def _mk(name, w, h):
    """A Result placed in its natural bucket, as analyse() would leave it."""
    from pathlib import Path
    res = k.Result(path=Path(name), name=name)
    res.src_w, res.src_h, res.src_ar = w, h, w / h
    res.family = k.assign_family(res.src_ar)
    fit = k.assign_tier(w, h, res.family)
    assert fit is not None, (name, w, h)
    k.place(res, fit[0], fit[1])
    res.status = k.ST_ACCEPTED
    return res


def test_orientation_is_never_flipped():
    portrait, square, landscape = (912, 1136), (1024, 1024), (1136, 912)
    assert k.orientation_ok(0.75, portrait)
    assert k.orientation_ok(0.75, square)
    assert not k.orientation_ok(0.75, landscape)
    assert k.orientation_ok(1.5, landscape)
    assert k.orientation_ok(1.5, square)
    assert not k.orientation_ok(1.5, portrait)
    assert k.orientation_ok(1.0, portrait) and k.orientation_ok(1.0, landscape)


def test_ar_distance_is_symmetric_in_log_space():
    """0.8 -> 1:1 and 1.25 -> 1:1 are the same move mirrored, so they must
    measure the same. Linear ratio distance gets this wrong (0.20 vs 0.25)."""
    square = (1024, 1024)
    assert abs(k.ar_distance(0.8, square) - k.ar_distance(1.25, square)) < 1e-9
    assert abs(0.8 - 1.0) != abs(1.25 - 1.0)          # the flaw being avoided


def test_crop_cap_refuses_a_brutal_move():
    tall = _mk("phone.png", 1080, 2340)             # 9:16, AR 0.46
    ok, why = k.can_accept(tall, k.bucket_for(1024, "4:5"))
    assert not ok and why == "crop", why
    ok, _ = k.can_accept(tall, k.bucket_for(1024, "2:3"))
    assert ok                                        # 31%, under the cap


def test_rescue_merges_an_orphan_512_tier():
    """The 5/3/1/1/1 shape that has no healthy bucket to aim at."""
    images = ([_mk(f"sq_{i}.jpg", 600, 600) for i in range(5)] +
              [_mk(f"cam_{i}.jpg", 640, 480) for i in range(3)] +
              [_mk("tall_916.jpg", 400, 700),
               _mk("tall_23.jpg", 480, 720),
               _mk("tall_45.jpg", 450, 580)])
    assert all(r.tier == 512 for r in images)
    assert max(k.bucket_counts(images).values()) == 5      # nothing healthy

    moves, unmerged, _before, after = k.plan_merge(images)

    assert after[(512, (512, 512))] == 10
    assert after[(512, (384, 672))] == 1                   # 43% crop, left alone
    assert len(moves) == 5
    assert [r.name for r, _why in unmerged] == ["tall_916.jpg"]


def test_merge_never_flips_orientation():
    images = ([_mk(f"port_{i}.jpg", 3024, 4032) for i in range(3)] +
              [_mk(f"land_{i}.jpg", 4032, 3024) for i in range(9)])
    k.plan_merge(images)
    for r in images:
        assert k.orientation_ok(r.src_ar, r.bucket), r.name


def test_healthy_buckets_are_left_alone():
    images = [_mk(f"land_{i}.jpg", 4032, 3024) for i in range(12)]
    before = k.bucket_counts(images)
    moves, unmerged, _b, after = k.plan_merge(images)
    assert not moves and not unmerged
    assert dict(before) == after


def test_too_few_images_cannot_be_rescued():
    """Five images cannot make a bucket of eight, so nothing is cropped for
    nothing."""
    images = [_mk("sq.jpg", 600, 600), _mk("cam.jpg", 640, 480),
              _mk("t1.jpg", 480, 720), _mk("t2.jpg", 450, 580),
              _mk("t3.jpg", 400, 700)]
    moves, unmerged, _b, _a = k.plan_merge(images)
    assert not moves
    assert len(unmerged) == 5


# ---------------------------------------------------------------------------
# --recursive
# ---------------------------------------------------------------------------

def test_recursive_refuses_sort():
    import contextlib, io
    for argv in (["f", "--recursive", "--sort"],
                 ["f", "--sort", "3", "--recursive"]):
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                k.parse_args(argv)
            except SystemExit as exc:
                assert exc.code != 0
            else:
                raise AssertionError("--sort was accepted with --recursive")
    assert k.parse_args(["f", "--recursive"]).recursive is True


def test_qualified_name_and_tier_dir():
    from pathlib import Path
    assert k.qualified_name("", "a.jpg") == "a.jpg"
    assert k.qualified_name("x/y", "a.jpg") == "x/y/a.jpg"
    prep = Path("root") / k.PREP_DIRNAME
    assert k.tier_dir(prep, "", 1024) == prep / "1024"
    assert k.tier_dir(prep, "x/y", 512) == prep / "x" / "y" / "512"


def _touch(path, data=b"x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def test_scan_tree_skips_underscore_dot_and_qualifies():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _touch(root / "rootimg.jpg")
        _touch(root / "a" / "x.jpg")
        _touch(root / "a" / "x.txt")
        _touch(root / "a" / "deep" / "y.png")
        _touch(root / "a" / "deep" / "odd.tiff")
        _touch(root / "_prep" / "1024" / "old.jpg")   # own output: never source
        _touch(root / "_side" / "z.jpg")
        _touch(root / ".hidden" / "h.jpg")
        (root / "b").mkdir()                          # no images: visited, empty
        pairs, unknown, dirs, _notes = k.scan_tree(root, recursive=True)
        assert [(ds, p.name) for ds, p in pairs] == [
            ("", "rootimg.jpg"), ("a", "x.jpg"), ("a/deep", "y.png")]
        assert unknown == ["a/deep/odd.tiff"]
        assert dirs == ["", "a", "a/deep", "b"]
        # non-recursive: the root only, exactly the old behaviour
        pairs, unknown, dirs, notes = k.scan_tree(root, recursive=False)
        assert [(ds, p.name) for ds, p in pairs] == [("", "rootimg.jpg")]
        assert (dirs, notes) == ([""], [])


def test_scan_tree_refuses_reserved_names():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _touch(root / "a" / "1024" / "x.jpg")         # tier name, any depth
        try:
            k.scan_tree(root, recursive=True)
        except k.ScanError:
            pass
        else:
            raise AssertionError("tier-named source folder was accepted")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _touch(root / "reports" / "x.jpg")            # first level only
        try:
            k.scan_tree(root, recursive=True)
        except k.ScanError:
            pass
        else:
            raise AssertionError("first-level 'reports' folder was accepted")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _touch(root / "a" / "reports" / "x.jpg")      # deeper is fine
        pairs, _u, _d, _n = k.scan_tree(root, recursive=True)
        assert [(ds, p.name) for ds, p in pairs] == [("a/reports", "x.jpg")]


def test_collisions_are_scoped_per_dataset():
    from pathlib import Path
    def named(dataset, filename):
        res = k.Result(path=Path(filename), name=k.qualified_name(dataset, filename),
                       dataset=dataset)
        res.tier = 1024
        return res
    across = [named("a", "photo.jpg"), named("b", "photo.jpg")]
    assert k.resolve_names(across, ".jpg") == []
    assert [r.out_stem for r in across] == ["photo", "photo"]
    within = [named("a", "photo.jpg"), named("a", "photo.png")]
    lines = k.resolve_names(within, ".jpg")
    assert [r.out_stem for r in within] == ["photo", "photo_2"]
    assert len(lines) == 1


def _write_image(path, w, h, color=(120, 90, 60)):
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (w, h), color)
    img.save(path, "JPEG", quality=95)


def test_recursive_end_to_end():
    """Two datasets plus root images -> one _prep mirroring the tree, one TOML
    with a [[datasets]] block per populated leaf, and a clean second run."""
    import contextlib, io, tempfile, tomllib
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_image(root / "rootimg.jpg", 1200, 1200)
        _write_image(root / "a" / "x.jpg", 1200, 1200)
        (root / "a" / "x.txt").write_text("a caption", encoding="utf-8")
        _write_image(root / "a" / "deep" / "y.jpg", 640, 480)
        _write_image(root / "_side" / "z.jpg", 1200, 1200)   # must be ignored
        argv = [str(root), "--recursive", "--threads", "1"]
        with contextlib.redirect_stdout(io.StringIO()):
            assert k.main(argv) == 0
        prep = root / k.PREP_DIRNAME
        assert (prep / "1024" / "rootimg.jpg").is_file()
        assert (prep / "a" / "1024" / "x.jpg").is_file()
        assert (prep / "a" / "1024" / "x.txt").read_text(encoding="utf-8") == "a caption"
        assert (prep / "a" / "deep" / "512" / "y.jpg").is_file()
        assert not list(prep.rglob("z.jpg"))
        with open(prep / k.TOML_FILENAME, "rb") as fh:
            doc = tomllib.load(fh)
        listed = [d["image_directory"] for d in doc["datasets"]]
        assert len(listed) == 3
        for d in listed:
            p = Path(d)
            assert p.is_dir() and any(p.iterdir()), d
        # outputs land on real buckets, and the second run rewrites nothing
        from PIL import Image
        for out in prep.rglob("*.jpg"):
            tier = int(out.parent.name)
            with Image.open(out) as img:
                assert img.size in k.generate_buckets(tier), out
        stamps = {p: p.stat().st_mtime_ns for p in prep.rglob("*.jpg")}
        with contextlib.redirect_stdout(io.StringIO()):
            assert k.main(argv) == 0
        assert {p: p.stat().st_mtime_ns for p in prep.rglob("*.jpg")} == stamps


def test_recursive_refuses_tier_named_folder_end_to_end():
    import contextlib, io, tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_image(root / "768" / "x.jpg", 1200, 1200)
        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            assert k.main([str(root), "--recursive", "--threads", "1"]) == 2
        assert not (root / k.PREP_DIRNAME).exists()


# ---------------------------------------------------------------------------
# --copy-to
# ---------------------------------------------------------------------------

def _detailed_image(path, w, h):
    """Noise upscaled 4x: enough detail to score 10, so the threshold test
    below depends on resolution and flatness alone."""
    import numpy as np
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(w * 10007 + h)
    arr = rng.integers(0, 255, (h // 4, w // 4, 3), dtype=np.uint8)
    Image.fromarray(arr).resize((w, h), Image.BICUBIC).save(path, "JPEG", quality=95)


def _rejects(argv):
    import contextlib, io
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            k.parse_args(argv)
        except SystemExit as exc:
            return exc.code != 0
    return False


def test_copy_to_option_rules():
    assert _rejects(["f", "--copy-to", "t", "--sort"])
    assert _rejects(["f", "--min-res", "1024"])           # needs --copy-to
    assert _rejects(["f", "--copy-to", "t", "--min-res", "-1"])
    assert _rejects(["f", "--copy-to", "t", "--min-res", "1024x768"])
    assert _rejects(["f", "--copy-to", "t", "--move"])
    args = k.parse_args(["f", "--copy-to", "t", "--min-res", "1024", "--recursive"])
    assert (args.copy_to, args.min_res, args.recursive) == ("t", 1024, True)


def test_min_res_is_area():
    from pathlib import Path
    def res(w, h):
        r = k.Result(path=Path("x.jpg"), name="x.jpg")
        r.src_w, r.src_h = w, h
        return r
    assert k.meets_min_res(res(1024, 1024), 1024)
    assert k.meets_min_res(res(1536, 768), 1024)          # wide, same budget
    assert not k.meets_min_res(res(1200, 800), 1024)      # 960,000 px
    assert not k.meets_min_res(res(1023, 1024), 1024)
    assert k.meets_min_res(res(1, 1), 0)


def test_copy_target_must_not_overlap_the_source():
    from pathlib import Path
    src = Path("C:/data/src")
    assert k.check_copy_target(src, src) is not None
    assert k.check_copy_target(src, Path("C:/data")) is not None
    assert k.check_copy_target(src, src / "picked") is not None
    assert k.check_copy_target(src, src / k.PREP_DIRNAME / "x") is not None
    assert k.check_copy_target(src, src / "_picked" / "x") is None
    assert k.check_copy_target(src, Path("C:/data/src2")) is None
    assert k.check_copy_target(src, Path("D:/picked")) is None


def test_copy_to_end_to_end():
    """Both gates, a mirrored tree, verbatim copies with captions, a dry run
    that writes nothing, idempotent re-runs, and a conflict that is reported
    rather than overwritten."""
    import contextlib, io, tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        src, out = Path(tmp) / "src", Path(tmp) / "out"
        _detailed_image(src / "top.jpg", 1600, 1200)
        (src / "top.txt").write_text("top", encoding="utf-8")
        _detailed_image(src / "a" / "deep" / "pano.jpg", 1536, 768)
        _detailed_image(src / "a" / "small.jpg", 800, 600)      # too small
        _write_image(src / "a" / "flat.jpg", 2000, 2000)        # scores 1
        _detailed_image(src / "1024" / "tier.jpg", 1400, 1400)  # name is fine here
        _detailed_image(src / "_skip" / "x.jpg", 2000, 2000)    # never scanned
        argv = [str(src), "--recursive", "--copy-to", str(out),
                "--min-res", "1024", "--threshold", "5", "--threads", "1"]

        with contextlib.redirect_stdout(io.StringIO()):
            assert k.main(argv + ["--report"]) == 0
        assert not out.exists()

        with contextlib.redirect_stdout(io.StringIO()):
            assert k.main(argv) == 0
        got = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
        assert got == ["1024/tier.jpg", "a/deep/pano.jpg", "top.jpg", "top.txt"], got
        for rel in got:
            assert (out / rel).read_bytes() == (src / rel).read_bytes(), rel

        stamps = {p: p.stat().st_mtime_ns for p in out.rglob("*")}
        with contextlib.redirect_stdout(io.StringIO()):
            assert k.main(argv) == 0
        assert {p: p.stat().st_mtime_ns for p in out.rglob("*")} == stamps

        (out / "top.txt").write_text("edited by hand", encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            assert k.main(argv) == 1
        assert (out / "top.txt").read_text(encoding="utf-8") == "edited by hand"
        with contextlib.redirect_stdout(io.StringIO()):
            assert k.main(argv + ["--force"]) == 0
        assert (out / "top.txt").read_text(encoding="utf-8") == "top"

        # A tighter run leaves earlier copies alone and names them.
        tighter = [a if a != "1024" else "1300" for a in argv]
        with contextlib.redirect_stdout(io.StringIO()):
            assert k.main(tighter) == 0
        assert (out / "a" / "deep" / "pano.jpg").is_file()
        report = max((src / k.PREP_DIRNAME / k.REPORTS_DIRNAME).glob("copy-2*.txt"),
                     key=lambda p: p.stat().st_mtime_ns)
        text = report.read_text(encoding="utf-8")
        assert "IN THE TARGET BUT NOT SELECTED  (1)" in text
        assert "a/deep/pano.jpg" in text.split("IN THE TARGET BUT NOT SELECTED")[1]

        # The grid has a row for the run's own --min-res, and its cell at the
        # run's threshold is starred and equals what the run selected.
        grid = text.split("SELECTION GRID  (")[1].split("\n\n")[0].splitlines()
        row = next(line.split() for line in grid if line.split()[:1] == ["1300"])
        assert row[5] == "2*", row                 # threshold 5: top, tier
        assert "\nSELECTED  (2 images)" in text


def test_selection_grid_counts_every_combination():
    from datetime import datetime
    from pathlib import Path
    def res(name, w, h, score):
        r = k.Result(path=Path(name), name=name)
        r.src_w, r.src_h, r.composite, r.scored = w, h, score, True
        return r
    results = [res("a", 2048, 2048, 9), res("b", 1024, 1024, 5),
               res("c", 800, 600, 9), res("d", 1536, 768, 2)]
    args = k.parse_args(["f", "--copy-to", "t"])
    text = k.build_copy_report(args, Path("f"), Path("t"), results, [], [],
                               datetime.now(), datetime.now(), 0.0, [])
    grid = text.split("SELECTION GRID  (")[1].split("\n\n")[0].splitlines()
    rows = {line.split()[0]: line.split()[1:] for line in grid[2:]}
    assert rows["any"][0] == "4*"                  # threshold 1, no gate
    assert rows["any"][8] == "2"                   # >= 9: a, c
    assert rows["1024"][0] == "3"                  # a, b, d (1536x768 = 1024^2 + 12.5%)
    assert rows["1024"][4] == "2"                  # >= 5: a, b
    assert rows["2048"] == ["1"] * 9 + ["0"]       # a only, score 9


def main():
    tests =[v for name, v in sorted(globals().items()) if name.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {fn.__name__}: {exc}")
        else:
            print(f"ok    {fn.__name__}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
