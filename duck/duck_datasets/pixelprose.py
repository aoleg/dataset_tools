"""PixelProse (tomg-group-umd/pixelprose): 16.9M dense Gemini captions in three splits."""

from duck_engine import DatasetSpec, NumericFilter, OutputColumn, RangeFilter, TextColumn

# 97% of vlm_captions begin " This image displays: ", which is pure boilerplate
# in a training caption. About 1.4% repeat it ("This image displays:\n\nThis
# image displays a ..."), so the group is matched one-or-more times.
PREAMBLE_RE = r"^(\s*This image displays\s*:?\s*)+"


def add_arguments(parser):
    group = parser.add_argument_group("pixelprose")
    group.add_argument("--strip-preamble", default="yes", choices=["yes", "no"],
                       help="Remove the ' This image displays: ' boilerplate from vlm_caption (default: yes).")


def caption_expr(column, args):
    if args.strip_preamble == "yes" and column == "vlm_caption":
        return f"trim(regexp_replace({column}, ?, '', 'i'))", [PREAMBLE_RE]
    return column, []


SPEC = DatasetSpec(
    name="pixelprose",
    title="PixelProse",
    hf_id="tomg-group-umd/pixelprose",
    file_glob="vlm_captions_{split}*.parquet",
    # "*" may be empty, so this also fits a named split: vlm_captions_redcaps_00.parquet
    quick_test_glob="vlm_captions_{split}*00.parquet",
    # split name -> filename infix
    splits={"all": "", "cc12m": "cc12m_", "commonpool": "common-pool_", "redcaps": "redcaps_"},
    text_columns=[
        TextColumn("vlm_caption", "vlm_caption", "Terms that must appear in the dense Gemini caption (same as --caption)."),
        TextColumn("original_caption", "original_caption", "Terms that must appear in the source caption."),
    ],
    default_caption="vlm_caption",
    range_filters=[
        RangeFilter("watermark-class", "watermark_class_id",
                    "0 = has watermark, 1 = clean, 2 = no watermark but has text. Range ok (1-2)."),
    ],
    numeric_filters=[
        # original_width/original_height are the real image dimensions; the plain
        # width/height are the resized copy fed to the captioning model.
        NumericFilter("min-width", "original_width", ">=", int, "Minimum original_width in pixels."),
        NumericFilter("min-height", "original_height", ">=", int, "Minimum original_height in pixels."),
        NumericFilter("min-aesthetic", "aesthetic_score", ">=", float, "Minimum aesthetic_score (0-10)."),
        NumericFilter("max-watermark-score", "watermark_class_score", "<=", float,
                      "Maximum watermark_class_score, the confidence of the predicted class (0-1)."),
    ],
    output_columns=[
        OutputColumn("vlm_caption", "vlm_caption", True),
        OutputColumn("original_caption", "original_caption", True),
        OutputColumn("watermark_class_id", "watermark_class_id"),
        OutputColumn("watermark_class_score", "watermark_class_score"),
        OutputColumn("aesthetic_score", "aesthetic_score"),
        OutputColumn("width", "original_width"),
        OutputColumn("height", "original_height"),
        OutputColumn("uid", "uid"),
    ],
    # uid/key/url are not unique on their own, so order by all three to keep
    # --start paging over a stable sequence.
    order_by="uid, key, url",
    add_arguments=add_arguments,
    caption_expr=caption_expr,
    notes="Splits: cc12m (9.1M), commonpool (6.5M), redcaps (1.3M, 82% of it >= 1024 px).",
)
