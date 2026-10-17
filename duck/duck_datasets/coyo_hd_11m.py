"""coyo-hd-11m-llavanext (CaptionEmporium): 11.4M COYO images with LLaVA captions and tags.

Hugging Face ships this one as Arrow IPC shards; run tools/arrow_to_parquet.py once
and point duck.conf at the resulting parquet folder.
"""

from duck_engine import DatasetSpec, NumericFilter, OutputColumn, RangeFilter, TextColumn

SPEC = DatasetSpec(
    name="coyo-hd-11m",
    title="coyo-hd-11m-llavanext",
    hf_id="CaptionEmporium/coyo-hd-11m-llavanext",
    file_glob="*.parquet",
    quick_test_glob="part-00000-*.parquet",
    text_columns=[
        TextColumn("caption_llava", "caption_llava", "Terms that must appear in the long LLaVA caption."),
        TextColumn("caption_llava_short", "caption_llava_short",
                   "Terms that must appear in the one-sentence LLaVA caption (same as --caption)."),
        # The dataset's own "caption" column is the original alt text. It is exposed
        # as caption_original so that --caption can mean "the default caption" on
        # every dataset, and because img2dataset renames its --caption_col to
        # "caption" and dies on a second column of that name.
        TextColumn("caption", "caption_original", "Terms that must appear in the original alt text."),
        TextColumn("tags_open_images", "tags_open_images",
                   'Terms that must appear in the Open Images tags, e.g. "Horse" (substring: "Cap" also hits "Baseball cap").'),
        TextColumn("tags_booru", "tags_booru",
                   'Terms that must appear in the booru tags, e.g. "baseball_cap" (substring: use the full tag).'),
    ],
    default_caption="caption_llava_short",
    range_filters=[
        RangeFilter("faces", "num_faces", "Face count: exact (1) or inclusive range (1-2)."),
    ],
    numeric_filters=[
        NumericFilter("min-width", "width", ">=", int, "Minimum width in pixels."),
        NumericFilter("min-height", "height", ">=", int, "Minimum height in pixels."),
        NumericFilter("min-aesthetic", "aesthetic_score_laion_v2", ">=", float, "Minimum aesthetic_score_laion_v2 (0-10)."),
        NumericFilter("max-watermark", "watermark_score", "<=", float, "Maximum watermark_score (0-1)."),
        NumericFilter("max-nsfw", "nsfw_score_opennsfw2", "<=", float, "Maximum nsfw_score_opennsfw2 (0-1)."),
    ],
    output_columns=[
        OutputColumn("caption_llava", "caption_llava", True),
        OutputColumn("caption_llava_short", "caption_llava_short", True),
        OutputColumn("caption_original", "caption", True),
        OutputColumn("tags_open_images", "tags_open_images", True),
        OutputColumn("tags_booru", "tags_booru", True),
        OutputColumn("num_faces", "num_faces"),
        OutputColumn("width", "width"),
        OutputColumn("height", "height"),
        OutputColumn("aesthetic_score_laion_v2", "aesthetic_score_laion_v2"),
        OutputColumn("watermark_score", "watermark_score"),
        OutputColumn("nsfw_score_opennsfw2", "nsfw_score_opennsfw2"),
        OutputColumn("key", "key"),
    ],
    order_by="key",
    notes="Every image is at least 512 px on the short edge. Needs tools/arrow_to_parquet.py once.",
)
