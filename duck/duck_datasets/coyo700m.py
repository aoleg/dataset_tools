"""COYO-700M (kakaobrain/coyo-700m): 747M alt-text image pairs, 128 Parquet files, 126 GB."""

from duck_engine import DatasetSpec, NumericFilter, OutputColumn, RangeFilter, TextColumn

SPEC = DatasetSpec(
    name="coyo700m",
    title="COYO-700M",
    hf_id="kakaobrain/coyo-700m",
    file_glob="*.parquet",
    quick_test_glob="part-00000-*.parquet",
    text_columns=[
        TextColumn("text", "text", "Terms that must appear in the alt text (same as --caption)."),
    ],
    default_caption="text",
    range_filters=[
        RangeFilter("faces", "num_faces", "Face count: exact (1) or inclusive range (1-2)."),
    ],
    numeric_filters=[
        NumericFilter("min-width", "width", ">=", int, "Minimum width in pixels."),
        NumericFilter("min-height", "height", ">=", int, "Minimum height in pixels."),
        NumericFilter("min-aesthetic", "aesthetic_score_laion_v2", ">=", float, "Minimum aesthetic_score_laion_v2 (0-10)."),
        NumericFilter("max-watermark", "watermark_score", "<=", float, "Maximum watermark_score (0-1)."),
        NumericFilter("max-nsfw", "nsfw_score_opennsfw2", "<=", float, "Maximum nsfw_score_opennsfw2 (0-1)."),
        NumericFilter("min-clip", "clip_similarity_vitb32", ">=", float,
                      "Minimum clip_similarity_vitb32, i.e. how well the alt text fits the image."),
    ],
    output_columns=[
        OutputColumn("width", "width"),
        OutputColumn("height", "height"),
        OutputColumn("num_faces", "num_faces"),
        OutputColumn("clip_similarity_vitb32", "clip_similarity_vitb32"),
        OutputColumn("nsfw_score_opennsfw2", "nsfw_score_opennsfw2"),
        OutputColumn("watermark_score", "watermark_score"),
        OutputColumn("aesthetic_score_laion_v2", "aesthetic_score_laion_v2"),
        OutputColumn("id", "id"),
    ],
    order_by="id",
    notes="Alt text only, no synthetic captions. A full scan reads 126 GB; use --count or --quick-test first.",
)
