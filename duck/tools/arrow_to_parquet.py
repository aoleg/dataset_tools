"""Convert a HuggingFace save_to_disk Arrow dataset into Parquet.

The .arrow shards written by HF datasets are Arrow IPC *streaming* format, so
pyarrow.ipc.open_stream is the only reader that opens them: open_file,
feather.read_table and dataset(format="arrow") all reject them.

Output is one Parquet file per input shard, named part-NNNNN-of-NNNNN.parquet so
duck's --quick-test glob (part-00000-*.parquet) works.
"""

import argparse
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq


def human(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def read_schema(path):
    with pa.OSFile(path, "rb") as f:
        return ipc.open_stream(f).schema


def convert_shard(src_path, dst_path, compression, row_group_size):
    """Convert one Arrow shard. Returns (rows, bytes_written).

    The source batches are tiny (650-1000 rows), so they are buffered up to
    row_group_size before being written. Writing them straight through would
    produce a row group per batch, and row-group statistics that granular are
    useless for skipping data.
    """
    schema = read_schema(src_path)
    tmp_path = dst_path + ".tmp"
    rows = 0
    buf = []
    buffered = 0

    writer = pq.ParquetWriter(tmp_path, schema, compression=compression, write_statistics=True)
    try:
        with pa.OSFile(src_path, "rb") as f:
            for batch in ipc.open_stream(f):
                buf.append(batch)
                buffered += batch.num_rows
                rows += batch.num_rows
                if buffered >= row_group_size:
                    table = pa.Table.from_batches(buf, schema=schema)
                    # Write whole row groups only and carry the remainder over. Handing
                    # write_table a buffer that merely overshoots the target makes it
                    # emit the overshoot as its own tiny row group.
                    full = (table.num_rows // row_group_size) * row_group_size
                    writer.write_table(table.slice(0, full), row_group_size=row_group_size)
                    tail = table.slice(full)
                    buf = tail.to_batches() if tail.num_rows else []
                    buffered = tail.num_rows
        if buf:
            writer.write_table(pa.Table.from_batches(buf, schema=schema), row_group_size=row_group_size)
    except BaseException:
        writer.close()
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    writer.close()

    # only now does the output become visible under its real name, so an
    # interrupted run never leaves a truncated file that --skip-existing trusts
    os.replace(tmp_path, dst_path)
    return rows, os.path.getsize(dst_path)


def main():
    parser = argparse.ArgumentParser(description="Convert HF Arrow shards to Parquet.")
    parser.add_argument("--src", required=True, help="Folder holding the .arrow shards (the dataset repo's data folder).")
    parser.add_argument("--dst", default=None, help="Folder to write .parquet files into (default: a 'parquet' folder next to --src).")
    parser.add_argument("--compression", default="zstd", choices=["zstd", "snappy", "gzip", "none"])
    parser.add_argument("--row-group-size", type=int, default=131072, help="Rows per Parquet row group.")
    parser.add_argument("--workers", type=int, default=4, help="Shards to convert in parallel.")
    parser.add_argument("--overwrite", action="store_true", help="Redo shards that already have output.")
    parser.add_argument("--limit-shards", type=int, default=0, help="Only convert the first N shards (for testing).")
    args = parser.parse_args()

    src_dir = os.path.normpath(args.src)
    dst_dir = os.path.normpath(args.dst) if args.dst else os.path.join(os.path.dirname(src_dir), "parquet")

    shards = sorted(f for f in os.listdir(src_dir) if f.endswith(".arrow"))
    if not shards:
        print(f"No .arrow files found in {src_dir}")
        sys.exit(1)
    shards_available = len(shards)
    if args.limit_shards:
        shards = shards[: args.limit_shards]

    os.makedirs(dst_dir, exist_ok=True)
    total = len(shards)
    compression = None if args.compression == "none" else args.compression

    jobs = []
    skipped = 0
    for i, name in enumerate(shards):
        out_name = f"part-{i:05d}-of-{total:05d}.parquet"
        out_path = os.path.join(dst_dir, out_name)
        if os.path.exists(out_path) and not args.overwrite:
            skipped += 1
            continue
        jobs.append((os.path.join(src_dir, name), out_path, out_name))

    print(f"Source      : {src_dir}  ({total} shards)")
    print(f"Destination : {dst_dir}")
    print(f"Settings    : compression={args.compression}, row_group_size={args.row_group_size:,}, workers={args.workers}")
    if skipped:
        print(f"Skipping    : {skipped} shard(s) already converted (use --overwrite to redo)")
    if not jobs:
        print("\nNothing to do.")
        return
    print(f"Converting  : {len(jobs)} shard(s)\n")

    done = 0
    rows_total = 0
    bytes_total = 0
    failures = []
    start = time.perf_counter()

    def run(job):
        src_path, out_path, out_name = job
        return out_name, convert_shard(src_path, out_path, compression, args.row_group_size)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for future in [pool.submit(run, j) for j in jobs]:
            try:
                out_name, (rows, size) = future.result()
            except Exception as e:
                failures.append(str(e))
                print(f"  FAILED: {e}")
                traceback.print_exc()
                continue
            done += 1
            rows_total += rows
            bytes_total += size
            elapsed = time.perf_counter() - start
            rate = done / elapsed if elapsed else 0
            eta = (len(jobs) - done) / rate if rate else 0
            print(
                f"  [{done:>3}/{len(jobs)}] {out_name}  {rows:>9,} rows  {human(size):>9}"
                f"   {elapsed/60:5.1f}m elapsed, ~{eta/60:.1f}m left"
            )

    elapsed = time.perf_counter() - start
    print(f"\nConverted {done}/{len(jobs)} shard(s) in {elapsed/60:.1f} min")
    print(f"  rows written : {rows_total:,}")
    print(f"  output size  : {human(bytes_total)}")
    if failures:
        print(f"  FAILURES     : {len(failures)}")
        sys.exit(1)

    # Verify against the Parquet footers rather than trusting the counters above.
    print("\nVerifying output...")
    files = sorted(f for f in os.listdir(dst_dir) if f.endswith(".parquet"))
    verified = 0
    groups = 0
    for f in files:
        md = pq.ParquetFile(os.path.join(dst_dir, f)).metadata
        verified += md.num_rows
        groups += md.num_row_groups
    print(f"  {len(files)} parquet file(s), {verified:,} rows, {groups} row groups")

    expected = 0
    info = os.path.join(src_dir, "dataset_info.json")
    if os.path.exists(info):
        import json

        with open(info, "r", encoding="utf-8") as fh:
            splits = json.load(fh).get("splits", {})
        expected = sum(s.get("num_examples", 0) for s in splits.values())

    # dataset_info.json counts the whole dataset, so it is only a fair comparison
    # when every shard was converted.
    if expected and len(files) == shards_available:
        if verified == expected:
            print(f"  OK: matches dataset_info.json ({expected:,} rows)")
        else:
            print(f"  MISMATCH: dataset_info.json says {expected:,} rows, output has {verified:,}")
            sys.exit(1)
    elif expected:
        print(f"  (partial run: {len(files)}/{shards_available} shards present, full dataset is {expected:,} rows)")


if __name__ == "__main__":
    main()
