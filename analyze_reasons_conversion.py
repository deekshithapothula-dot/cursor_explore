"""
Reason Tokenization & High-Conversion Analysis
================================================

Purpose
-------
Many "Row Labels" columns in pivot tables actually contain MULTIPLE reasons
packed into a single cell (e.g. separated by commas, slashes, "and", "&",
semicolons, or pipes). This script:

1. Reads a CSV/Excel export (the same data behind your pivot table).
2. Splits ("tokenizes") each Row Label into individual reasons using regex.
3. Explodes rows so each reason gets its own row (while keeping the
   counts/conversions associated with the original row).
4. Aggregates volume + conversions per individual reason.
5. Ranks reasons by conversion rate (with a minimum volume filter so rare
   reasons with tiny sample sizes don't dominate the "top" list misleadingly).
6. Also mines free-text reasons with a word/phrase frequency analysis
   (n-grams) in case reasons are written as sentences rather than short tags.

Usage
-----
    python analyze_reasons_conversion.py --input data.csv \
        --reason-col "Row Labels" \
        --count-col "Count of ..." \
        --conversion-col "Converted"  \
        --min-volume 10

Input expectations
-------------------
Your input file should have (at minimum):
  - A column with the reason text (can contain multiple reasons per cell).
  - EITHER:
      a) a column with total volume/count per row AND a column with the
         number/rate of conversions per row, OR
      b) a single 0/1 "converted" column at the row (transaction) level,
         in which case leave --count-col unset and this script will treat
         each row as one unit with count=1.

If you only have the pivoted summary (Row Labels + Count + Conversion Rate),
that's fine too -- pass --count-col and --conversion-rate-col instead of
--conversion-col (see --help).

Output
------
  - reasons_by_conversion.csv : one row per atomic reason with volume,
    conversions, and conversion rate, sorted descending by conversion rate.
  - reason_ngrams.csv : top word/phrase n-grams extracted from the free text,
    useful when reasons are written as sentences instead of short tags.
"""

import argparse
import re
import sys
from collections import Counter

import pandas as pd

# Regex used to split a single cell into multiple candidate reasons.
# Splits on commas, semicolons, slashes, pipes, " and ", " & ", " / ", newlines.
SPLIT_PATTERN = re.compile(
    r"\s*(?:,|;|\||/|\n|\band\b|&)\s*",
    flags=re.IGNORECASE,
)

# Regex to clean up a token: strip whitespace, trailing punctuation, normalize case/spacing.
CLEAN_PATTERN = re.compile(r"^[\s\-\.]+|[\s\-\.]+$")
MULTISPACE_PATTERN = re.compile(r"\s+")

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "was", "were", "be", "been", "it", "this", "that", "at", "by",
    "as", "not", "no", "due", "because",
}


def tokenize_reason_cell(cell: str):
    """Split a single Row Label cell into a list of cleaned atomic reasons."""
    if cell is None:
        return []
    text = str(cell).strip()
    if not text:
        return []
    parts = SPLIT_PATTERN.split(text)
    cleaned = []
    for part in parts:
        part = CLEAN_PATTERN.sub("", part)
        part = MULTISPACE_PATTERN.sub(" ", part).strip()
        if part:
            cleaned.append(part.lower())
    return cleaned if cleaned else [text.lower()]


def explode_reasons(df: pd.DataFrame, reason_col: str) -> pd.DataFrame:
    df = df.copy()
    df["_reasons"] = df[reason_col].apply(tokenize_reason_cell)
    exploded = df.explode("_reasons").rename(columns={"_reasons": "reason"})
    exploded = exploded[exploded["reason"].notna() & (exploded["reason"] != "")]
    return exploded


def aggregate_conversion(
    exploded: pd.DataFrame,
    count_col: str | None,
    conversion_col: str | None,
    conversion_rate_col: str | None,
    min_volume: int,
) -> pd.DataFrame:
    if count_col is None:
        exploded = exploded.copy()
        exploded["_count"] = 1
        count_col = "_count"

    if conversion_rate_col is not None:
        # Weighted average of an existing rate column, weighted by volume.
        exploded = exploded.copy()
        exploded["_conversions"] = exploded[conversion_rate_col].astype(float) * exploded[count_col].astype(float)
        conversion_col = "_conversions"
    elif conversion_col is None:
        raise ValueError("Provide either --conversion-col or --conversion-rate-col")

    grouped = (
        exploded.groupby("reason")
        .agg(volume=(count_col, "sum"), conversions=(conversion_col, "sum"))
        .reset_index()
    )
    grouped["conversion_rate"] = grouped["conversions"] / grouped["volume"].replace(0, pd.NA)
    grouped = grouped[grouped["volume"] >= min_volume]
    grouped = grouped.sort_values("conversion_rate", ascending=False)
    return grouped


def word_ngrams(text: str, n: int):
    words = [w for w in re.findall(r"[a-zA-Z0-9']+", text.lower()) if w not in STOPWORDS]
    return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]


def ngram_frequency(series: pd.Series, n_values=(1, 2, 3), top_k=30) -> pd.DataFrame:
    rows = []
    for n in n_values:
        counter = Counter()
        for text in series.dropna().astype(str):
            counter.update(word_ngrams(text, n))
        for phrase, freq in counter.most_common(top_k):
            rows.append({"n": n, "phrase": phrase, "frequency": freq})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Path to CSV or Excel file")
    parser.add_argument("--sheet", default=0, help="Sheet name/index for Excel input")
    parser.add_argument("--reason-col", default="Row Labels", help="Column containing (possibly multi-)reason text")
    parser.add_argument("--count-col", default=None, help="Column with volume/count per row. Omit if each row = 1 unit")
    parser.add_argument("--conversion-col", default=None, help="Column with number of conversions per row")
    parser.add_argument("--conversion-rate-col", default=None, help="Column with an existing conversion RATE per row (0-1)")
    parser.add_argument("--min-volume", type=int, default=10, help="Minimum volume required for a reason to be ranked")
    parser.add_argument("--out-prefix", default="reasons", help="Prefix for output CSV files")
    args = parser.parse_args()

    if args.input.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(args.input, sheet_name=args.sheet)
    else:
        df = pd.read_csv(args.input)

    if args.reason_col not in df.columns:
        sys.exit(f"Column '{args.reason_col}' not found. Available columns: {list(df.columns)}")

    exploded = explode_reasons(df, args.reason_col)

    ranked = aggregate_conversion(
        exploded,
        count_col=args.count_col,
        conversion_col=args.conversion_col,
        conversion_rate_col=args.conversion_rate_col,
        min_volume=args.min_volume,
    )
    ranked_path = f"{args.out_prefix}_by_conversion.csv"
    ranked.to_csv(ranked_path, index=False)
    print(f"Wrote {ranked_path} ({len(ranked)} reasons, min_volume={args.min_volume})")
    print(ranked.head(20).to_string(index=False))

    ngrams = ngram_frequency(df[args.reason_col])
    ngram_path = f"{args.out_prefix}_ngrams.csv"
    ngrams.to_csv(ngram_path, index=False)
    print(f"\nWrote {ngram_path} (top n-grams across all reason text)")


if __name__ == "__main__":
    main()
