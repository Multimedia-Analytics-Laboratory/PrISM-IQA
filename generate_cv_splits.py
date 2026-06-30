import argparse
import csv
import os
import random


def read_rows(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def write_rows(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def split_rows(rows, train_ratio, val_ratio, seed):
    rows = sorted(rows, key=lambda row: row["image"])
    rng = random.Random(seed)
    rng.shuffle(rows)

    n_total = len(rows)
    n_train = int(round(n_total * train_ratio))
    n_val = int(round(n_total * val_ratio))
    train_rows = rows[:n_train]
    val_rows = rows[n_train:n_train + n_val]
    test_rows = rows[n_train + n_val:]
    return train_rows, val_rows, test_rows


def parse_args():
    parser = argparse.ArgumentParser(description="Generate PrISM-IQA train/val/test CSV splits.")
    parser.add_argument("--master-csv", required=True, help="CSV with image plus PrISM-IQA label columns.")
    parser.add_argument("--output-root", default="csv_file_merged", help="Directory to write split folders.")
    parser.add_argument("--num-splits", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20200626)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    return parser.parse_args()


def main():
    args = parse_args()
    fieldnames, rows = read_rows(args.master_csv)
    if "image" not in fieldnames:
        raise ValueError("master CSV must contain an 'image' column")
    if args.train_ratio <= 0 or args.val_ratio <= 0 or args.train_ratio + args.val_ratio >= 1:
        raise ValueError("expected positive ratios with train_ratio + val_ratio < 1")

    for split_id in range(args.num_splits):
        split_dir = os.path.join(args.output_root, f"split{split_id}")
        train_rows, val_rows, test_rows = split_rows(
            rows,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed + split_id,
        )
        write_rows(os.path.join(split_dir, "train.csv"), fieldnames, train_rows)
        write_rows(os.path.join(split_dir, "val.csv"), fieldnames, val_rows)
        write_rows(os.path.join(split_dir, "test.csv"), fieldnames, test_rows)
        print(
            f"split{split_id}: train={len(train_rows)} val={len(val_rows)} "
            f"test={len(test_rows)} -> {split_dir}"
        )


if __name__ == "__main__":
    main()
