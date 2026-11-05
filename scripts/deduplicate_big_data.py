import os
import hashlib
import json
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET_DIR = os.path.join(PROJECT_ROOT, "big_data_test")
CHECKPOINT_FILE = os.path.join(TARGET_DIR, "progress.json")

SPLITS = ["train", "val", "test"]
CLASSES = ["mask_proper", "mask_improper", "no_mask"]


def deduplicate_dataset():
    print("=" * 70)
    print("      BIG DATA TEST: EXACT PIXEL DEDUPLICATION & LEAKAGE PRUNING")
    print("=" * 70)

    hashes = defaultdict(list)
    total_scanned = 0

    print("\n[Step 1/3] Scanning dataset and computing byte-exact file hashes...")
    for split in SPLITS:
        for cls in CLASSES:
            folder = os.path.join(TARGET_DIR, split, cls)
            if not os.path.exists(folder):
                continue
            for fname in os.listdir(folder):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                fpath = os.path.join(folder, fname)
                total_scanned += 1
                try:
                    with open(fpath, "rb") as f:
                        h = hashlib.md5(f.read()).hexdigest()
                    hashes[h].append((split, cls, fpath))
                except Exception as e:
                    print(f"  [Warning] Could not read {fpath}: {e}")

    print(f"Total images scanned : {total_scanned:,}")
    print(f"Unique image hashes  : {len(hashes):,}")

    # Determine files to delete
    # 1. Cross-class conflicts: If an image is labeled as multiple different classes, remove ALL copies (unreliable label)
    # 2. Cross-split duplicates: Keep in 'train', remove from 'val' and 'test' (prevent leakage)
    # 3. Within-split duplicates: Keep 1 copy, remove remainder
    to_delete = []
    conflict_count = 0
    leakage_count = 0
    redundant_count = 0

    print("\n[Step 2/3] Analyzing hash collisions and label consistency...")
    for h, entries in hashes.items():
        if len(entries) <= 1:
            continue

        unique_classes = set(e[1] for e in entries)
        if len(unique_classes) > 1:
            # Ambiguous/conflicting label across source datasets -> purge all
            conflict_count += len(entries)
            for split, cls, fpath in entries:
                to_delete.append((fpath, f"Conflict: labeled as both {unique_classes}"))
            continue

        # Same class, but multiple entries exist
        # Sort so 'train' comes first, then 'val', then 'test'
        def split_priority(entry):
            split = entry[0]
            if split == "train":
                return 0
            if split == "val":
                return 1
            return 2

        sorted_entries = sorted(entries, key=split_priority)
        # Keep the very first occurrence
        keeper = sorted_entries[0]

        for dup in sorted_entries[1:]:
            fpath = dup[2]
            if dup[0] != keeper[0]:
                leakage_count += 1
                reason = f"Cross-split leakage (already in {keeper[0]})"
            else:
                redundant_count += 1
                reason = f"Duplicate within {dup[0]}/{dup[1]}"
            to_delete.append((fpath, reason))

    print(f"Identified {len(to_delete):,} files to prune:")
    print(f"  - Conflicting cross-class labels purged : {conflict_count:,}")
    print(f"  - Cross-split leakage pruned           : {leakage_count:,}")
    print(f"  - Redundant identical copies pruned    : {redundant_count:,}")

    # Execute Deletions
    print(f"\n[Step 3/3] Pruning {len(to_delete):,} duplicate and contaminated files...")
    deleted_count = 0
    for fpath, reason in to_delete:
        try:
            if os.path.exists(fpath):
                os.remove(fpath)
                deleted_count += 1
        except Exception as e:
            print(f"  [Error] Failed to remove {fpath}: {e}")

    print(f"Successfully pruned {deleted_count:,} files from disk.")

    # Audit final clean counts and sync progress.json
    final_counts = {split: {cls: 0 for cls in CLASSES} for split in SPLITS}
    final_total = 0
    print("\n" + "=" * 65)
    print("         BIG_DATA_TEST FINAL CLEAN DATASET AUDIT")
    print("=" * 65)
    for split in SPLITS:
        print(f"\nSplit: {split.upper()}")
        subtotal = 0
        for cls in CLASSES:
            folder = os.path.join(TARGET_DIR, split, cls)
            count = len(os.listdir(folder)) if os.path.exists(folder) else 0
            final_counts[split][cls] = count
            subtotal += count
            final_total += count
            print(f"  {cls:<16}: {count:,} clean unique images")
        print(f"  Subtotal {split:<10}: {subtotal:,} images")
    print("-" * 65)
    print(f"Grand Total Clean Images : {final_total:,} images")
    print("=" * 65 + "\n")

    # Update progress.json
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r") as f:
                data = json.load(f)
        except Exception:
            data = {}
        data["counts"] = final_counts
        data["deduplication_completed"] = True
        data["clean_total"] = final_total
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump(data, f, indent=2)
        print("Updated big_data_test/progress.json with clean audit figures.")


if __name__ == "__main__":
    deduplicate_dataset()
