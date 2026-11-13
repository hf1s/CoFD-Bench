#!/usr/bin/env python3
"""Add site_correct and bond_correct columns to pdb_benchmark.csv.

site_correct : the existing Top-1-correct definition, carried over unchanged --
               the closest nucleophile of the matching residue type in the
               Top-1 pose is the deposited one.
bond_correct : site_correct AND the predicted warhead-to-nucleophile distance
               is within the range of a deposited covalent bond in this
               benchmark (Top-1-distance <= max(real_distance)).
"""
import csv
import sys

src, dst = sys.argv[1], sys.argv[2]

with open(src, newline="") as fh:
    reader = csv.DictReader(fh)
    fields = list(reader.fieldnames)
    rows = list(reader)


def num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# The threshold is the benchmark's own ground truth: the longest deposited
# warhead-to-nucleophile distance anywhere in the set.
deposited = [d for d in (num(r["real_distance"]) for r in rows) if d is not None]
threshold = max(deposited)

for r in rows:
    call = r["Top-1-correct"].strip()
    if call not in ("True", "False"):
        # Top-1-correct is blank for this entry upstream; leave both derived
        # columns blank rather than asserting a result.
        r["site_correct"] = ""
        r["bond_correct"] = ""
        continue
    site = call == "True"
    predicted = num(r["Top-1-distance"])
    r["site_correct"] = str(site)
    r["bond_correct"] = str(site and predicted is not None and predicted <= threshold)

for col in ("site_correct", "bond_correct"):
    if col not in fields:
        fields.append(col)

with open(dst, "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)

# ---- report -------------------------------------------------------------
site_n = sum(1 for r in rows if r["site_correct"] == "True")
bond_n = sum(1 for r in rows if r["bond_correct"] == "True")
missing = sum(1 for r in rows if num(r["Top-1-distance"]) is None)

print(f"entries                 : {len(rows)}")
print(f"deposited bond range    : {min(deposited):.2f} - {threshold:.2f} A")
print(f"site_correct = True     : {site_n}")
print(f"bond_correct = True     : {bond_n}")
print(f"site but not bond       : {site_n - bond_n}")
print(f"non-numeric Top-1-dist  : {missing}")
print()
print(f"{'residue':<8}{'n':>5}{'site':>7}{'bond':>7}{'site only':>11}")
seen = []
for r in rows:
    if r["res_name"] not in seen:
        seen.append(r["res_name"])
for res in seen:
    sub = [r for r in rows if r["res_name"] == res]
    s = sum(1 for r in sub if r["site_correct"] == "True")
    b = sum(1 for r in sub if r["bond_correct"] == "True")
    print(f"{res:<8}{len(sub):>5}{s:>7}{b:>7}{s - b:>11}")
