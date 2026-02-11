import json
from pathlib import Path
from math import prod, ceil
import copy

# ---------------- CONFIG ----------------
OUTPUT_DIR = "doe_parts"
N_FILES = 200

PARAMS_ORDER = [
    "hb_hl_upper", "hb_hl_lower", "hb_ll_upper", "hb_ll_lower",
    "hb_gen_slope_upper", "hb_gen_slope_lower",
    "hb_gen_macd_slope_upper", "hb_gen_macd_slope_lower",
    "date_gap_upper", "date_gap_lower",
]

PARAMS = {
    "hb_hl_upper": {"min": 40, "max": 90, "step": 5},
    "hb_hl_lower": {"min": 40, "max": 90, "step": 5},
    "hb_ll_upper": {"min": 40, "max": 90, "step": 5},
    "hb_ll_lower": {"min": 40, "max": 90, "step": 5},
    "hb_gen_slope_upper": {"min": 0, "max": 70, "step": 5},
    "hb_gen_slope_lower": {"min": 0, "max": 70, "step": 5},
    "hb_gen_macd_slope_upper": {"min": 0, "max": 70, "step": 5},
    "hb_gen_macd_slope_lower": {"min": 0, "max": 70, "step": 5},
    "date_gap_upper": {"min": 0, "max": 30, "step": 5},
    "date_gap_lower": {"min": 0, "max": 30, "step": 5},
}
# ---------------------------------------

def get_levels(p):
    return list(range(p["min"], p["max"] + 1, p["step"]))

levels = [get_levels(PARAMS[name]) for name in PARAMS_ORDER]
cardinalities = [len(lv) for lv in levels]
total_combinations = prod(cardinalities)

print(f"Total combinations: {total_combinations:,}")

Path(OUTPUT_DIR).mkdir(exist_ok=True)

offsets = [0] * len(cardinalities)
remaining_total = total_combinations
file_idx = 0

def advance_offsets(offsets, cardinalities, num):
    carry = num
    for d in range(len(offsets) - 1, -1, -1):
        if carry == 0:
            break
        temp = offsets[d] + carry
        offsets[d] = temp % cardinalities[d]
        carry = temp // cardinalities[d]

while file_idx < N_FILES and remaining_total > 0:
    remaining_files = N_FILES - file_idx
    target = ceil(remaining_total / remaining_files)

    range_dict = copy.deepcopy(PARAMS)
    covered = 0
    partial_found = False
    block_size = 1

    for dd in range(len(cardinalities) - 1, -1, -1):  # fast → slow
        if partial_found:
            name = PARAMS_ORDER[dd]
            val = levels[dd][offsets[dd]]
            range_dict[name] = {"min": val, "max": val, "step": PARAMS[name]["step"]}
            continue

        name = PARAMS_ORDER[dd]
        step = PARAMS[name]["step"]
        curr_idx = offsets[dd]
        rem = cardinalities[dd] - curr_idx
        inner_size_this = block_size

        take = min(rem, target // inner_size_this) if inner_size_this > 0 else 0

        if take > 0:
            start_val = levels[dd][curr_idx]
            end_val = levels[dd][curr_idx + take - 1]
            range_dict[name] = {"min": start_val, "max": end_val, "step": step}
            covered = inner_size_this * take

            if take < rem:
                partial_found = True
            else:
                next_inner = inner_size_this * cardinalities[dd]
                if target // next_inner > 0:
                    block_size = next_inner
                else:
                    partial_found = True
        else:
            val = levels[dd][offsets[dd]]
            range_dict[name] = {"min": val, "max": val, "step": step}
            covered = inner_size_this
            partial_found = True

    filename = Path(OUTPUT_DIR) / f"doe_params_HBullD_Strategy_1_1_{file_idx + 1}.json"

    with open(filename, "w") as f:
        json.dump(range_dict, f, indent=2)

    print(f"part_{file_idx:03d}.json → {covered/1000000:,} million combinations")

    advance_offsets(offsets, cardinalities, covered)
    remaining_total -= covered
    file_idx += 1

print(f"\nDone – {file_idx} files generated, all {total_combinations/1000000000:,} billion combinations covered exactly once.")