PRESENCE_COLS = ["face", "light source", "building", "sky", "greenery", "main color"]

GLOBAL_QUESTION_COLS = [
    "too bright",
    "too dark",
    "contrast too high",
    "contrast too low",
    "highlight clipped",
    "highlight over-suppressed",
    "WB yellow cast",
    "WB blue cast",
    "WB red cast",
    "WB green cast",
    "low clarity",
    "sharpness too high",
    "noise obvious",
]

LOCAL_QUESTION_COLS_BY_REGION = {
    "face": [
        "Face too bright",
        "Face too dark",
        "Face contrast too high",
        "Face contrast too low",
        "Face highlight clipped",
        "Face highlight over-suppressed",
        "Skin yellow cast",
        "Skin blue cast",
        "Skin red cast",
        "Skin green cast",
        "Face saturation too high",
        "Face saturation too low",
        "Lip color issue",
        "Face low clarity",
        "Face fake texture",
        "Hair detail loss",
        "Sharpening artifacts on face",
        "Noise obvious on face",
    ],
    "building": [
        "Building too bright",
        "Building too dark",
        "Building contrast too high",
        "Building contrast too low",
        "Building low clarity",
    ],
    "sky": [
        "Sky color cast",
        "Sky saturation too high",
        "Sky saturation too low",
        "Sky contrast too high",
        "Sky contrast too low",
        "Sky highlight clipped",
        "Sky highlight over-suppressed",
        "Sky compositing artifact",
    ],
    "greenery": [
        "Greenery color cast",
        "Greenery saturation too high",
        "Greenery saturation too low",
        "Greenery contrast too high",
        "Greenery contrast too low",
        "Greenery low clarity",
    ],
    "main color": [
        "Main color cast",
        "Main color saturation too high",
        "Main color saturation too low",
    ],
}

QUESTION_COLS = GLOBAL_QUESTION_COLS + [
    question
    for region in ["face", "building", "sky", "greenery", "main color"]
    for question in LOCAL_QUESTION_COLS_BY_REGION[region]
]

MUTEX_GROUPS_BY_NAME = [
    ["too bright", "too dark"],
    ["contrast too high", "contrast too low"],
    ["highlight clipped", "highlight over-suppressed"],
    ["WB yellow cast", "WB blue cast", "WB red cast", "WB green cast"],
    ["Face too bright", "Face too dark"],
    ["Face contrast too high", "Face contrast too low"],
    ["Face highlight clipped", "Face highlight over-suppressed"],
    ["Skin yellow cast", "Skin blue cast", "Skin red cast", "Skin green cast"],
    ["Face saturation too high", "Face saturation too low"],
    ["Building too bright", "Building too dark"],
    ["Building contrast too high", "Building contrast too low"],
    ["Sky saturation too high", "Sky saturation too low"],
    ["Sky contrast too high", "Sky contrast too low"],
    ["Sky highlight clipped", "Sky highlight over-suppressed"],
    ["Greenery saturation too high", "Greenery saturation too low"],
    ["Greenery contrast too high", "Greenery contrast too low"],
    ["Main color saturation too high", "Main color saturation too low"],
]

# (src_label, src_threshold, dst_label, dst_threshold)
# threshold semantics in zero-based ordinal levels:
#   @1 -> level >= 1 (minor+)
#   @2 -> level >= 2 (severe+)
#   @3 -> level >= 3 (critical)
SUBSUMPTION_EDGES_BY_NAME = [
    ("Hair detail loss", 1, "Face low clarity", 1),
    ("Sharpening artifacts on face", 1, "Face fake texture", 1),
    ("highlight clipped", 2, "too bright", 1),
    ("Face too bright", 2, "too bright", 1),
    ("Face too dark", 2, "too dark", 1),
    ("Face highlight clipped", 2, "highlight clipped", 1),
    ("Face highlight over-suppressed", 2, "highlight over-suppressed", 1),
    ("Face low clarity", 2, "low clarity", 1),
    ("Building too bright", 2, "too bright", 1),
    ("Building too dark", 2, "too dark", 1),
    ("Building low clarity", 2, "low clarity", 1),
    ("Greenery low clarity", 2, "low clarity", 1),
]

# Fixed evaluation whitelist based on the full reconstructed SPAQ label set.
# Criterion: positive count >= 100 over all 11125 images.
EVAL_LABELS_MIN100_GLOBAL = [
    "too bright",
    "too dark",
    "contrast too high",
    "contrast too low",
    "highlight clipped",
    "WB yellow cast",
    "WB blue cast",
    "low clarity",
    "noise obvious",
    "Face too bright",
    "Face too dark",
    "Face contrast too high",
    "Face contrast too low",
    "Face saturation too low",
    "Face low clarity",
    "Face fake texture",
    "Hair detail loss",
    "Noise obvious on face",
    "Building too bright",
    "Building too dark",
    "Building contrast too low",
    "Building low clarity",
    "Sky saturation too low",
    "Sky contrast too low",
    "Sky highlight clipped",
    "Sky highlight over-suppressed",
    "Greenery saturation too high",
    "Greenery saturation too low",
    "Greenery contrast too high",
    "Greenery contrast too low",
    "Greenery low clarity",
    "Main color cast",
    "Main color saturation too high",
    "Main color saturation too low",
]


def _build_groups_in_question_order():
    label_to_group = {}
    for group in MUTEX_GROUPS_BY_NAME:
        for label in group:
            if label in label_to_group:
                raise ValueError(f"label appears in multiple mutex groups: {label}")
            label_to_group[label] = group

    groups = []
    used = set()
    idx = 0
    while idx < len(QUESTION_COLS):
        label = QUESTION_COLS[idx]
        group = label_to_group.get(label)
        if group is None:
            groups.append([label])
            idx += 1
            continue

        group_tuple = tuple(group)
        if group_tuple in used:
            raise ValueError(f"mutex group duplicated in QUESTION_COLS order: {group}")

        width = len(group)
        actual = QUESTION_COLS[idx:idx + width]
        if actual != group:
            raise ValueError(
                "mutex group must be contiguous and ordered exactly as listed. "
                f"expected {group}, got {actual}"
            )

        groups.append(group)
        used.add(group_tuple)
        idx += width

    if len(used) != len(MUTEX_GROUPS_BY_NAME):
        raise ValueError("some mutex groups were not found in QUESTION_COLS")

    return groups


QUESTION_GROUPS = _build_groups_in_question_order()
GROUP_SIZES = [len(group) for group in QUESTION_GROUPS]
ATTR_GROUP_ID = [
    group_idx
    for group_idx, group in enumerate(QUESTION_GROUPS)
    for _ in group
]

NUM_QUESTION_LABELS = len(QUESTION_COLS)
