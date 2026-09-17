"""Official HumanML3D ids <-> ids of the local copy at /iridisfs/scratch/pf2m24/data/HumanML3D/HumanML3D.

The local copy was renumbered (see zhuanhhh.py there): official ids 007975, 009707, 011059 are missing and
every later id is shifted down by the number of missing ids below it; mirrored clips 'M<id>' become
local id + 14613 (no 'M' prefix). index.csv, our physics clips and HF files use OFFICIAL ids; texts/,
new_joint_vecs/ and the split files use LOCAL ids. Never use the local new_joints/ directory.
"""
MISSING_OFFICIAL = (7975, 9707, 11059)
N_NON_MIRROR_LOCAL = 14613  # 14616 - 3


def official_to_local(name):
    """'007976' -> '007975'; 'M000001' -> '014614'; returns None for the 3 missing official ids."""
    mirror = name.startswith("M")
    num = int(name[1:] if mirror else name)
    if num in MISSING_OFFICIAL:
        return None
    off = sum(1 for m in MISSING_OFFICIAL if num > m)
    local = num - off + (N_NON_MIRROR_LOCAL if mirror else 0)
    return f"{local:06d}"


def local_to_official(name):
    """inverse of official_to_local ('014613' -> 'M000000')."""
    num = int(name)
    mirror = num >= N_NON_MIRROR_LOCAL
    if mirror:
        num -= N_NON_MIRROR_LOCAL
    for m in MISSING_OFFICIAL:  # ascending: re-insert the gaps
        if num >= m:
            num += 1
    return ("M" if mirror else "") + f"{num:06d}"


def is_local_mirror(name):
    return int(name) >= N_NON_MIRROR_LOCAL


if __name__ == "__main__":
    for o in ["000000", "007974", "007976", "009708", "011060", "014615", "M000000", "M007976", "M014615"]:
        l = official_to_local(o); assert local_to_official(l) == o, (o, l)
        print(o, "->", l)
    assert official_to_local("007975") is None
    assert official_to_local("M014615") == "029225"
