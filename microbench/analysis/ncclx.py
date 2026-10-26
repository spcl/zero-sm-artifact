"""NCCLX persistent AllGatherP rows from ncclx_<platform>_<N>n<G>g runs (trials.csv: rank-max time per trial)."""
import csv
import io

from rawio import Run, from_times

PLACEMENTS = [("GB200", "1n4g", "1 node", 4), ("GB200", "4n16g", "4 nodes", 16), ("GB200", "18n72g", "18 nodes", 72),
              ("B200", "1n8g", "1 node", 8), ("B200", "4n32g", "4 nodes", 32), ("B200", "8n64g", "8 nodes", 64)]


def rows(root):
    out = []
    for plat, tag, placement, n in PLACEMENTS:
        run = Run(root, f"ncclx_{plat.lower()}_{tag}")
        if not run.found:
            continue
        per, algo = {}, ""
        for r in csv.DictReader(io.StringIO(run.text("trials.csv"))):
            per.setdefault(int(r["size_bytes"]), []).append(float(r["time_us"]))
            algo = r["backend"].replace("ncclx_allgatherp_", "")
        for r in from_times(per, n):
            out.append(dict(platform=plat, placement=placement, ngpus=n, arm="NCCLX", build="", **r,
                            stat_note=f"mean; NCCLX AllGatherP {algo}", source=run.source))
    return out
