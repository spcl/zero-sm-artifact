"""Read benchmark runs stored as <root>/<run>/."""
import fnmatch
import os
import re
import statistics as st


class Run:
    def __init__(self, root, name):
        self.name = name
        self.files = {}
        d = os.path.join(root, name)
        if os.path.isdir(d):
            for base, _, fs in os.walk(d):
                for f in fs:
                    p = os.path.join(base, f)
                    self.files[os.path.relpath(p, d)] = p
        self.source = name
        self.found = bool(self.files)

    def names(self, pattern):
        return sorted(n for n in self.files if fnmatch.fnmatch(n, pattern))

    def text(self, name):
        with open(self.files[name], errors="replace") as f:
            return f.read()


NUM = re.compile(r"^[0-9]+(\.[0-9]+)?$")


def nccl_tests_trials(run, pattern):
    """{size: [(time_us, busbw), ...]} from the out-of-place columns of nccl-tests logs."""
    data = {}
    for n in run.names(pattern):
        for line in run.text(n).splitlines():
            p = line.split()
            if len(p) < 8 or not p[0].isdigit() or not (NUM.match(p[5]) and NUM.match(p[7])):
                continue
            data.setdefault(int(p[0]), []).append((float(p[5]), float(p[7])))
    return data


def nccl_version(run, pattern):
    for n in run.names(pattern):
        m = re.search(r"NCCL version (\d+\.\d+\.\d+)", run.text(n))
        if m:
            return m.group(1)
    return ""


def sd(xs):
    return st.stdev(xs) if len(xs) > 1 else 0.0


def summarize(data):
    """Mean and sample std over trials, per size."""
    rows = []
    for size in sorted(data):
        t = [x[0] for x in data[size]]
        b = [x[1] for x in data[size]]
        rows.append(dict(size_bytes=size, trials=len(t), busbw_mean=st.mean(b), busbw_std=sd(b),
                         time_us_mean=st.mean(t), time_us_std=sd(t)))
    return rows


def from_times(per_size, ngpus):
    """Rows from {size: [time_us per trial]}; AllGather bus bandwidth = (n-1)/n * size / time."""
    rows = []
    f = (ngpus - 1) / ngpus
    for size in sorted(per_size):
        t = per_size[size]
        m, s = st.mean(t), sd(t)
        bw = f * size / m / 1e3
        rows.append(dict(size_bytes=size, trials=len(t), busbw_mean=bw, busbw_std=bw * s / m,
                         time_us_mean=m, time_us_std=s))
    return rows
