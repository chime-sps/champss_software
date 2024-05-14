import numpy as np
import matplotlib.pyplot as plt
import glob

files = glob.glob("*fake_pointing*.npz")
files.sort()
num_rows = 12
num_cols = 90
plt.figure(figsize=(num_rows * 4, num_cols * 2))
for i, file in enumerate(files):
    f = np.load(file, allow_pickle=True)
    group = f["group"].item()
    s = f["group_summary"].item()
    psrs = []
    for k in s:
        if not s[k]["rfi"]:
            psrs.append(k)
            if s[k]["harmonics"]:
                for h in s[k]["harmonics"]:
                    psrs.append(h)
    idx = (
        (num_rows * num_cols)
        - (i * num_cols)
        + (num_rows * num_cols * (i // num_rows) - (i // num_rows))
    )
    plt.subplot(num_rows, num_cols, idx)
    freqs = []
    dms = []
    hhats = []
    for g in group:
        freqs += list(group[g]["f"])
        dms += list(group[g]["dm"])
        hhats += [0.5]  # list(2 + group[g]['hhat']//2)
    plt.scatter(freqs, dms, s=hhats, alpha=0.5, label="RFI")
    astro = {k: group[k] for k in psrs}
    freqs = []
    dms = []
    hhats = []
    for g in astro:
        freqs += list(astro[g]["f"])
        dms += list(astro[g]["dm"])
        hhats += list(5 + astro[g]["hhat"])
    plt.scatter(freqs, dms, s=hhats, c="red", label="PULSAR")
    plt.xscale("Log")
    plt.xlim(1e-2, 500)
    plt.ylim(0, 400)
    plt.text(0.02, 300, "{}, {}".format(file.split("_")[0], file.split("_")[1]))
    if idx in [num_cols * v + 1 for v in range(num_rows)]:
        plt.ylabel("DM")
    if idx in [(num_rows - 1) * num_cols + 1 + v for v in range(num_cols)]:
        plt.xlabel("frequency (Hz)")
    # plt.legend()
plt.savefig("pointing_grid.png")
