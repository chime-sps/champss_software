import sys
import numpy as np
import asyncio


async def run_fold_candidate(date, sigma, dm, f0, ra, dec, known, sem):
    async with sem:  # controls/allows running MAX_PROCESSES concurrent subprocesses at a time
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "./fold_candidate.py",
            f"{date}",
            f"{sigma}",
            f"{dm}",
            f"{f0}",
            f"{ra}",
            f"{dec}",
            f"{known}",
        )
        await proc.wait()


async def main(MAX_PROCESSES, date, sigmas, dms, f0s, ras, decs, knowns=False):
    sem = asyncio.Semaphore(MAX_PROCESSES)
    await asyncio.gather(
        *[
            run_fold_candidate(date, sigma, dm, f0, ra, dec, known, sem)
            for sigma, dm, f0, ra, dec, known in zip(
                sigmas, dms, f0s, ras, decs, knowns
            )
        ]
    )


if len(sys.argv) < 5:
    print("Usage = fold_batch.py date")

MAX_PROCESSES = 4

date = sys.argv[1]
cand_fn = str(sys.argv[2])

cand_file = np.load(cand_fn)
sigmas = cand_file["sigmas"].astype(float)
dms = cand_file["dms"].astype(float)
f0s = cand_file["f0s"].astype(float)
ras = cand_file["ras"].astype(float)
decs = cand_file["decs"].astype(float)
knowns = cand_file["known"]

print(
    "Running {0} concurrent processes on {1} pointings".format(MAX_PROCESSES, len(ras))
)

# asyncio.run(main(MAX_PROCESSES, year, month, day, sigmas, dms, f0s, ras, decs, knowns))
