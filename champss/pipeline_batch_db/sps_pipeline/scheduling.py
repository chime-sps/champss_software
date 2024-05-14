"""Methods for improving the scheduling. Currently work in progress and not implemented."""

import logging
import numpy as np

log = logging.getLogger(__package__)


def add_process(memory_array, start, length, requirement, limit, time_tiers):
    """Add process to the memory array."""
    current_iter = 0
    while (memory_array[start : start + length] + requirement > limit).any():
        start += 1
        current_iter += 1
        if current_iter > time_tiers.max():
            log.error(
                f"Can't fit job with requirement {requirement} memory requirement.."
            )
            return start - current_iter
    else:
        memory_array[start : start + length] += requirement
    return start


def move_process(
    memory_array,
    time_index,
    time_needed,
    memory_needed,
    df,
    random_choice=False,
    gap=3,
    move_sparse=True,
):
    """Add a random job to the current location."""
    all_viable_jobs = df.loc[
        (df["memory_allocation"] == memory_needed) & (df["time step"] > time_index)
    ]
    if len(all_viable_jobs) > 0:
        # Restrict to single date
        current_date = df[df["time step"] <= time_index].iloc[0]["date"]
        all_viable_jobs = all_viable_jobs[all_viable_jobs["date"] == current_date]
        if len(all_viable_jobs) == 0:
            return False
        if random_choice:
            inserted_job = all_viable_jobs.sample()
            job_index = inserted_job.index
            inserted_job_series = inserted_job.iloc[0]
        else:
            # First only jobs where (index % gap) != 0 will be moved.
            # This means that after the first pass a sparse grid of jobs will be computed first,
            # Which can be used for RFI information of subsequent jobs
            # This sparse grid will also be moved if move_sparse=True
            all_viable_jobs_sparse = all_viable_jobs.iloc[
                (all_viable_jobs.index.values % gap) != 0
            ]
            if len(all_viable_jobs_sparse) == 0:
                if move_sparse:
                    inserted_job_series = all_viable_jobs.iloc[-1]
                    job_index = all_viable_jobs.index.values[-1]
                else:
                    return False
            else:
                inserted_job_series = all_viable_jobs_sparse.iloc[-1]
                job_index = all_viable_jobs_sparse.index.values[-1]

        memory_array[time_index : time_index + time_needed] += memory_needed
        old_time_index = inserted_job_series["time step"]
        memory_array[old_time_index : old_time_index + time_needed] -= memory_needed
        df.loc[job_index, "time step"] = time_index
        return True
    else:
        return False


def remove_empty_space(df, memory_array):
    """Remove time steps if nothing is scheduled in those."""
    for i in range(len(memory_array)):
        while memory_array[i] == 0 and (memory_array[i:] != 0).any():
            memory_array[i:] = np.roll(memory_array[i:], -1)
            df.loc[df["time step"] >= i, "time step"] -= 1


def backfill_jobs(df_processes, max_memory):
    """Reverse backfill jobs."""
    memory_tiers = np.sort(df_processes["memory_allocation"].unique())[::-1]
    time_tiers = (memory_tiers / memory_tiers.min()).astype(int)
    # First step through all jobs, starting from the last
    memory_array = np.zeros(int(len(df_processes) * time_tiers.max()))
    time_index = 0
    df_processes["time step"] = -1
    for job_index in np.arange(len(df_processes))[::-1]:
        job_row = df_processes.loc[job_index]
        job_needs = job_row["nchan"]
        time_length = int(job_row["nchan"] / 1024)
        time_index = add_process(
            memory_array, time_index, time_length, job_needs, max_memory, time_tiers
        )
        df_processes.loc[job_index, "time step"] = time_index

    # Reduce memory array to reasonable length.
    first_0 = np.where(memory_array == 0)[0][0]
    memory_array = memory_array[:first_0]

    moved_jobs = 0
    for job_type_index, memory_needed in enumerate(memory_tiers):
        print(memory_needed)
        time_needed = time_tiers[job_type_index]
        for time_index in range(len(memory_array)):
            while (
                memory_array[time_index : time_index + time_needed] + memory_needed
                < max_memory
            ).all():
                moved_process = move_process(
                    memory_array, time_index, time_needed, memory_needed, df_processes
                )
                if moved_process:
                    moved_jobs += 1
                else:
                    break
    log.error(f"Moved {moved_jobs} jobs.")
    # Currently time steps start from the end, reverse this
    df_processes["time step"] = (
        df_processes["time step"].max() - df_processes["time step"]
    )
    # Remove empty spaces in the memory array, This does not change the actual ordering
    remove_empty_space(df_processes, memory_array)

    try:
        first_0_after = np.where(memory_array == 0)[0][0]
    except IndexError:
        first_0_after = first_0
    log.error(f"Length in time steps reduced from {first_0} to {first_0_after}.")
    df_processes.sort_values(by=["time step"], inplace=True, ascending=True)
    return df_processes, memory_array
