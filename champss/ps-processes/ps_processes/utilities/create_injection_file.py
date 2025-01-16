import os

import click
import numpy as np
import numpy.random as rand
import pandas as pd
import yaml
from ps_processes.processes import ps_inject


@click.command()
@click.option("--n-injections", "-n", default=1, type=int, help="Number of injections")
@click.option(
    "--file-name",
    "--fn",
    default="test_injections.pickle",
    type=str,
    help="Name of target file",
)
@click.option(
    "--file-type",
    "--ft",
    type=click.Choice(["pickle", "yaml"], case_sensitive=False),
    default="pickle",
    help=(
        "Output type. Possible: pickle, yaml. Pickle is a pickled pd.DataFrame. yaml"
        " has bad performance due to the profile array"
    ),
)
@click.option(
    "--injection-path",
    "--path",
    default="random",
    help="Path to injection profile npy file",
)
@click.option(
    "--focus",
    default=None,
    help="Iterates over selected field (sigma, frequency, or DM).",
)
def main(n_injections, file_name, file_type, injection_path, focus):
    if injection_path != "random":
        load_profs = np.load(injection_path)
        # n_injections = len(load_profs)

    if focus == "frequency" or focus == "freq":
        # frequencies = np.logspace(1.8, 2.3, n_injections)
        frequencies = np.logspace(-2, 2, n_injections)
        dms = 107.3817479147 * np.ones(n_injections)
        sigmas = 11.28372911 * np.ones(n_injections)

    elif focus == "dm" or focus == "DM":
        dms = np.linspace(3, 200, n_injections)
        frequencies = 8.138748235982394 * np.ones(n_injections)
        sigmas = 11.28372911 * np.ones(n_injections)

    elif focus == "sigma" or focus == "sig":
        sigmas = np.linspace(6, 17, n_injections)
        dms = 107.3817479147 * np.ones(n_injections)
        frequencies = 8.138748235982394 * np.ones(n_injections)

    elif focus == "duty":
        frequencies = 8.138748235982394 * np.ones(n_injections)
        dms = 107.3817479147 * np.ones(n_injections)
        sigmas = 11.28372911 * np.ones(n_injections)

    else:
        sigmas = np.random.uniform(6, 20, n_injections)
        frequencies = np.random.uniform(0.1, 10, n_injections)
        dms = np.random.uniform(3, 500, n_injections)

    data = []
    print(f"Creating {n_injections} fake pulsars into {injection_path}")

    for i in range(n_injections):
        n_dict = {}

        # .item() allows a simpler output in the yanl file
        # alternatively could use float()
        n_dict["frequency"] = frequencies[i].item()
        n_dict["DM"] = dms[i].item()
        n_dict["sigma"] = sigmas[i].item()

        print(f"{i}: {n_dict}")
        if injection_path == "random":
            n_dict["profile"] = ps_inject.generate_pulse().tolist()

        else:
            chosen_profile = np.random.choice(load_profs.shape[0])
            n_dict["profile"] = load_profs[chosen_profile].tolist()

        data.append(n_dict)

    file_name = os.getcwd() + "/" + file_name
    if file_type == "pickle":
        df = pd.DataFrame(data)
        # move profile to last pos, for easier reading of csv
        cols = df.columns
        cols = cols[cols != "profile"].append(pd.Index(["profile"]))
        df = df[cols]
        df.to_pickle(file_name)
    else:
        stream = open(file_name, "w")
        yaml.dump(data, stream)


if __name__ == "__main__":
    main()
