from ps_processes.processes import ps_inject
import click
import yaml
import os
import numpy.random as rand
import numpy as np

@click.command()
@click.option(
        "--n-injections",
        "-n",
        default = 1,
        type = int,
        help = ("Number of injections")
)

@click.option(
        "--file-name",
        "--fn",
        default = "test_injections.yaml",
        type = str,
        help = ("Name of target file")
)

@click.option(
        "--injection-path",
        "--path",
        default = "random",
        help = ("Path to injection profile npy file")
)

def get(n_injections, file_name, injection_path):
    
    if injection_path != 'random':
        load_profs = np.load(injection_path)
        n = len(load_profs)
    
    frequencies = np.random.uniform(0.1, 100, n_injections)
    dms = np.random.uniform(3, 200, n_injections)
    sigmas = np.random.uniform(10, 20, n_injections)
    data = []
    print(f"Creating {n_injections} fake pulsars into {injection_path}")
    
    for i in range(n_injections):
        
        n_dict = {}

        # .item() allows a simpler output in the yanl file
        # alternatively could use float()
        n_dict['frequency'] = frequencies[i].item()
        n_dict['DM'] = dms[i].item()
        n_dict['sigma'] = sigmas[i].item()

        print(f"{i}: {n_dict}")
        if injection_path == 'random':
            n_dict['profile'] = ps_inject.generate().tolist()

        else:
            n_dict['profile'] = load_profs[i]
        
        data.append(n_dict)

    file_name = os.getcwd()+'/'+file_name
    stream = open(file_name, 'w')
    yaml.dump(data, stream)

if __name__ == "__main__":
    get()
