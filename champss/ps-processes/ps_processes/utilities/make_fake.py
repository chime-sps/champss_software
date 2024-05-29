import ps_inject
import click
import yaml
import os
import numpy.random as rand
import numpy as np

@click.command()
@click.option(
        "--n-injections",
        "--n",
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
    
    data = []
    
    for i in range(n_injections):
        
        n_dict = {}

        n_dict['frequency'] = rand.choice(np.linspace(0.1, 50, 1000))
        n_dict['DM'] = rand.choice(np.linspace(10, 200, 10000))
        n_dict['sigma'] = rand.choice(np.linspace(1, 20, 1000))

        if injection_path == 'random':
            n_dict['profile'] = ps_inject.generate()

        else:
            n_dict['profile'] = load_profs[i]
        
        data.append(n_dict)

    file_name = os.getcwd()+'/'+file_name
    stream = open(file_name, 'w')
    yaml.dump(data, stream)

if __name__ == "__main__":
    get()
