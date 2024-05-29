from ps_inject import generate
import click
import yaml
import os


@click.argument(
        "n",
        type = int,
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

def main(n, file_name, path):
    
    if path != 'random':
        load_profs = np.load(path)
        n = len(load_profs)
    
    data = []

    for i in range(n):
        
        n_dict = {}

        n_dict['frequency'] = rand.choice(
            np.linspace(0.1, 100, 10000), num_injections, replace=False
        )
        n_dict['DM'] = rand.choice(np.linspace(10, 200, 10000), num_injections, replace=False)
        
        n_dict['sigma'] = rand.choice(np.linspace(1, 20, 1000))

        if path == 'random':
            n_dict['profile'] = ps_inject.generate()

        else:
            n_dict['profile'] = load_profs[i]
        
        data.append(n_dict)

    file_name = os.getcwd()+'/'+file_name
    stream = file(file_name, 'w')
    yaml.dump(data, stream)
