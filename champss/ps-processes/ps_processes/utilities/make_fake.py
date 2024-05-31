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

def get(n, file_name, path):
    
    if path != 'random':
        load_profs = np.load(path)
        n = len(load_profs)
    
    data = []
    print('hello world')
    for i in range(n):
        
        n_dict = {}

<<<<<<< HEAD:champss/ps-processes/ps_processes/utilities/make_fake.py
        n_dict['frequency'] = rand.choice(np.linspace(0.1, 50, 1000))
        n_dict['DM'] = rand.choice(np.linspace(10, 200, 10000))
=======
        n_dict['frequency'] = rand.choice(
            np.linspace(0.1, 100, 10000), num_injections, replace=False
        )
        n_dict['DM'] = rand.choice(np.linspace(10, 200, 10000), num_injections, replace=False)
        
>>>>>>> parent of e35d8d5... Fixed click issue:champss/ps-processes/ps_processes/processes/make_fake.py
        n_dict['sigma'] = rand.choice(np.linspace(1, 20, 1000))

        if path == 'random':
            n_dict['profile'] = ps_inject.generate()

        else:
            n_dict['profile'] = load_profs[i]
        
        data.append(n_dict)

    file_name = os.getcwd()+'/'+file_name
    stream = file(file_name, 'w')
    yaml.dump(data, stream)

if __name__ == "__main__":
    get()
