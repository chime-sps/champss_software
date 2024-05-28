from ps_inject import generate
import click

@click.argument(
        "-n",
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

    with open(file_name, 'w') as file:

        for i in range(n):

            freq = rand.choice(
                np.linspace(0.1, 100, 10000), num_injections, replace=False
            )
            dm = rand.choice(np.linspace(10, 200, 10000), num_injections, replace=False)
            
            sigma = rand.choice(np.linspace(1, 20, 1000))

            if path == 'random':
                prof = ps_inject.generate()

            else:
                prof = load_profs[i]

            file.write(f'- frequency: {freq}\n')
            file.write(f'  DM: {dm}\n')
            file.write(f'  sigma: {sigma}\n')
            file.write(f'  profile:\n')

            for bin in prof:
                file.write(f'    - {bin}\n')

