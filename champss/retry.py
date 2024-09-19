import click
import os

@click.command()
@click.argument('arg1', type=str, default="default_value", required=False)
@click.argument('arg2', type=int, required=False)
@click.option('--option1', '-o1', default="default_option1", help="A dummy option with default value.")
@click.option('--option2', '-o2', type=int, default=42, help="An integer option with default value.")
@click.option('--flag1/--no-flag1', default=True, help="A boolean flag.")
@click.option('--flag2', is_flag=True, help="Another flag option.")
def run_once(arg1, arg2, option1, option2, flag1, flag2):
    """A dummy function with a bunch of Click options, arguments, and flags"""
    flag_file = 'run_once_flag.txt'

    click.echo(f"arg1: {arg1}")
    click.echo(f"arg2: {arg2}")
    click.echo(f"option1: {option1}")
    click.echo(f"option2: {option2}")
    click.echo(f"flag1: {flag1}")
    click.echo(f"flag2: {flag2}")
    
    if os.path.exists(flag_file):
        # If the flag file exists, the function works (second run)
        click.echo("Function ran successfully!")
        return True
    else:
        # If the flag file doesn't exist, the function fails (first run)
        click.echo("Function failed the first time.")
        
        # Create the flag file to indicate the function has been run
        with open(flag_file, 'w') as f:
            f.write('This file marks that the function has been run before.')
        
        # Raise an exception to simulate a failure
        raise RuntimeError("This is the first time the function is run and it fails.")

