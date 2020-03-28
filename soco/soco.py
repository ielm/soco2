import os
import sys
sys.path.insert(0, f"{os.path.join(os.getcwd(),'..')}")
from soco.utils import *
import click


@click.group()
@click.pass_context
def soco(ctx):
    """Central application entry point for SoCo Homework III"""
    print("\nGender Classification - Homework III Social Computing"
          "\n\t\t\tAuthor: Ivan Leon"
          "\n\t\t\tProfessor: Tomek Strzalkowski\n")


@soco.command()
@click.option('--agree/--no_agree', default=True, prompt="\nThis will remove and replace everything in your data working directory. \nAre you okay with this?")
@click.pass_context
def build(ctx, agree: bool):
    """Clean the data directory and rebuild data."""
    if agree:
        print("\nCleaning data directory")
        clean_data_dir()
        print("\nRebuilding data directory")
        build_data_dirs()


@soco.command()
@click.pass_context
def train(ctx):
    """Train gender classification model"""
    # --mode/-m (SOFT/HARD)
    # if soft:
    #   if model is saved:
    #     use saved model
    #   else
    #     train new model
    # if hard:
    #   always train new model

    model = fit_model()


@soco.command()
@click.pass_context
def evaluate(ctx):
    """Evaluate model """
    x, y = get_eval_data()
    model = load_from_json()
    loss_func='categorical_crossentropy'
    optimizer='rmsprop'
    metrics=['acc']
    model.compile(loss=loss_func,
                  optimizer=optimizer,
                  metrics=metrics)
    loss, accuracy = model.evaluate(x, y, verbose=0)
    click.echo(f"Loss: {loss}")
    click.echo(f"Accuracy: {int(accuracy * 100)}")


def start():
    soco(obj={})


if __name__ == '__main__':
    start()
