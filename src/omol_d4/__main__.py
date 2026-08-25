from typer import Typer

APP = Typer()


@APP.command()
def download():
    # download the OMol-D4-4M dataset (location TBD)
    pass


@APP.command()
def plot():
    pass


def main():
    APP()


if __name__ == "__main__":
    main()
