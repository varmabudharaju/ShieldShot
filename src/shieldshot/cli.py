"""ShieldShot CLI."""

import click


@click.group()
@click.version_option()
def main():
    """ShieldShot — Protect your photos from deepfake misuse."""
    pass


@main.command()
def init():
    """Set up ShieldShot (download models, generate keys)."""
    click.echo("ShieldShot initialized.")


@main.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("-o", "--output", "output_path", type=click.Path(), default=None,
              help="Output path for protected image.")
@click.option("--mode", type=click.Choice(["fast", "thorough"]), default="fast",
              help="Protection mode: fast (generator) or thorough (PGD).")
def protect(input_path, output_path, mode):
    """Protect a photo from deepfake misuse."""
    click.echo(f"Protecting {input_path} (mode={mode})...")


@main.command()
@click.argument("input_path", type=click.Path(exists=True))
def verify(input_path):
    """Verify provenance of a protected photo."""
    click.echo(f"Verifying {input_path}...")


@main.command()
@click.argument("input_path", type=click.Path(exists=True))
def extract(input_path):
    """Extract watermark from a protected photo."""
    click.echo(f"Extracting watermark from {input_path}...")


if __name__ == "__main__":
    main()
