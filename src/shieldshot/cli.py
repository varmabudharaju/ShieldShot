"""ShieldShot CLI."""

from pathlib import Path
import click
from shieldshot import __version__


@click.group()
@click.version_option(version=__version__)
def main():
    """ShieldShot — Protect your photos from deepfake misuse."""
    pass


@main.command()
def init():
    """Set up ShieldShot (download models, generate keys)."""
    from shieldshot.provenance.c2pa import init_keys
    keys_dir = init_keys()
    click.echo(f"Keys generated at {keys_dir}")
    click.echo("ShieldShot initialized.")


@main.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("-o", "--output", "output_path", type=click.Path(), default=None)
@click.option("--mode", type=click.Choice(["fast", "thorough"]), default="fast")
@click.option("--sign", is_flag=True, help="Apply C2PA provenance signing.")
@click.option("--user-id", default="default")
def protect(input_path, output_path, mode, sign, user_id):
    """Protect a photo from deepfake misuse."""
    from shieldshot.protect import protect_image

    if output_path is None:
        p = Path(input_path)
        output_path = str(p.parent / f"{p.stem}_protected{p.suffix}")

    click.echo(f"Protecting {input_path} (mode={mode})...")

    result = protect_image(
        input_path, output_path, mode=mode, user_id=user_id,
        skip_no_face=True, sign_c2pa=sign,
    )

    if result["success"]:
        click.echo(f"Protected image saved to {output_path}")
        click.echo(f"  Faces found: {result['faces_found']}")
        click.echo(f"  SSIM: {result['ssim']:.4f}")
        click.echo(f"  Time: {result['time_seconds']}s")
    else:
        click.echo(f"Protection failed: {result.get('reason', 'unknown')}", err=True)
        raise SystemExit(1)


@main.command()
@click.argument("input_path", type=click.Path(exists=True))
def verify(input_path):
    """Verify provenance of a protected photo."""
    from shieldshot.provenance.c2pa import verify_image
    result = verify_image(input_path)
    if result["valid"]:
        click.echo("Valid C2PA manifest found")
        click.echo(f"  Software: {result.get('software', 'unknown')}")
    else:
        click.echo(f"No valid C2PA manifest: {result.get('reason', 'not found')}")


@main.command()
@click.argument("input_path", type=click.Path(exists=True))
def extract(input_path):
    """Extract watermark from a protected photo."""
    import torch
    from shieldshot.utils.image import load_image, to_tensor
    from shieldshot.watermark.decoder import WatermarkDecoder
    from shieldshot.watermark.payload import decode_payload, PAYLOAD_BITS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    decoder = WatermarkDecoder(payload_bits=PAYLOAD_BITS).to(device)
    weights_path = Path.home() / ".shieldshot" / "models" / "watermark_decoder.pt"
    if weights_path.exists():
        decoder.load_state_dict(torch.load(weights_path, weights_only=True))
    decoder.eval()

    img = load_image(input_path)
    tensor = to_tensor(img).to(device)
    with torch.no_grad():
        logits = decoder(tensor)
    bits = (logits > 0).int().squeeze(0).tolist()

    result = decode_payload(bits)
    if result["valid"]:
        click.echo("Watermark extracted successfully")
        click.echo(f"  User ID hash: {result['user_id_hash']}")
        click.echo(f"  Timestamp: {result['timestamp']}")
    else:
        click.echo("No valid watermark found (model may need training)")


if __name__ == "__main__":
    main()
