"""C2PA provenance signing and verification."""

import datetime
import json
from pathlib import Path

KEYS_DIR = Path.home() / ".shieldshot" / "keys"


def init_keys(keys_dir: str | None = None) -> str:
    """Generate a CA and end-entity certificate chain for C2PA signing.

    Creates a root CA certificate and an end-entity certificate signed by
    that CA. The cert.pem file contains the full chain (end-entity + CA).

    Args:
        keys_dir: Directory to store keys. Defaults to ~/.shieldshot/keys.

    Returns:
        Path to the keys directory.
    """
    from cryptography import x509
    from cryptography.x509.oid import NameOID, ExtendedKeyUsageOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    kd = Path(keys_dir) if keys_dir else KEYS_DIR
    kd.mkdir(parents=True, exist_ok=True)
    cert_path = kd / "cert.pem"
    key_path = kd / "key.pem"
    if cert_path.exists() and key_path.exists():
        return str(kd)

    now = datetime.datetime.now(datetime.timezone.utc)
    expire = now + datetime.timedelta(days=3650)

    # 1. Generate CA key and self-signed CA certificate
    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_name = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "ShieldShot"),
        x509.NameAttribute(NameOID.COMMON_NAME, "ShieldShot Root CA"),
    ])
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(expire)
        .add_extension(
            x509.BasicConstraints(ca=True, path_length=None), critical=True
        )
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                key_cert_sign=True,
                crl_sign=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )

    # 2. Generate end-entity key and certificate signed by CA
    ee_key = ec.generate_private_key(ec.SECP256R1())
    ee_name = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "ShieldShot"),
        x509.NameAttribute(NameOID.COMMON_NAME, "ShieldShot Signer"),
    ])
    ee_cert = (
        x509.CertificateBuilder()
        .subject_name(ee_name)
        .issuer_name(ca_name)
        .public_key(ee_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(expire)
        .add_extension(
            x509.BasicConstraints(ca=False, path_length=None), critical=True
        )
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=True,
                key_cert_sign=False,
                crl_sign=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.EMAIL_PROTECTION]),
            critical=True,
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(ee_key.public_key()),
            critical=False,
        )
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(
                ca_key.public_key()
            ),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )

    # 3. Write cert chain (end-entity + CA) and private key
    chain_pem = (
        ee_cert.public_bytes(serialization.Encoding.PEM)
        + ca_cert.public_bytes(serialization.Encoding.PEM)
    )
    key_pem = ee_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    cert_path.write_bytes(chain_pem)
    key_path.write_bytes(key_pem)

    return str(kd)


def sign_image(
    input_path: str, output_path: str, keys_dir: str | None = None
) -> None:
    """Sign an image with a C2PA manifest.

    Args:
        input_path: Path to the source image.
        output_path: Path to write the signed image.
        keys_dir: Directory containing cert.pem and key.pem.

    Raises:
        FileNotFoundError: If no certificate is found.
    """
    import c2pa

    kd = Path(keys_dir) if keys_dir else KEYS_DIR
    cert_path = kd / "cert.pem"
    key_path = kd / "key.pem"
    if not cert_path.exists():
        raise FileNotFoundError(
            "No certificate found. Run 'shieldshot init' first."
        )

    manifest_json = json.dumps({
        "claim_generator": "shieldshot/0.1.0",
        "title": Path(input_path).name,
        "assertions": [
            {
                "label": "c2pa.actions",
                "data": {
                    "actions": [
                        {
                            "action": "c2pa.edited",
                            "softwareAgent": "shieldshot/0.1.0",
                            "description": "Applied deepfake protection",
                        }
                    ]
                },
            }
        ],
    })

    builder = c2pa.Builder(manifest_json)

    cert_data = cert_path.read_bytes()
    key_data = key_path.read_bytes()

    # Build C2paSignerInfo manually to allow NULL ta_url (no TSA).
    # The __init__ method rejects None for ta_url, but the ctypes
    # struct accepts NULL (None) when fields are set directly.
    signer_info = c2pa.C2paSignerInfo.__new__(c2pa.C2paSignerInfo)
    signer_info.alg = b"es256"
    signer_info.sign_cert = cert_data
    signer_info.private_key = key_data
    # ta_url defaults to None (NULL pointer) = no timestamp authority
    signer = c2pa.Signer.from_info(signer_info)

    builder.sign_file(input_path, output_path, signer)


def verify_image(image_path: str) -> dict:
    """Verify a C2PA manifest in an image.

    Args:
        image_path: Path to the image to verify.

    Returns:
        Dict with 'valid' bool, and if valid, 'software' and 'title' fields.
    """
    import c2pa

    try:
        reader = c2pa.Reader(image_path)
        manifest_store = json.loads(reader.json())
        active = manifest_store.get("active_manifest")
        if not active:
            return {"valid": False, "reason": "No active manifest"}
        manifest = manifest_store["manifests"].get(active, {})
        # Extract software info: prefer softwareAgent from assertions,
        # fall back to claim_generator or claim_generator_info.
        software = ""
        for assertion in manifest.get("assertions", []):
            for action in assertion.get("data", {}).get("actions", []):
                agent = action.get("softwareAgent", "")
                if agent:
                    software = agent
                    break
            if software:
                break
        if not software:
            software = manifest.get("claim_generator", "")
        if not software:
            gen_info = manifest.get("claim_generator_info", [])
            if gen_info:
                software = gen_info[0].get("name", "")
        return {
            "valid": True,
            "software": software,
            "title": manifest.get("title", ""),
        }
    except Exception:
        return {"valid": False, "reason": "No C2PA manifest found"}
