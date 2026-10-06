#!/usr/bin/env python3
"""Verify checksums and Oxide-compatible public keys at every registry location."""

import argparse
import base64
import binascii
import hashlib
import json
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit
from urllib.request import Request, urlopen


MAX_KEY_BYTES = 1024 * 1024
PARTY_SLUGS = frozenset(("fau", "berkeley-university", "kaist-university"))
USER_AGENT = "Mozilla/5.0 (compatible; iris-mpc-key-check/1.0; +https://github.com/worldcoin/iris-mpc)"


def parse_registry(registry):
    if not isinstance(registry, dict) or not isinstance(registry.get("iris"), dict):
        raise ValueError("Expected an object containing iris.parties")
    parties = registry["iris"].get("parties")
    if not isinstance(parties, list):
        raise ValueError("iris.parties must be an array")
    if len(parties) != len(PARTY_SLUGS):
        raise ValueError("iris.parties must contain exactly three parties")

    locations = []
    slugs = set()
    for index, party in enumerate(parties):
        label = f"iris.parties[{index}]"
        if not isinstance(party, dict):
            raise ValueError(f"{label} must be an object")
        slug = party.get("slu")
        if not isinstance(slug, str) or slug not in PARTY_SLUGS or slug in slugs:
            raise ValueError(f"{label}.slu must be one of {', '.join(sorted(PARTY_SLUGS))}, each exactly once")
        slugs.add(slug)
        checksum = party.get("chk")
        if not isinstance(checksum, str):
            raise ValueError(f"{label}.chk must be a string")
        algorithm, separator, digest = checksum.partition(":")
        lengths = {"sha256": 64, "sha512": 128}
        if algorithm not in lengths or not separator:
            raise ValueError(f"{label}.chk must use sha256 or sha512")
        if not re.fullmatch(rf"[0-9a-fA-F]{{{lengths[algorithm]}}}", digest):
            raise ValueError(f"{label}.chk must contain a {lengths[algorithm]}-digit hex digest")
        urls = party.get("pub")
        if isinstance(urls, str):
            urls = [urls]
        if not isinstance(urls, list) or not urls:
            raise ValueError(f"{label}.pub must be a URL or a nonempty URL array")
        for url in urls:
            if not isinstance(url, str):
                raise ValueError(f"{label}.pub URLs must be strings")
            parsed = urlsplit(url)
            if parsed.scheme not in ("http", "https") or not parsed.hostname:
                raise ValueError(f"{label}.pub must contain HTTP(S) URLs")
            if parsed.username is not None or parsed.password is not None:
                raise ValueError(f"{label}.pub URLs must not contain credentials")
            # Accessing port also rejects malformed or out-of-range ports.
            _ = parsed.port
            if "#" in url:
                raise ValueError(f"{label}.pub URLs must not contain fragments")
            locations.append((slug, url, algorithm, digest.lower()))
    return locations


def validate_public_key(data):
    """Match Oxide's base64::STANDARD decoder and 32-byte PublicKey::from_slice."""
    try:
        decoded = base64.b64decode(data, validate=True)
    except binascii.Error as error:
        raise ValueError(f"Key is not valid base64: {error}") from error
    # Python's strict decoder still accepts nonzero unused padding bits. Rust's
    # STANDARD decoder requires canonical padding and zero unused bits. Do not
    # strip whitespace: Oxide decodes the original key without normalization.
    if base64.b64encode(decoded) != data:
        raise ValueError("Key must use canonical standard base64")
    if len(decoded) != 32:
        raise ValueError(f"Key must decode to exactly 32 bytes, got {len(decoded)}")


def checkout_path(url, repository, repo_root):
    """Map this repository's main URLs to the proposed checkout contents."""
    parsed = urlsplit(url)
    if not repository or parsed.hostname != "raw.githubusercontent.com":
        return None
    default_port = 443 if parsed.scheme == "https" else 80
    if parsed.username is not None or parsed.password is not None or parsed.port not in (None, default_port):
        raise ValueError("Raw GitHub URLs must not contain credentials or non-default ports")
    for ref in ("refs/heads/main", "main"):
        prefix = f"/{repository}/{ref}/"
        if parsed.path.startswith(prefix) and not parsed.query:
            root = Path(repo_root).resolve()
            relative = unquote(parsed.path[len(prefix):])
            relative_path = Path(relative)
            if not relative or relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError("Repository key path must stay inside the checkout")
            path = root
            for component in relative_path.parts:
                path = path / component
                if path.is_symlink():
                    raise ValueError("Repository key path must not contain symlinks")
            path = path.resolve()
            if not relative or not path.is_relative_to(root):
                raise ValueError("Repository key path must stay inside the checkout")
            return path
    return None


def verify_registry(registry, repository=None, repo_root="."):
    locations = parse_registry(registry)
    failures = 0
    for slug, url, algorithm, expected in locations:
        try:
            local_path = checkout_path(url, repository, repo_root)
            source = f"checkout:{local_path}" if local_path is not None else url
            if local_path is not None:
                response = local_path.open("rb")
            else:
                request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
                response = urlopen(request, timeout=30)
            with response:
                data = response.read(MAX_KEY_BYTES + 1)
            if len(data) > MAX_KEY_BYTES:
                raise ValueError("Key exceeds the 1 MiB download limit")
            actual = hashlib.new(algorithm, data).hexdigest()
            if actual != expected:
                raise ValueError(f"Checksum mismatch: expected {algorithm}:{expected}, got {algorithm}:{actual}")
            validate_public_key(data)
            print(f"OK: {slug} {url} (source: {source})")
        except (OSError, ValueError) as error:
            failures += 1
            print(f"FAIL: {slug} {url}: {error}", file=sys.stderr)
    print(f"Checked {len(locations)} key locations; {failures} failed.")
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registry", nargs="?", type=Path, default=Path("keys.json"))
    parser.add_argument("--repository", help="owner/repo whose main raw URLs should use local files")
    parser.add_argument("--repo-root", type=Path, help="Checkout root (defaults to the registry directory)")
    args = parser.parse_args()
    path = args.registry
    try:
        return verify_registry(
            json.loads(path.read_text(encoding="utf-8")),
            args.repository,
            args.repo_root if args.repo_root is not None else path.resolve().parent,
        )
    except (OSError, ValueError) as error:
        print(f"Invalid key registry: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
