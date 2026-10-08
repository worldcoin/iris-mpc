import base64
import contextlib
import hashlib
import importlib.util
import io
import itertools
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError


spec = importlib.util.spec_from_file_location(
    "check_key_checksums", Path(__file__).resolve().parents[1] / "check-key-checksums.py"
)
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


PARTY_SLUGS = ("fau", "berkeley-university", "kaist-university")
KEY = base64.b64encode(bytes(range(32)))


class CheckKeyChecksumsTests(unittest.TestCase):
    def registry(self, algorithm="sha256", urls="https://example.org/key.pub", data=KEY):
        return {"iris": {"parties": [{
            "slu": slug,
            "pub": urls,
            "chk": f"{algorithm}:{hashlib.new(algorithm, data).hexdigest()}",
        } for slug in PARTY_SLUGS]}}

    def verify(self, registry, responses, **kwargs):
        # Each party uses the supplied mirror responses, with fresh byte streams.
        responses = [
            io.BytesIO(response) if isinstance(response, bytes) else response
            for _ in registry["iris"]["parties"] for response in responses
        ]
        with patch.object(validator, "urlopen", side_effect=responses) as download:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                result = validator.verify_registry(registry, **kwargs)
        return result, download

    def test_both_algorithms_and_single_url(self):
        for algorithm in ("sha256", "sha512"):
            with self.subTest(algorithm=algorithm):
                registry = self.registry(algorithm)
                party = registry["iris"]["parties"][0]
                name, digest = party["chk"].split(":")
                party["chk"] = f"{name}:{digest.upper()}"
                result, download = self.verify(registry, [KEY])
                self.assertEqual(result, 0)
                self.assertEqual(download.call_count, 3)
                request = download.call_args.args[0]
                self.assertEqual(request.full_url, "https://example.org/key.pub")
                self.assertEqual(request.get_header("User-agent"), validator.USER_AGENT)
                self.assertEqual(request.get_header("Accept"), "*/*")
                self.assertEqual(download.call_args.kwargs, {"timeout": 30})

    def test_checks_every_mirror_after_mismatch(self):
        urls = ["https://example.org/first", "https://example.org/second"]
        result, download = self.verify(self.registry(urls=urls), [b"wrong key", KEY])
        self.assertEqual(result, 1)
        self.assertEqual([call.args[0].full_url for call in download.call_args_list], urls * 3)

    def test_all_mirrors_match(self):
        result, download = self.verify(self.registry(urls=["https://example.org/a", "https://example.org/b"]), [KEY, KEY])
        self.assertEqual(result, 0)
        self.assertEqual(download.call_count, 6)

    def test_line_endings_are_not_normalized(self):
        result, _ = self.verify(self.registry(), [KEY + b"\r\n"])
        self.assertEqual(result, 1)

    def test_all_party_orders_are_accepted(self):
        for slugs in itertools.permutations(PARTY_SLUGS):
            with self.subTest(slugs=slugs):
                registry = self.registry()
                for party, slug in zip(registry["iris"]["parties"], slugs):
                    party["slu"] = slug
                result, download = self.verify(registry, [KEY])
                self.assertEqual(result, 0)
                self.assertEqual(download.call_count, 3)

    def test_incorrect_party_sets_are_rejected_before_download(self):
        for slugs in (
            (), PARTY_SLUGS[:2], (*PARTY_SLUGS, "unknown"),
            ("fau", "fau", "kaist-university"),
            ("fau", "berkeley", "kaist-university"),
            ("fau", "berkeley-university", "kaist-univeristy"),
            ("FAU", "berkeley-university", "kaist-university"),
            (" fau", "berkeley-university", "kaist-university"),
            (None, "berkeley-university", "kaist-university"),
            ([], "berkeley-university", "kaist-university"),
        ):
            with self.subTest(slugs=slugs):
                template = self.registry()["iris"]["parties"][0]
                registry = {"iris": {"parties": [dict(template, slu=slug) for slug in slugs]}}
                with patch.object(validator, "urlopen") as download:
                    with self.assertRaises(ValueError):
                        validator.verify_registry(registry)
                    download.assert_not_called()

    def test_malformed_keys_fail_even_with_matching_checksums(self):
        malformed = (
            b"not base64", b"\xff" * 44,
            KEY + b"\n", KEY + b"\r\n", b" " + KEY,
            KEY[:4] + b"\t" + KEY[4:],
            KEY[:-1], KEY + b"=", b"=" + KEY,
            # The last symbol has nonzero unused bits but decodes to the same bytes.
            KEY[:-2] + b"9=",
            base64.urlsafe_b64encode(b"\xfb" * 32),
        )
        for data in malformed:
            for algorithm in ("sha256", "sha512"):
                with self.subTest(data=data, algorithm=algorithm):
                    result, download = self.verify(self.registry(algorithm, data=data), [data])
                    self.assertEqual(result, 1)
                    self.assertEqual(download.call_count, 3)

    def test_wrong_decoded_lengths_fail_with_matching_checksums(self):
        for length in (0, 1, 31, 33, 64):
            with self.subTest(length=length):
                data = base64.b64encode(bytes(length))
                with self.assertRaisesRegex(ValueError, "exactly 32 bytes"):
                    validator.validate_public_key(data)
                result, _ = self.verify(self.registry(data=data), [data])
                self.assertEqual(result, 1)

    def test_checksum_is_verified_before_decoding(self):
        with patch.object(validator, "validate_public_key") as decode:
            result, _ = self.verify(self.registry(), [b"wrong checksum and malformed key"])
            self.assertEqual(result, 1)
            decode.assert_not_called()

    def test_standard_base64_alphabet_is_accepted(self):
        data = base64.b64encode(b"\xfb\xff" * 16)
        self.assertIn(b"+", data)
        self.assertIn(b"/", data)
        result, _ = self.verify(self.registry(data=data), [data])
        self.assertEqual(result, 0)

    def test_every_mirror_is_decoded(self):
        data = KEY + b"\n"
        registry = self.registry(urls=["https://example.org/a", "https://example.org/b"], data=data)
        with patch.object(validator, "validate_public_key", wraps=validator.validate_public_key) as decode:
            result, download = self.verify(registry, [data, data])
        self.assertEqual(result, 1)
        self.assertEqual(download.call_count, 6)
        self.assertEqual(decode.call_count, 6)

    def test_download_failures(self):
        for error in (URLError("unreachable"), TimeoutError("timed out"), HTTPError("https://example.org", 404, "not found", {}, None)):
            with self.subTest(error=error):
                result, _ = self.verify(self.registry(), [error])
                self.assertEqual(result, 1)
                if isinstance(error, HTTPError):
                    error.close()

    def test_oversized_key(self):
        result, _ = self.verify(self.registry(), [b"x" * (validator.MAX_KEY_BYTES + 1)])
        self.assertEqual(result, 1)

    def test_invalid_entries_rejected_before_download(self):
        changes = [
            {"chk": "poly1305:" + "a" * 32},
            {"chk": "sha256:abc"},
            {"chk": "sha256:" + "z" * 64},
            {"chk": None},
            {"pub": []},
            {"pub": [123]},
            {"pub": "file:///etc/passwd"},
            {"pub": "https:///key.pub"},
            {"slu": ""},
        ]
        for change in changes:
            with self.subTest(change=change):
                registry = self.registry()
                registry["iris"]["parties"][0].update(change)
                with patch.object(validator, "urlopen") as download:
                    with self.assertRaises(ValueError):
                        validator.verify_registry(registry)
                    download.assert_not_called()

    def test_invalid_registry_structure_and_duplicate_parties(self):
        registry = self.registry()
        registry["iris"]["parties"] *= 2
        for invalid in (None, {}, {"iris": []}, {"iris": {"parties": {}}}, {"iris": {"parties": [None] * 3}}, registry):
            with self.subTest(registry=invalid):
                with self.assertRaises(ValueError):
                    validator.parse_registry(invalid)

    def test_rejected_urls_fail_before_download(self):
        for url in (
            "ftp://example.org/key", "file:///key", "//example.org/key",
            "https:///key", "https://", "https://example.org/key#", "https://example.org/key#key",
            "https://user@example.org/key", "https://user:password@example.org/key",
            "https://:password@example.org/key", "https://@example.org/key",
            "https://raw.githubusercontent.com:invalid/worldcoin/iris-mpc/main/key.pub",
            "https://example.org:invalid/key", "https://example.org:65536/key", "https://[invalid/key",
            "https://user@raw.githubusercontent.com/worldcoin/iris-mpc/main/key.pub",
        ):
            with self.subTest(url=url), patch.object(validator, "urlopen") as download:
                with self.assertRaises(ValueError):
                    validator.verify_registry(self.registry(urls=url), repository="worldcoin/iris-mpc")
                download.assert_not_called()

    def test_http_https_hosts_ports_and_queries_are_accepted(self):
        for url in ("http://example.org/key", "https://example.org:8443/key?version=2", "https://[::1]/key"):
            with self.subTest(url=url):
                result, _ = self.verify(self.registry(urls=url), [KEY])
                self.assertEqual(result, 0)

    def test_url_fragments_are_rejected_before_download(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "party.pub").write_bytes(b"changed public key\n")
            urls = [
                "https://raw.githubusercontent.com/worldcoin/iris-mpc/main/party.pub#key",
                "https://raw.githubusercontent.com/worldcoin/iris-mpc/refs/heads/main/party.pub#key",
                "https://raw.githubusercontent.com/worldcoin/iris-mpc/main/party.pub#",
                "https://example.org/party.pub#key",
            ]
            for url in urls:
                with self.subTest(url=url):
                    # The old digest could match main while the checkout has a new key.
                    with patch.object(validator, "urlopen") as download:
                        with self.assertRaisesRegex(ValueError, "must not contain fragments"):
                            validator.verify_registry(
                                self.registry(urls=url), "worldcoin/iris-mpc", root,
                            )
                        download.assert_not_called()

    def test_empty_registry_is_rejected_before_download(self):
        with patch.object(validator, "urlopen") as download:
            with self.assertRaisesRegex(ValueError, "exactly three parties"):
                validator.verify_registry({"iris": {"parties": []}})
            download.assert_not_called()

    def test_repository_main_urls_use_checkout(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "certs" / "party.pub"
            path.parent.mkdir()
            path.write_bytes(KEY)
            for ref in ("main", "refs/heads/main"):
                with self.subTest(ref=ref):
                    url = f"https://raw.githubusercontent.com/worldcoin/iris-mpc/{ref}/certs/party.pub"
                    result, download = self.verify(
                        self.registry(urls=url), [],
                        repository="worldcoin/iris-mpc", repo_root=directory,
                    )
                    self.assertEqual(result, 0)
                    download.assert_not_called()
            path.write_bytes(b"changed key\n")
            result, download = self.verify(
                self.registry(urls=url), [],
                repository="worldcoin/iris-mpc", repo_root=directory,
            )
            self.assertEqual(result, 1)
            download.assert_not_called()

    def test_missing_repository_file_does_not_fall_back_to_main(self):
        with TemporaryDirectory() as directory:
            url = "https://raw.githubusercontent.com/worldcoin/iris-mpc/main/certs/missing.pub"
            result, download = self.verify(
                self.registry(urls=url), [],
                repository="worldcoin/iris-mpc", repo_root=directory,
            )
            self.assertEqual(result, 1)
            download.assert_not_called()

    def test_malformed_checkout_keys_fail_with_matching_checksums(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "unpublished.pub"
            url = "https://raw.githubusercontent.com/worldcoin/iris-mpc/main/unpublished.pub"
            for data in (b"not base64", KEY + b"\n", base64.b64encode(bytes(31)), KEY[:-2] + b"9="):
                for algorithm in ("sha256", "sha512"):
                    with self.subTest(data=data, algorithm=algorithm):
                        path.write_bytes(data)
                        result, download = self.verify(
                            self.registry(algorithm, urls=url, data=data), [],
                            repository="worldcoin/iris-mpc", repo_root=directory,
                        )
                        self.assertEqual(result, 1)
                        download.assert_not_called()

    def test_external_repository_and_other_refs_are_downloaded(self):
        urls = [
            "https://raw.githubusercontent.com/other/repo/main/certs/party.pub",
            "https://raw.githubusercontent.com/worldcoin/iris-mpc/feature/certs/party.pub",
            "https://raw.githubusercontent.com/worldcoin/iris-mpc/refs/heads/main-extra/certs/party.pub",
        ]
        result, download = self.verify(
            self.registry(urls=urls), [KEY for _ in urls],
            repository="worldcoin/iris-mpc",
        )
        self.assertEqual(result, 0)
        self.assertEqual(download.call_count, 3 * len(urls))

    def test_checkout_key_and_external_mirror(self):
        with TemporaryDirectory() as directory:
            (Path(directory) / "party.pub").write_bytes(KEY)
            urls = [
                "https://raw.githubusercontent.com/worldcoin/iris-mpc/main/party.pub",
                "https://example.org/party.pub",
            ]
            result, download = self.verify(
                self.registry(urls=urls), [KEY],
                repository="worldcoin/iris-mpc", repo_root=directory,
            )
            self.assertEqual(result, 0)
            self.assertEqual(download.call_count, 3)
            self.assertEqual(download.call_args.args[0].full_url, urls[1])
            self.assertEqual(download.call_args.kwargs, {"timeout": 30})

    def test_repository_path_cannot_escape_checkout(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "checkout"
            root.mkdir()
            outside = Path(directory) / "outside.pub"
            outside.write_bytes(KEY)
            (root / "link.pub").symlink_to(outside)
            for relative in ("../outside.pub", "%2e%2e/outside.pub", "%2foutside.pub", "link.pub"):
                with self.subTest(relative=relative):
                    url = f"https://raw.githubusercontent.com/worldcoin/iris-mpc/main/{relative}"
                    result, download = self.verify(
                        self.registry(urls=url), [],
                        repository="worldcoin/iris-mpc", repo_root=root,
                    )
                    self.assertEqual(result, 1)
                    download.assert_not_called()

    def test_repository_key_symlink_inside_checkout_is_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "actual.pub").write_bytes(KEY)
            (root / "link.pub").symlink_to("actual.pub")
            for ref in ("main", "refs/heads/main"):
                with self.subTest(ref=ref):
                    url = f"https://raw.githubusercontent.com/worldcoin/iris-mpc/{ref}/link.pub"
                    with self.assertRaisesRegex(ValueError, "must not contain symlinks"):
                        validator.checkout_path(url, "worldcoin/iris-mpc", root)
                    result, download = self.verify(
                        self.registry(urls=url), [],
                        repository="worldcoin/iris-mpc", repo_root=root,
                    )
                    self.assertEqual(result, 1)
                    download.assert_not_called()

    def test_repository_key_symlinked_parent_is_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "actual").mkdir()
            (root / "actual" / "party.pub").write_bytes(KEY)
            (root / "certs").symlink_to("actual", target_is_directory=True)
            for relative in ("certs/party.pub", "%63erts/party.pub"):
                with self.subTest(relative=relative):
                    url = f"https://raw.githubusercontent.com/worldcoin/iris-mpc/main/{relative}"
                    with self.assertRaisesRegex(ValueError, "must not contain symlinks"):
                        validator.checkout_path(url, "worldcoin/iris-mpc", root)
                    result, download = self.verify(
                        self.registry(urls=url), [],
                        repository="worldcoin/iris-mpc", repo_root=root,
                    )
                    self.assertEqual(result, 1)
                    download.assert_not_called()

    def test_dangling_repository_key_symlink_is_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "link.pub").symlink_to("missing.pub")
            url = "https://raw.githubusercontent.com/worldcoin/iris-mpc/main/link.pub"
            with self.assertRaisesRegex(ValueError, "must not contain symlinks"):
                validator.checkout_path(url, "worldcoin/iris-mpc", root)

    def test_equivalent_raw_github_authorities_use_checkout(self):
        authorities = (
            "RAW.GITHUBUSERCONTENT.COM",
            "raw.githubusercontent.com:443",
            "Raw.Githubusercontent.Com:443",
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            key = root / "party.pub"
            for authority in authorities:
                for ref in ("main", "refs/heads/main"):
                    with self.subTest(authority=authority, ref=ref):
                        url = f"https://{authority}/worldcoin/iris-mpc/{ref}/party.pub"
                        key.write_bytes(KEY)
                        result, download = self.verify(
                            self.registry(urls=url), [],
                            repository="worldcoin/iris-mpc", repo_root=root,
                        )
                        self.assertEqual(result, 0)
                        download.assert_not_called()
                        # Retaining the old checksum must fail against the new local key.
                        key.write_bytes(b"changed public key\n")
                        result, download = self.verify(
                            self.registry(urls=url), [],
                            repository="worldcoin/iris-mpc", repo_root=root,
                        )
                        self.assertEqual(result, 1)
                        download.assert_not_called()

    def test_raw_github_non_default_ports_are_rejected(self):
        for authority in (
            "raw.githubusercontent.com:444",
        ):
            with self.subTest(authority=authority):
                url = f"https://{authority}/worldcoin/iris-mpc/main/party.pub"
                with self.assertRaises(ValueError):
                    validator.checkout_path(url, "worldcoin/iris-mpc", ".")
                result, download = self.verify(
                    self.registry(urls=url), [], repository="worldcoin/iris-mpc",
                )
                self.assertEqual(result, 1)
                download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
