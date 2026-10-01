import contextlib
import hashlib
import importlib.util
import io
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


class CheckKeyChecksumsTests(unittest.TestCase):
    def registry(self, algorithm="sha256", urls="https://example.org/key.pub"):
        return {"iris": {"parties": [{
            "slu": "party",
            "pub": urls,
            "chk": f"{algorithm}:{hashlib.new(algorithm, b'public key\n').hexdigest()}",
        }]}}

    def verify(self, registry, responses, **kwargs):
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
                result, download = self.verify(registry, [io.BytesIO(b"public key\n")])
                self.assertEqual(result, 0)
                download.assert_called_once()
                request = download.call_args.args[0]
                self.assertEqual(request.full_url, "https://example.org/key.pub")
                self.assertEqual(request.get_header("User-agent"), validator.USER_AGENT)
                self.assertEqual(request.get_header("Accept"), "*/*")
                self.assertEqual(download.call_args.kwargs, {"timeout": 30})

    def test_checks_every_mirror_after_mismatch(self):
        urls = ["https://example.org/first", "https://example.org/second"]
        result, download = self.verify(self.registry(urls=urls), [io.BytesIO(b"wrong key"), io.BytesIO(b"public key\n")])
        self.assertEqual(result, 1)
        self.assertEqual([call.args[0].full_url for call in download.call_args_list], urls)

    def test_all_mirrors_match(self):
        result, download = self.verify(self.registry(urls=["https://example.org/a", "https://example.org/b"]), [io.BytesIO(b"public key\n"), io.BytesIO(b"public key\n")])
        self.assertEqual(result, 0)
        self.assertEqual(download.call_count, 2)

    def test_line_endings_are_not_normalized(self):
        result, _ = self.verify(self.registry(), [io.BytesIO(b"public key\r\n")])
        self.assertEqual(result, 1)

    def test_download_failures(self):
        for error in (URLError("unreachable"), TimeoutError("timed out"), HTTPError("https://example.org", 404, "not found", {}, None)):
            with self.subTest(error=error):
                result, _ = self.verify(self.registry(), [error])
                self.assertEqual(result, 1)

    def test_oversized_key(self):
        result, _ = self.verify(self.registry(), [io.BytesIO(b"x" * (validator.MAX_KEY_BYTES + 1))])
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
        for invalid in (None, {}, {"iris": []}, {"iris": {"parties": {}}}, {"iris": {"parties": [None]}}, registry):
            with self.subTest(registry=invalid):
                with self.assertRaises(ValueError):
                    validator.parse_registry(invalid)

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

    def test_empty_registry_checks_no_locations(self):
        result, download = self.verify({"iris": {"parties": []}}, [])
        self.assertEqual(result, 0)
        download.assert_not_called()

    def test_repository_main_urls_use_checkout(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "certs" / "party.pub"
            path.parent.mkdir()
            path.write_bytes(b"public key\n")
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

    def test_external_repository_and_other_refs_are_downloaded(self):
        urls = [
            "https://raw.githubusercontent.com/other/repo/main/certs/party.pub",
            "https://raw.githubusercontent.com/worldcoin/iris-mpc/feature/certs/party.pub",
            "https://raw.githubusercontent.com/worldcoin/iris-mpc/refs/heads/main-extra/certs/party.pub",
        ]
        result, download = self.verify(
            self.registry(urls=urls), [io.BytesIO(b"public key\n") for _ in urls],
            repository="worldcoin/iris-mpc",
        )
        self.assertEqual(result, 0)
        self.assertEqual(download.call_count, len(urls))

    def test_checkout_key_and_external_mirror(self):
        with TemporaryDirectory() as directory:
            (Path(directory) / "party.pub").write_bytes(b"public key\n")
            urls = [
                "https://raw.githubusercontent.com/worldcoin/iris-mpc/main/party.pub",
                "https://example.org/party.pub",
            ]
            result, download = self.verify(
                self.registry(urls=urls), [io.BytesIO(b"public key\n")],
                repository="worldcoin/iris-mpc", repo_root=directory,
            )
            self.assertEqual(result, 0)
            download.assert_called_once()
            self.assertEqual(download.call_args.args[0].full_url, urls[1])
            self.assertEqual(download.call_args.kwargs, {"timeout": 30})

    def test_repository_path_cannot_escape_checkout(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "checkout"
            root.mkdir()
            outside = Path(directory) / "outside.pub"
            outside.write_bytes(b"public key\n")
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
            (root / "actual.pub").write_bytes(b"public key\n")
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
            (root / "actual" / "party.pub").write_bytes(b"public key\n")
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
                        key.write_bytes(b"public key\n")
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

    def test_raw_github_credentials_and_non_default_ports_are_rejected(self):
        for authority in (
            "user@raw.githubusercontent.com",
            "user:password@raw.githubusercontent.com",
            "raw.githubusercontent.com:444",
            "raw.githubusercontent.com:invalid",
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
