"""Offline archives and HTTP mocks; these are not benchmark/GPU results."""

from contextlib import ExitStack
import io
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock
import urllib.error
import zipfile

import numpy as np
from PIL import Image

from detector import image_data, prepare_imagenet as prep
from detector.common import save_json, sha256_file
from detector.io_utils import DETECTOR_ROOT


class Response(io.BytesIO):
    def __init__(self, payload, status=200, **headers):
        super().__init__(payload)
        self.status, self.headers = status, headers

    def getcode(self):
        return self.status


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=DETECTOR_ROOT / "tests")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def zipped_bytes(self):
        payload = io.BytesIO()
        with zipfile.ZipFile(payload, "w") as archive:
            archive.writestr("example.txt", "complete bytes, not a benchmark")
        return payload.getvalue()

    def spec(self, dataset):
        result = image_data.dataset_spec(dataset)
        result.update(classes=2, classes_per_task=2, tasks=1, train_per_class=2,
                      test_per_class=1, image_size=4)
        return result

    def use_spec(self, spec):
        stack = ExitStack()
        stack.enter_context(mock.patch.object(image_data, "dataset_spec", return_value=spec))
        stack.enter_context(mock.patch.object(prep, "dataset_spec", return_value=spec))
        return stack

    def test_full_mini_split_is_preserved_not_recreated(self):
        arrays = self.root / "original"
        arrays.mkdir()
        spec = self.spec("split_mini_imagenet")
        for split, count in (("train", 2), ("test", 1)):
            # Deliberately non-grouped order must not be sorted or repartitioned.
            labels = np.tile([1, 0], count)
            np.save(arrays / (split + "_y.npy"), labels)
            np.save(arrays / (split + "_x.npy"), np.arange(len(labels) * 48, dtype=np.uint8).reshape(-1, 4, 4, 3))
        archive = self.root / "mini.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            for name in prep.MINI_FILES:
                bundle.write(arrays / name, "miniImagenet/" + name)
        output = self.root / "prepared"
        output.mkdir()
        with self.use_spec(spec):
            target = prep.prepare_one("mini", archive, output, {"kind": "test"})
            before = {name: sha256_file(target / name) for name in prep.MINI_FILES}
            self.assertEqual(before, {name: sha256_file(arrays / name) for name in prep.MINI_FILES})
            self.assertEqual(target, prep.prepare_one("mini", archive, output, {"kind": "test"}))
            np.save(target / "train_y.npy", np.zeros(4, dtype=np.int64))
            with self.assertRaises(ValueError):
                prep.prepare_one("mini", archive, output, {})
        self.assertFalse(any(output.glob(".prepare_*")))

    def test_mini_raw_images_do_not_trigger_invented_split(self):
        with self.assertRaisesRegex(ValueError, "No arbitrary"):
            prep.prepare_mini(self.root, self.root / "product")
        self.assertFalse((self.root / "product").exists())

    def tiny_zip(self, missing=False):
        archive = self.root / "tiny.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("tiny/wnids.txt", "nB\nnA\n")
            annotations = []
            for name in ("nB", "nA"):
                for i in range(2):
                    pixels = np.full((4, 4), 60 if name == "nA" else 180, dtype=np.uint8)
                    buffer = io.BytesIO()
                    Image.fromarray(pixels).save(buffer, format="PNG")
                    bundle.writestr("tiny/train/{}/images/{}.PNG".format(name, i), buffer.getvalue())
                filename = name + "_val.PNG"
                annotations.append(filename + "\t" + name + "\t0\t0\t4\t4")
                if not (missing and name == "nB"):
                    bundle.writestr("tiny/val/images/" + filename, buffer.getvalue())
            bundle.writestr("tiny/val/val_annotations.txt", "\n".join(annotations))
        return archive

    def test_tiny_full_train_and_labelled_validation_rgb(self):
        archive = self.tiny_zip()
        output = self.root / "prepared"
        output.mkdir()
        with self.use_spec(self.spec("split_tiny_imagenet")):
            target = prep.prepare_one("tiny", archive, output, {"kind": "test"})
            record = image_data.register("split_tiny_imagenet", target)
            with np.load(target, allow_pickle=False) as arrays:
                self.assertEqual(arrays["train_data"].shape, (4, 3, 4, 4))
                self.assertEqual(arrays["train_lbls"].tolist(), [0, 0, 1, 1])
                self.assertEqual(arrays["tst_lbls"].tolist(), [0, 1])
                np.testing.assert_allclose(arrays["train_data"][0], 60 / 255., rtol=1e-6)
            metadata = json.loads((output / "tiny_preparation.json").read_text())
            self.assertEqual(metadata["conversion"]["class_order"], ["nA", "nB"])
            self.assertEqual(record["hashes"], metadata["registration"]["hashes"])
        self.assertFalse(any(output.glob(".prepare_*")))

    def test_missing_tiny_image_leaves_no_prepared_product(self):
        archive = self.tiny_zip(missing=True)
        output = self.root / "prepared"
        output.mkdir()
        with self.use_spec(self.spec("split_tiny_imagenet")):
            with self.assertRaisesRegex(ValueError, "annotations differ"):
                prep.prepare_one("tiny", archive, output, {})
        self.assertEqual(list(output.iterdir()), [])

    def test_existing_output_is_never_overwritten(self):
        archive = self.tiny_zip()
        (self.root / "tiny_imagenet.npz").write_bytes(b"user data")
        with self.assertRaises(FileExistsError):
            prep.prepare_one("tiny", archive, self.root, {})
        self.assertEqual((self.root / "tiny_imagenet.npz").read_bytes(), b"user data")

    def test_unsafe_zip_paths_links_duplicates_and_size_rejected(self):
        for name in ("../escape", "/absolute", "x\\escape", "C:/absolute", "link", "duplicate", "huge"):
            archive = self.root / "bad.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                if name == "link":
                    info = zipfile.ZipInfo("link")
                    info.create_system = 3
                    info.external_attr = (stat.S_IFLNK | 0o777) << 16
                    bundle.writestr(info, "../escape")
                else:
                    bundle.writestr(name, "data")
                    if name == "duplicate":
                        import warnings
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            bundle.writestr(name, "second")
            destination = self.root / "extract"
            with self.assertRaises(ValueError):
                prep.safe_extract(archive, destination, max_bytes=1 if name == "huge" else 1000)
            self.assertFalse(destination.exists())

    def test_download_resume_206_and_verified_cache(self):
        data = self.zipped_bytes()
        destination = self.root / "download.zip"
        destination.with_suffix(".zip.part").write_bytes(data[:20])
        save_json(dict(url="https://example.invalid/pinned", etag='"version1"'), destination.with_suffix(".zip.download.json"))
        response = Response(data[20:], 206, **{"Content-Range": "bytes 20-{}/{}".format(len(data) - 1, len(data))})
        with mock.patch.object(prep.urllib.request, "urlopen", return_value=response) as request:
            result = prep.download("https://example.invalid/pinned", destination)
        self.assertEqual(result.read_bytes(), data)
        self.assertEqual(request.call_args.args[0].get_header("Range"), "bytes=20-")
        with mock.patch.object(prep.urllib.request, "urlopen", side_effect=AssertionError("unnecessary download")):
            prep.download("https://example.invalid/pinned", destination)
        with self.assertRaisesRegex(ValueError, "provenance"):
            prep.download("https://example.invalid/different", destination)

    def test_range_ignored_restarts_instead_of_appending(self):
        data = self.zipped_bytes()
        destination = self.root / "download.zip"
        destination.with_suffix(".zip.part").write_bytes(data[:20])
        save_json(dict(url="https://example.invalid/pinned"), destination.with_suffix(".zip.download.json"))
        with mock.patch.object(prep.urllib.request, "urlopen", return_value=Response(data, 200, **{"Content-Length": str(len(data))})):
            prep.download("https://example.invalid/pinned", destination)
        self.assertEqual(destination.read_bytes(), data)

    def test_wrong_range_hash_or_incomplete_bytes_not_promoted(self):
        data = self.zipped_bytes()
        for i, response in enumerate((Response(data, 206, **{"Content-Range": "bytes 5-20/21"}),
                                     Response(data, 200),
                                     Response(data[:10], 200, **{"Content-Length": str(len(data))}))):
            destination = self.root / ("download{}.zip".format(i))
            with mock.patch.object(prep.urllib.request, "urlopen", return_value=response):
                with self.assertRaises((ValueError, OSError)):
                    prep.download("https://example.invalid/pinned", destination,
                                  expected_sha256="a" * 64 if i == 1 else None, attempts=1)
            self.assertFalse(destination.exists())

    def test_completed_partial_416_is_crc_checked(self):
        data = self.zipped_bytes()
        destination = self.root / "download.zip"
        destination.with_suffix(".zip.part").write_bytes(data)
        save_json(dict(url="https://example.invalid/pinned", expected_bytes=len(data)), destination.with_suffix(".zip.download.json"))
        error = urllib.error.HTTPError("https://example.invalid/pinned", 416, "range", {}, None)
        with mock.patch.object(prep.urllib.request, "urlopen", side_effect=error):
            prep.download("https://example.invalid/pinned", destination)
        self.assertEqual(destination.read_bytes(), data)

    def test_lock_rejects_second_worker_and_releases(self):
        with prep.preparation_lock(self.root):
            with self.assertRaisesRegex(RuntimeError, "Another"):
                with prep.preparation_lock(self.root):
                    self.fail("lock bypassed")
        with prep.preparation_lock(self.root):
            pass

    def test_resolve_revision_requires_immutable_commit(self):
        for revision in ("main", "a" * 40):
            with mock.patch.object(prep.urllib.request, "urlopen", return_value=Response(json.dumps({"sha": revision}).encode())):
                if revision == "main":
                    with self.assertRaises(ValueError):
                        prep.resolve_revision()
                else:
                    self.assertEqual(prep.resolve_revision(), revision)

    def test_offline_cli_prepares_both_without_network_or_gpu(self):
        mini = self.root / "mini.zip"
        with zipfile.ZipFile(mini, "w") as archive:
            for split, count in (("train", 2), ("test", 1)):
                labels = np.tile([1, 0], count)
                for suffix, array in (("y", labels), ("x", np.zeros((len(labels), 4, 4, 3), dtype=np.uint8))):
                    buffer = io.BytesIO()
                    np.save(buffer, array)
                    archive.writestr("mini/{}_{}.npy".format(split, suffix), buffer.getvalue())
        tiny = self.tiny_zip()
        specs = {name: self.spec(name) for name in ("split_mini_imagenet", "split_tiny_imagenet")}
        with mock.patch.object(prep, "resolve_revision", side_effect=AssertionError("offline means offline")), \
             mock.patch.object(prep.shutil, "disk_usage", return_value=mock.Mock(free=30 * 1024**3)), \
             mock.patch.object(image_data, "dataset_spec", side_effect=lambda name: specs[name]), \
             mock.patch.object(prep, "dataset_spec", side_effect=lambda name: specs[name]):
            prep.main(["--output-dir", str(self.root / "prepared"), "--cache-dir", str(self.root / "cache"),
                       "--mini-archive", str(mini), "--tiny-archive", str(tiny)])
        self.assertTrue((self.root / "prepared/miniImagenet/train_x.npy").is_file())
        self.assertTrue((self.root / "prepared/tiny_imagenet.npz").is_file())
        self.assertFalse(list(self.root.rglob("plan.json")))


if __name__ == "__main__":
    unittest.main()
