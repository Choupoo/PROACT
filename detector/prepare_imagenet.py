"""Download the BrainWash authors' datasets and prepare full detector inputs.

CPU-only. No pickle loading, invented miniImageNet split, resizing, subsampling,
upstream edits, or automatic experiment launch. All outputs stay in detector/.
"""

import argparse
from contextlib import contextmanager
import hashlib
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
import time
import urllib.error
import urllib.request
import zipfile

import numpy as np

from detector.common import save_json, sha256_file
from detector.image_data import dataset_spec, register
from detector.io_utils import DETECTOR_ROOT, ensure_output_path

REPOSITORY = "mintlabvandy/BrainWash-CVPR24"
ARCHIVES = {"mini": "miniImagenet.zip", "tiny": "tiny-imagenet-200.zip"}
MINI_FILES = ("train_x.npy", "train_y.npy", "test_x.npy", "test_y.npy")
DEFAULT_DATA = DETECTOR_ROOT / "data_prepared"
DEFAULT_CACHE = DETECTOR_ROOT / "data_downloads"


@contextmanager
def preparation_lock(root):
    # flock releases on process exit; a stale lock file does not block recovery.
    import fcntl
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".prepare.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another data-preparation process is active: " + str(root)) from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def resolve_revision():
    url = "https://huggingface.co/api/datasets/" + REPOSITORY + "/revision/main"
    with urllib.request.urlopen(url, timeout=60) as response:
        record = json.loads(response.read(4 * 1024 * 1024))
    revision = record.get("sha", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Hugging Face did not return an immutable dataset revision.")
    return revision


def download(url, destination, expected_sha256=None, attempts=4):
    """Resume a pinned HTTPS object; never disable certificate verification.

    Interrupted bytes stay in .part. A 200 response to Range restarts the file,
    rather than appending an entire archive to its interrupted prefix.
    """
    destination = ensure_output_path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    sidecar = destination.with_name(destination.name + ".download.json")
    ensure_output_path(partial)
    ensure_output_path(sidecar)
    if expected_sha256 and not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("Expected SHA-256 must contain 64 lowercase hexadecimal characters.")
    saved = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    if destination.exists():
        digest = sha256_file(destination)
        if saved.get("url") != url or saved.get("sha256") != digest:
            raise ValueError("Existing download lacks matching provenance: " + str(destination))
        if expected_sha256 and digest != expected_sha256:
            raise ValueError("Downloaded archive SHA-256 differs from the expected hash.")
        return destination
    if partial.exists() and saved.get("url") != url:
        raise ValueError("Interrupted download belongs to another revision; use a new cache directory.")
    if not url.startswith("https://"):
        raise ValueError("Dataset downloads require HTTPS.")
    for attempt in range(attempts):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "PROACT-detector-data/3", "Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = "bytes={}-".format(offset)
            if saved.get("etag"):
                headers["If-Range"] = saved["etag"]
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as response:
                status = response.getcode()
                append = offset > 0 and status == 206
                total = None
                if status == 206:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                    if not match or int(match[1]) != offset or int(match[2]) + 1 != int(match[3]):
                        raise ValueError("Invalid partial-download range; refusing to combine bytes.")
                    total = int(match[3])
                elif status != 200:
                    raise ValueError("Unexpected download status: " + str(status))
                length = response.headers.get("Content-Length")
                if length is not None and total is None:
                    total = int(length)
                if not append:
                    offset = 0
                saved = dict(url=url, etag=response.headers.get("ETag"), expected_bytes=total)
                save_json(saved, sidecar)
                written, last_print = offset, offset
                with partial.open("ab" if append else "wb") as output:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        output.write(block)
                        written += len(block)
                        if written - last_print >= 32 * 1024 * 1024:
                            print("{}: {:.0f} MiB{}".format(destination.name, written / 2**20,
                                  " / {:.0f} MiB".format(total / 2**20) if total else ""), flush=True)
                            last_print = written
                if total is not None and written != total:
                    raise OSError("Incomplete response: received {} of {} bytes.".format(written, total))
            digest = sha256_file(partial)
            if expected_sha256 and digest != expected_sha256:
                raise ValueError("Archive SHA-256 mismatch; interrupted file retained for inspection: " + str(partial))
            # A ZIP CRC check also catches truncation when Content-Length is absent.
            with zipfile.ZipFile(partial) as archive:
                bad = archive.testzip()
                if bad:
                    raise ValueError("Archive CRC failed: " + bad)
            os.replace(str(partial), str(destination))
            save_json(dict(saved, sha256=digest, bytes=destination.stat().st_size), sidecar)
            return destination
        except urllib.error.HTTPError as error:
            # 416 can occur after all bytes arrived but before final validation.
            if error.code == 416 and offset and offset == saved.get("expected_bytes"):
                digest = sha256_file(partial)
                if expected_sha256 and digest != expected_sha256:
                    raise ValueError("Completed partial file has the wrong SHA-256.") from error
                with zipfile.ZipFile(partial) as archive:
                    if archive.testzip():
                        raise ValueError("Completed partial archive failed its CRC check.") from error
                os.replace(str(partial), str(destination))
                save_json(dict(saved, sha256=digest, bytes=offset), sidecar)
                return destination
            if error.code not in (408, 429, 500, 502, 503, 504) or attempt + 1 == attempts:
                raise
        except (OSError, urllib.error.URLError, http.client.IncompleteRead):
            if attempt + 1 == attempts:
                raise
        print("Download interrupted; retrying without discarding valid partial bytes.", flush=True)
        time.sleep(min(2 ** attempt, 8))
    raise RuntimeError("Download attempts exhausted.")


def safe_extract(archive_path, destination, max_bytes=12 * 1024**3):
    """Extract only regular members, with bounds and no path/link traversal."""
    destination = ensure_output_path(destination)
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        if len(members) > 250000 or sum(m.file_size for m in members) > max_bytes:
            raise ValueError("Dataset archive exceeds extraction limits.")
        seen = set()
        for member in members:
            name = PurePosixPath(member.filename)
            mode = member.external_attr >> 16
            if (name.is_absolute() or ".." in name.parts or "\\" in member.filename or
                    not name.parts or ":" in name.parts[0] or
                    stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR)):
                raise ValueError("Unsafe archive member: " + member.filename)
            target = ensure_output_path(destination.joinpath(*name.parts))
            if target in seen or (target != destination and destination not in target.parents):
                raise ValueError("Duplicate or escaped archive member: " + member.filename)
            seen.add(target)
        for member in members:
            target = destination.joinpath(*PurePosixPath(member.filename).parts)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, 1024 * 1024)


def prepare_mini(extracted, output):
    candidates = [p.parent for p in Path(extracted).rglob("train_x.npy")
                  if all((p.parent / name).is_file() for name in MINI_FILES)]
    if len(candidates) != 1:
        raise ValueError("miniImageNet must supply exactly one existing train/test NPY split. "
                         "No arbitrary 500/100 split is created. Inspect archive contents: " + str(extracted))
    source = candidates[0]
    record = register("split_mini_imagenet", source)
    output.mkdir()
    for name in MINI_FILES:
        shutil.copyfile(source / name, output / name)
    return dict(format="authors_prepared_npy", split="preserved_byte_for_byte", original_hashes=record["hashes"])


def archive_metadata(path, root):
    """Ignore only known OS sidecars, never unlabelled or unreadable images."""
    return any(part in (".DS_Store", "Thumbs.db", "__MACOSX") or part.startswith("._")
               for part in path.relative_to(root).parts)


def tiny_validation_index(root, annotations, classes):
    """Resolve original flat and class-organized validation files by annotation.

    PROACT's existing preprocessing moves images into val/images/<class>/.
    Read these files in place; do not infer/replace labels from folder names.
    """
    validation = root / "val"
    files, unexpected = {}, []
    ignored = 0
    layouts = set()
    for path in sorted(validation.rglob("*")):
        if not path.is_file() or path == validation / "val_annotations.txt":
            continue
        if archive_metadata(path, validation):
            ignored += 1
            continue
        name = path.name
        relative = path.relative_to(validation)
        if name not in annotations:
            unexpected.append(str(relative))
            continue
        if name in files:
            raise ValueError("Duplicate validation image filename: {} ({}, {}).".format(name, files[name], path))
        label = annotations[name]
        parents = relative.parts[:-1]
        allowed = {("images",): "flat", ("images", label): "images/class",
                   (label,): "class", (label, "images"): "class/images"}
        if parents not in allowed:
            if any(part in classes and part != label for part in parents):
                raise ValueError("Validation folder disagrees with annotation: {} must be {}.".format(relative, label))
            raise ValueError("Unsupported validation image layout: " + str(relative))
        layouts.add(allowed[parents])
        files[name] = path
    missing = sorted(set(annotations) - set(files))
    if missing or unexpected:
        raise ValueError("Validation images and annotations differ: {} annotations, {} matched images; "
                         "{} missing (examples: {}); {} unexpected files (examples: {}).".format(
                             len(annotations), len(files), len(missing), missing[:5],
                             len(unexpected), unexpected[:5]))
    return files, dict(layouts=sorted(layouts), ignored_os_sidecars=ignored)


def prepare_tiny(extracted, output, scratch):
    from PIL import Image
    spec = dataset_spec("split_tiny_imagenet")
    candidates = [p.parent for p in Path(extracted).rglob("wnids.txt")
                  if (p.parent / "train").is_dir() and (p.parent / "val/val_annotations.txt").is_file()]
    if len(candidates) != 1:
        raise ValueError("Expected one complete standard tiny-imagenet-200 tree with validation annotations.")
    root = candidates[0]
    classes = sorted((root / "wnids.txt").read_text().split())
    if len(classes) != spec["classes"] or len(set(classes)) != len(classes):
        raise ValueError("tinyImageNet has incomplete or duplicate class identifiers.")
    if sorted(p.name for p in (root / "train").iterdir()
              if p.is_dir() and not archive_metadata(p, root)) != classes:
        raise ValueError("Training class folders differ from wnids.txt.")
    mapping = {name: i for i, name in enumerate(classes)}
    annotations = {}
    for line in (root / "val/val_annotations.txt").read_text().splitlines():
        fields = line.split()
        if (len(fields) < 2 or fields[0] in annotations or fields[1] not in mapping or
                Path(fields[0]).name != fields[0] or "\\" in fields[0]):
            raise ValueError("Invalid or duplicate validation annotation.")
        annotations[fields[0]] = fields[1]
    validation_files, validation_layout = tiny_validation_index(root, annotations, mapping)
    for name in classes:
        if sum(label == name for label in annotations.values()) != spec["test_per_class"]:
            raise ValueError("Validation annotations have the wrong image count for " + name)
    print("tinyImageNet validation: {} annotated images; layout {}".format(
        len(validation_files), ", ".join(validation_layout["layouts"])), flush=True)
    arrays, index = {}, hashlib.sha256()
    for split, xkey, ykey, per_class in (("train", "train_data", "train_lbls", spec["train_per_class"]),
                                        ("val", "tst_data", "tst_lbls", spec["test_per_class"])):
        n = spec["classes"] * per_class
        size = spec["image_size"]
        images = np.lib.format.open_memmap(Path(scratch) / (xkey + ".npy"), mode="w+", dtype=np.float32,
                                           shape=(n, 3, size, size))
        labels = np.empty(n, dtype=np.int64)
        offset = 0
        for name in classes:
            files = (sorted(p for p in (root / "train" / name / "images").iterdir()
                            if not archive_metadata(p, root)) if split == "train" else
                     [validation_files[f] for f in sorted(annotations) if annotations[f] == name])
            if len(files) != per_class or not all(p.is_file() for p in files):
                raise ValueError("{} {} does not contain {} images.".format(split, name, per_class))
            for path in files:
                with Image.open(path) as image:
                    pixels = np.asarray(image.convert("RGB"))
                if pixels.shape != (size, size, 3):
                    raise ValueError("Unexpected native image size; resizing is disabled: " + str(path))
                images[offset] = pixels.transpose(2, 0, 1).astype(np.float32) / 255.
                labels[offset] = mapping[name]
                index.update((str(path.relative_to(root)) + "\t" + str(mapping[name]) + "\n").encode())
                offset += 1
            if (mapping[name] + 1) % 20 == 0:
                print("tinyImageNet {}: {}/{} classes converted".format(split, mapping[name] + 1, len(classes)), flush=True)
        images.flush()
        arrays[xkey], arrays[ykey] = images, labels
    # Uncompressed NPZ avoids repeated DEFLATE work at each upstream subprocess.
    # NumPy streams memory-mapped arrays into ZIP members instead of copying all
    # 5.4 GB into a second conversion-time ndarray.
    np.savez(output, **arrays)
    return dict(format="standard_tiny_imagenet", class_order=classes, image_index_sha256=index.hexdigest(),
                validation_layout=validation_layout,
                split="all official train images / all labelled validation images", resizing=False,
                row_order="sorted class ID, then sorted original filename", rgb="Pillow RGB / 255, float32 NCHW")


def prepare_one(short, archive, output_root, source_info):
    dataset = "split_mini_imagenet" if short == "mini" else "split_tiny_imagenet"
    target = output_root / ("miniImagenet" if short == "mini" else "tiny_imagenet.npz")
    provenance = output_root / (short + "_preparation.json")
    digest = sha256_file(archive)
    if provenance.exists():
        previous = json.loads(provenance.read_text())
        if previous["archive_sha256"] != digest:
            raise ValueError("Prepared data came from another archive; use a new --output-dir.")
        record = register(dataset, target)
        if record["hashes"] != previous["registration"]["hashes"]:
            raise ValueError("Prepared data changed after conversion.")
        print("Verified existing full data:", target, flush=True)
        return target
    if target.exists():
        raise FileExistsError("Untracked prepared output exists; refusing to overwrite: " + str(target))
    # This temporary directory is unique and automatically cleaned, never a
    # broad user path. A failed conversion cannot appear as prepared output.
    with tempfile.TemporaryDirectory(prefix=".prepare_" + short + "_", dir=str(output_root)) as temporary:
        stage = Path(temporary)
        extracted = stage / "extracted"
        safe_extract(archive, extracted)
        product = stage / target.name
        if short == "mini":
            conversion = prepare_mini(extracted, product)
        else:
            conversion = prepare_tiny(extracted, product, stage)
        print("Validating all images and class counts:", dataset, flush=True)
        record = register(dataset, product)
        # Reruns require the final target, this sidecar and matching file hashes.
        # A crash between promotion and sidecar saving fails closed as untracked
        # output; it never silently overwrites or trusts the promoted bytes.
        record["files"] = {k: str(target / Path(v).name) if short == "mini" else str(target)
                           for k, v in record["files"].items()}
        metadata = dict(dataset=dataset, archive_sha256=digest, archive_bytes=archive.stat().st_size,
                        source=source_info, conversion=conversion, registration=record)
        os.replace(str(product), str(target))
        save_json(metadata, provenance)
    print("Ready:", target, flush=True)
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_DATA))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    parser.add_argument("--dataset", choices=("both", "mini", "tiny"), default="both")
    parser.add_argument("--revision", help="Optional immutable 40-character Hugging Face dataset revision.")
    for short in ARCHIVES:
        parser.add_argument("--" + short + "-archive", help="Existing trusted ZIP: no download for this dataset.")
        parser.add_argument("--" + short + "-sha256", help="Optional externally obtained archive SHA-256.")
    args = parser.parse_args(argv)
    root, cache = ensure_output_path(args.output_dir), ensure_output_path(args.cache_dir)
    selected = list(ARCHIVES) if args.dataset == "both" else [args.dataset]
    if root == cache:
        raise ValueError("Use separate prepared-data and download-cache directories.")
    with preparation_lock(cache), preparation_lock(root):
        pending = any(not (root / (s + "_preparation.json")).is_file() for s in selected)
        if pending and shutil.disk_usage(root).free < 20 * 1024**3:
            raise OSError("Allow at least 20 GiB free disk for archives, extraction and temporary arrays.")
        revision = args.revision
        revision_file = cache / "revision.json"
        if any(not getattr(args, s + "_archive") for s in selected):
            if revision is None:
                revision = json.loads(revision_file.read_text())["revision"] if revision_file.exists() else resolve_revision()
            if not re.fullmatch(r"[0-9a-f]{40}", revision):
                raise ValueError("Use an immutable 40-character dataset revision, not main.")
            cache.mkdir(parents=True, exist_ok=True)
            save_json(dict(repository=REPOSITORY, revision=revision), revision_file)
        for short in selected:
            supplied = getattr(args, short + "_archive")
            expected = getattr(args, short + "_sha256")
            if supplied:
                archive = Path(supplied).expanduser().resolve()
                if expected and sha256_file(archive) != expected:
                    raise ValueError("Offline archive SHA-256 mismatch: " + str(archive))
                source = dict(kind="user_supplied_archive", path=str(archive), external_hash_checked=bool(expected))
            else:
                url = "https://huggingface.co/datasets/{}/resolve/{}/datasets/{}?download=true".format(
                    REPOSITORY, revision, ARCHIVES[short])
                archive = download(url, cache / revision / ARCHIVES[short], expected)
                source = dict(kind="authors_repository_https", repository=REPOSITORY, revision=revision,
                              url=url, external_hash_checked=bool(expected))
            prepare_one(short, archive, root, source)
    print("Data preparation finished. No GPU experiments were started.\n"
          "Next: bash detector/run_advisor_imagenet.sh plan --seeds 6 7", flush=True)


if __name__ == "__main__":
    main()
