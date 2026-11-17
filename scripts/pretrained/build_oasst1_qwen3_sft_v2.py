from pathlib import Path
import json
import hashlib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "pretrained/qwen3_0.6b/data/oasst1_qwen3_sft_v1"
OUTPUT = ROOT / "pretrained/qwen3_0.6b/data/oasst1_qwen3_sft_v2"

STOP_ID = 151645
NEWLINE_ID = 198
IGNORE_INDEX = -100

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def main():
    if not SOURCE.is_dir():
        raise FileNotFoundError(SOURCE)

    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite: {OUTPUT}")

    manifest_path = SOURCE / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if manifest.get("assistant_end_token_id") != STOP_ID:
        raise RuntimeError("Unexpected STOP token in V1 manifest")

    OUTPUT.mkdir(parents=True)

    total = 0
    masked = 0

    try:
        for split in ("train", "validation", "test"):
            x_path = SOURCE / f"{split}_input_ids.npy"
            y_path = SOURCE / f"{split}_labels.npy"

            x = np.load(x_path, mmap_mode="r")
            y = np.load(y_path)

            if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 512:
                raise RuntimeError(f"Invalid array shape: {split}")

            labels = y.copy()
            split_masked = 0

            for i in range(len(labels)):
                stops = np.flatnonzero(labels[i] == STOP_ID)

                if len(stops) != 1:
                    raise RuntimeError(
                        f"{split}[{i}]: expected exactly one STOP"
                    )

                stop = int(stops[0])

                if stop + 1 >= labels.shape[1]:
                    raise RuntimeError(
                        f"{split}[{i}]: STOP has no following position"
                    )

                if int(x[i, stop + 1]) != NEWLINE_ID:
                    raise RuntimeError(
                        f"{split}[{i}]: expected newline after STOP"
                    )

                if int(labels[i, stop + 1]) != NEWLINE_ID:
                    raise RuntimeError(
                        f"{split}[{i}]: unexpected label after STOP"
                    )

                labels[i, stop + 1] = IGNORE_INDEX
                split_masked += 1

                supervised = np.flatnonzero(labels[i] != IGNORE_INDEX)

                if not len(supervised) or int(supervised[-1]) != stop:
                    raise RuntimeError(
                        f"{split}[{i}]: STOP is not final supervised token"
                    )

            np.save(OUTPUT / f"{split}_input_ids.npy", np.asarray(x))
            np.save(OUTPUT / f"{split}_labels.npy", labels)

            total += len(labels)
            masked += split_masked

            print(
                f"{split}: examples={len(labels)}, "
                f"newline_labels_masked={split_masked}"
            )

        v2 = dict(manifest)
        v2["dataset_version"] = "v2"
        v2["parent_dataset"] = "oasst1_qwen3_sft_v1"
        v2["parent_manifest_sha256"] = sha256(manifest_path)
        v2["loss_masking"] = (
            "assistant final message only; trailing newline after "
            "assistant end token ignored"
        )
        v2["post_stop_newline_token_id"] = NEWLINE_ID
        v2["post_stop_newline_label"] = IGNORE_INDEX
        v2["v2_transformation"] = (
            "Copied V1 arrays; masked exactly one newline label "
            "immediately following each assistant end token"
        )

        (OUTPUT / "manifest.json").write_text(
            json.dumps(v2, indent=2),
            encoding="utf-8",
        )

        print("\n=== V2 CREATED AND VERIFIED ===")
        print("Output:", OUTPUT)
        print("Total examples:", total)
        print("Newline labels masked:", masked)
        print("Original V1 remains unchanged.")

    except Exception:
        import shutil
        shutil.rmtree(OUTPUT)
        raise

if __name__ == "__main__":
    main()
