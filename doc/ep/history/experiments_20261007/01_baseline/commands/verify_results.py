"""Verify downloaded baseline data against the server SHA256 manifest."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2] / "results" / "01_baseline"
    rows = json.loads((root / "R07_audit" / "manifest.json").read_text())
    failures = []
    for row in rows:
        path = root / row["path"]
        if not path.is_file():
            failures.append(f"Missing: {row['path']}")
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if path.stat().st_size != row["bytes"] or digest.hexdigest() != row["sha256"]:
            failures.append(f"Mismatch: {row['path']}")
    if failures:
        raise SystemExit("\n".join(failures))
    result = {"status": "passed", "files": len(rows),
              "bytes": sum(row["bytes"] for row in rows)}
    (root / "R07_audit" / "verification.json").write_text(
        json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
