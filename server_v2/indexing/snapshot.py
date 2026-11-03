"""기존 인덱스를 일관된 상태로 복사한다. 모델 호출·재인덱싱은 하지 않는다.

반드시 rag_service 중지 후 switch.sh가 실행한다. 원본 DB에는 쓰지 않는다.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--target", type=Path, required=True)
    args = ap.parse_args()
    old, new = args.baseline.resolve(), args.target.resolve()
    if old == new or old in new.parents:
        raise SystemExit("V2 root must be separate from baseline root")
    for unit in ("rag_service", "orchestrator"):
        state = subprocess.check_output(["systemctl", "show", unit, "-p", "ActiveState", "--value"], text=True).strip()
        if state not in ("inactive", "failed"):
            raise SystemExit(f"Stop {unit} before copying the DB")
    root = new / "indexing"
    manifest = root / "snapshot.json"
    if manifest.exists():
        data = json.loads(manifest.read_text())
        if not (root / "chroma_db").is_dir():
            raise SystemExit("Snapshot manifest exists but DB is missing")
        print(data["revision"])
        return
    chunks = old / "indexing/output_chunks/chunks.jsonl"
    db = old / "indexing/chroma_db"
    if not chunks.is_file() or not db.is_dir():
        raise SystemExit("Baseline chunks/Chroma DB missing")
    # 중간 실패 흔적은 자동으로 삭제하거나 기존 스냅샷에 덮어쓰지 않는다.
    for dest in (root / "chroma_db", root / "output_chunks"):
        if dest.exists():
            raise SystemExit(f"Incomplete snapshot; inspect manually: {dest}")
    revision = "legacy-" + hashlib.sha256(chunks.read_bytes()).hexdigest()
    shutil.copytree(db, root / "chroma_db")
    (root / "output_chunks").mkdir()
    shutil.copy2(chunks, root / "output_chunks/chunks.jsonl")
    if (old / "indexing/output").is_dir() and not (root / "output").exists():
        shutil.copytree(old / "indexing/output", root / "output")
    manifest.write_text(json.dumps({"revision": revision, "baseline": str(old),
        "schema": "legacy-chroma-snapshot-v1", "chunks_sha256": revision[7:]}, indent=2), encoding="utf-8")
    print(revision)


if __name__ == "__main__":
    main()
