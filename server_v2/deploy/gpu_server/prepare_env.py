"""기존 인증·모델 경로를 보존하며 v2 전용 경로만 설정한다. 비밀 값은 출력하지 않는다."""
import argparse
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("baseline", type=Path)
    p.add_argument("target", type=Path)
    args = p.parse_args()
    old, new = args.baseline.resolve(), args.target.resolve()
    src = old / "deploy/gpu_server/.env"
    dst = new / "deploy/gpu_server/.env"
    if not src.is_file():
        raise SystemExit("Baseline .env missing")
    revision = json.loads((new / "indexing/snapshot.json").read_text())["revision"]
    changes = {"APP_ROOT": str(new), "CHROMA_DIR": str(new / "indexing/chroma_db"),
               "QA_LOG_PATH": str(new / "orchestrator/logs/qa_log.jsonl"),
               "V2_TRACE_PATH": str(new / "rag_service/logs/trace.jsonl"),
               "INDEX_REVISION": revision}
    # 재전환 때 운영자가 정한 V2_VALIDATION_MODE 등은 유지한다.
    lines = (dst if dst.exists() else src).read_text(encoding="utf-8").splitlines()
    kept = [line for line in lines if line.partition("=")[0].strip() not in changes]
    if not any(line.startswith("V2_VALIDATION_MODE=") for line in kept):
        kept.append("V2_VALIDATION_MODE=log")
    kept += [f"{key}={value}" for key, value in changes.items()]
    tmp = dst.with_suffix(".pending")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write("\n".join(kept) + "\n")
    os.chmod(tmp, 0o600)
    tmp.replace(dst)
    print("v2 environment prepared (secrets omitted)")


if __name__ == "__main__":
    main()
