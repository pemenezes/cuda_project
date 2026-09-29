import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_dataset_generation_and_schema() -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "src" / "prepare_data.py")],
        check=True,
        cwd=ROOT,
    )
    expected_fields = {
        "id",
        "knowledge_id",
        "category",
        "system",
        "history",
        "question",
        "answer",
    }
    ids: set[str] = set()
    for split in ("train", "validation", "test"):
        rows = read_jsonl(ROOT / "data" / "processed" / f"{split}.jsonl")
        assert rows
        for row in rows:
            assert set(row) == expected_fields
            assert row["id"] not in ids
            ids.add(row["id"])
            assert row["question"].strip()
            assert row["answer"].strip()
            for message in row["history"]:
                assert message["role"] in {"user", "assistant"}
                assert message["content"].strip()


def test_canonical_knowledge_is_unique() -> None:
    knowledge = json.loads(
        (ROOT / "data" / "barista_knowledge.json").read_text(encoding="utf-8")
    )
    knowledge_ids = [item["id"] for item in knowledge]
    assert len(knowledge) >= 30
    assert len(knowledge_ids) == len(set(knowledge_ids))
