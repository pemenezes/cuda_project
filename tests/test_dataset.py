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
        "policy_id",
        "category",
        "system",
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


def test_canonical_policies_are_unique() -> None:
    policies = json.loads(
        (ROOT / "data" / "policies.json").read_text(encoding="utf-8")
    )
    policy_ids = [policy["id"] for policy in policies]
    assert len(policies) >= 20
    assert len(policy_ids) == len(set(policy_ids))
