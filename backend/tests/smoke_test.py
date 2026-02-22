from __future__ import annotations

from pathlib import Path
import sys

from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.main import app


def main() -> None:
    student_csv = REPO_ROOT / "students/student_1.csv"
    if not student_csv.exists():
        raise FileNotFoundError(f"Missing test CSV: {student_csv}")

    client = TestClient(app)

    # Success case.
    with student_csv.open("rb") as handle:
        response = client.post(
            "/predict",
            data={"student_id": "1"},
            files={"file": ("student_1.csv", handle, "text/csv")},
        )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert "probabilities" in payload and isinstance(payload["probabilities"], list), payload
    assert len(payload["probabilities"]) > 0, payload
    print("PASS: success case")

    # Validation failure case (missing required columns).
    bad_csv = b"foo,bar\n1,2\n"
    response = client.post(
        "/predict",
        data={"student_id": "1"},
        files={"file": ("bad.csv", bad_csv, "text/csv")},
    )
    assert response.status_code == 400, response.text
    print("PASS: missing-columns validation case")

    # Not-found student case.
    with student_csv.open("rb") as handle:
        response = client.post(
            "/predict",
            data={"student_id": "9999"},
            files={"file": ("student_1.csv", handle, "text/csv")},
        )
    assert response.status_code == 404, response.text
    print("PASS: unknown-student case")

    print("PASS: all backend smoke tests completed.")


if __name__ == "__main__":
    main()

