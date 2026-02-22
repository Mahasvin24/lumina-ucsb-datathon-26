from __future__ import annotations

import csv
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Attempt:
    question_id: int
    response: int
    timestamp: int
    concept_ids: list[str]


@dataclass
class StudentData:
    student_id: int
    user_id: int
    attempts: list[Attempt]


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _parse_concept_ids(raw: str) -> list[str]:
    value = (raw or "").strip()
    if not value:
        return []
    if "|" in value:
        return [part.strip() for part in value.split("|") if part.strip()]
    if "," in value:
        return [part.strip() for part in value.split(",") if part.strip()]
    return [value]


def _load_test_question_ids(students_dir: Path) -> list[int]:
    path = students_dir / "test_questions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing test question file: {path}")

    question_ids: list[int] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "question_id" not in (reader.fieldnames or []):
            raise ValueError(f"test_questions.csv is missing required column 'question_id': {path}")
        for row in reader:
            qid = _safe_int(row.get("question_id"))
            if qid > 0:
                question_ids.append(qid)

    if not question_ids:
        raise ValueError(f"No valid question_id rows found in: {path}")
    return sorted(set(question_ids))


def _load_students(students_dir: Path) -> list[StudentData]:
    student_files = sorted(students_dir.glob("student_*.csv"))
    if not student_files:
        raise FileNotFoundError(f"No student CSV files found in: {students_dir}")

    students: list[StudentData] = []
    for idx, csv_path in enumerate(student_files, start=1):
        attempts: list[Attempt] = []
        user_id = 0

        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"question_id", "response", "timestamp", "concept_ids", "user_id"}
            missing = sorted(required - set(reader.fieldnames or []))
            if missing:
                raise ValueError(f"{csv_path.name} missing required columns: {missing}")

            for row in reader:
                user_id = _safe_int(row.get("user_id"), user_id)
                attempts.append(
                    Attempt(
                        question_id=_safe_int(row.get("question_id")),
                        response=1 if _safe_int(row.get("response")) == 1 else 0,
                        timestamp=_safe_int(row.get("timestamp")),
                        concept_ids=_parse_concept_ids(str(row.get("concept_ids", ""))),
                    )
                )

        students.append(StudentData(student_id=idx, user_id=user_id, attempts=attempts))

    return students


def _latest_test_attempts(students: list[StudentData], test_question_ids: set[int]) -> dict[int, dict[int, Attempt]]:
    # For each student and test question, keep the latest attempt by timestamp.
    latest: dict[int, dict[int, Attempt]] = {}
    for student in students:
        per_question: dict[int, Attempt] = {}
        for attempt in student.attempts:
            if attempt.question_id not in test_question_ids:
                continue
            prev = per_question.get(attempt.question_id)
            if prev is None or attempt.timestamp >= prev.timestamp:
                per_question[attempt.question_id] = attempt
        latest[student.student_id] = per_question
    return latest


def _pct(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return round((numerator / denominator) * 100.0, 2)


def _median_time_minutes(attempts: list[Attempt]) -> float:
    if not attempts:
        return 0.0
    # We don't have per-question duration, so estimate with spacing between attempts.
    timestamps = sorted(a.timestamp for a in attempts if a.timestamp > 0)
    if len(timestamps) < 2:
        return 0.0
    deltas = [(b - a) / 1000.0 for a, b in zip(timestamps, timestamps[1:]) if b >= a]
    if not deltas:
        return 0.0
    return round(statistics.median(deltas) / 60.0, 2)


def build_dashboard_data(
    repo_root: Path,
    student_threshold: float = 70.0,
    question_threshold: float = 60.0,
    exclude_unanswered: bool = False,
    top_n_hardest: int = 15,
) -> dict[str, Any]:
    students_dir = repo_root / "students"
    students = _load_students(students_dir)
    test_qids = _load_test_question_ids(students_dir)
    test_qid_set = set(test_qids)
    latest = _latest_test_attempts(students, test_qid_set)
    total_questions = len(test_qids)

    student_rows: list[dict[str, Any]] = []
    student_scores: list[float] = []
    flagged_students: list[int] = []

    per_question_correct: dict[int, int] = {qid: 0 for qid in test_qids}
    per_question_attempts: dict[int, int] = {qid: 0 for qid in test_qids}
    per_question_concepts: dict[int, set[str]] = {qid: set() for qid in test_qids}
    per_question_wrong_students: dict[int, list[int]] = {qid: [] for qid in test_qids}

    matrix_rows: list[dict[str, Any]] = []

    for student in students:
        per_q = latest.get(student.student_id, {})
        answered = len(per_q)
        correct = sum(1 for attempt in per_q.values() if attempt.response == 1)
        score_denominator = answered if exclude_unanswered else total_questions
        score_pct = _pct(correct, score_denominator)
        completion_pct = _pct(answered, total_questions)
        status = "Complete" if answered == total_questions else "Incomplete"

        attempts_sorted = sorted(per_q.values(), key=lambda item: item.timestamp)
        if attempts_sorted and attempts_sorted[0].timestamp > 0 and attempts_sorted[-1].timestamp > 0:
            total_minutes = round(
                max(0, attempts_sorted[-1].timestamp - attempts_sorted[0].timestamp) / 60000.0, 2
            )
        else:
            total_minutes = 0.0

        concept_miss_counts: dict[str, int] = {}
        for qid, attempt in per_q.items():
            per_question_attempts[qid] += 1
            per_question_correct[qid] += attempt.response
            per_question_concepts[qid].update(attempt.concept_ids)
            if attempt.response == 0:
                per_question_wrong_students[qid].append(student.student_id)
                for concept in attempt.concept_ids:
                    concept_miss_counts[concept] = concept_miss_counts.get(concept, 0) + 1

        most_missed_concept = "None"
        if concept_miss_counts:
            most_missed_concept = max(concept_miss_counts, key=concept_miss_counts.get)

        if score_pct < student_threshold:
            flagged_students.append(student.student_id)

        student_scores.append(score_pct)
        student_rows.append(
            {
                "studentId": student.student_id,
                "name": f"Student {student.student_id}",
                "userId": student.user_id,
                "scorePct": score_pct,
                "correctCount": correct,
                "totalQuestions": total_questions,
                "answeredCount": answered,
                "completionPct": completion_pct,
                "status": status,
                "timeSpentMinutes": total_minutes,
                "medianSpacingMinutes": _median_time_minutes(attempts_sorted),
                "mostMissedConcept": most_missed_concept,
            }
        )

        matrix_cells = []
        for qid in test_qids:
            attempt = per_q.get(qid)
            matrix_cells.append(
                {
                    "questionId": qid,
                    "state": "unanswered" if attempt is None else ("correct" if attempt.response == 1 else "wrong"),
                }
            )
        matrix_rows.append({"studentId": student.student_id, "name": f"Student {student.student_id}", "cells": matrix_cells})

    question_rows: list[dict[str, Any]] = []
    for qid in test_qids:
        attempts = per_question_attempts[qid]
        correct = per_question_correct[qid]
        accuracy = _pct(correct, attempts)
        flagged = accuracy < question_threshold
        question_rows.append(
            {
                "questionId": qid,
                "classCorrectPct": accuracy,
                "correctCount": correct,
                "attempts": attempts,
                "skillTags": sorted(per_question_concepts[qid]),
                "flagged": flagged,
            }
        )

    question_rows.sort(key=lambda item: item["classCorrectPct"])
    hardest = question_rows[0] if question_rows else None
    easiest = question_rows[-1] if question_rows else None

    class_average = round(statistics.mean(student_scores), 2) if student_scores else 0.0
    median_score = round(statistics.median(student_scores), 2) if student_scores else 0.0
    class_completion = round(statistics.mean([row["completionPct"] for row in student_rows]), 2) if student_rows else 0.0

    hardest_ids = [row["questionId"] for row in question_rows[: max(1, top_n_hardest)]]
    low_accuracy_ids = [row["questionId"] for row in question_rows if row["classCorrectPct"] < question_threshold]
    missed_by_flagged: set[int] = set()
    for qid in test_qids:
        wrong_students = per_question_wrong_students[qid]
        if any(student_id in flagged_students for student_id in wrong_students):
            missed_by_flagged.add(qid)

    remediation_ids = sorted(set(low_accuracy_ids) | set(hardest_ids) | missed_by_flagged)
    remediation_question_rows = [row for row in question_rows if row["questionId"] in remediation_ids]
    remediation_skills = sorted(
        {
            skill
            for row in remediation_question_rows
            for skill in row["skillTags"]
        }
    )

    summary = {
        "classAveragePct": class_average,
        "medianScorePct": median_score,
        "studentsBelowThresholdPct": _pct(len(flagged_students), len(students)),
        "hardestQuestion": (
            {"questionId": hardest["questionId"], "classCorrectPct": hardest["classCorrectPct"]}
            if hardest
            else None
        ),
        "easiestQuestion": (
            {"questionId": easiest["questionId"], "classCorrectPct": easiest["classCorrectPct"]}
            if easiest
            else None
        ),
        "completionRatePct": class_completion,
    }

    return {
        "settings": {
            "studentThresholdPct": round(student_threshold, 2),
            "questionThresholdPct": round(question_threshold, 2),
            "excludeUnanswered": bool(exclude_unanswered),
            "topNHardest": max(1, int(top_n_hardest)),
            "totalStudents": len(students),
            "totalQuestions": total_questions,
        },
        "summary": summary,
        "students": sorted(student_rows, key=lambda item: item["scorePct"]),
        "questions": question_rows,
        "matrix": {
            "questionIds": test_qids,
            "rows": matrix_rows,
        },
        "remediation": {
            "flaggedStudentIds": sorted(flagged_students),
            "selectedQuestionIds": remediation_ids,
            "questionCount": len(remediation_ids),
            "estimatedTimeMinutes": int(math.ceil(len(remediation_ids) * 2.0)),
            "skillCoverage": remediation_skills,
            "previewQuestions": remediation_question_rows,
        },
    }
