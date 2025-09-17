from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from app import parse_email_list, _build_dataframe, _summarise_user, compute_admin_insights


def _make_task_namespace(**overrides):
    now = overrides.pop("now", datetime.utcnow())
    created_at = overrides.pop("created_at", now - timedelta(hours=1))
    task = SimpleNamespace(
        id=overrides.pop("id", 1),
        title=overrides.pop("title", "Sample"),
        description=overrides.pop("description", "Task"),
        assigned_email=overrides.pop("assigned_email", "user@example.com"),
        status=overrides.pop("status", "Pending"),
        created_at=created_at,
        start_time=overrides.pop("start_time", None),
        start_ip=overrides.pop("start_ip", None),
        end_time=overrides.pop("end_time", None),
        end_ip=overrides.pop("end_ip", None),
    )
    if overrides:
        raise TypeError(f"Unexpected overrides: {', '.join(overrides.keys())}")
    return task


def test_parse_email_list_deduplicates_and_normalises():
    raw = "Alice@example.com; bob@example.com, alice@example.com\ncarol@example.com"
    result = parse_email_list(raw)
    assert result == ["alice@example.com", "bob@example.com", "carol@example.com"]


def test_build_dataframe_computes_expected_fields():
    now = datetime.utcnow()
    completed_task = _make_task_namespace(
        id=1,
        status="Completed",
        assigned_email="alice@example.com",
        created_at=now - timedelta(hours=3),
        start_time=now - timedelta(hours=2, minutes=30),
        end_time=now - timedelta(hours=2),
    )
    pending_task = _make_task_namespace(
        id=2,
        status="Pending",
        assigned_email="bob@example.com",
        created_at=now - timedelta(hours=1),
    )

    df = _build_dataframe([completed_task, pending_task])
    assert set(df.columns) == {
        "task_id",
        "assigned_email",
        "status",
        "start_delay_minutes",
        "completion_time_minutes",
        "time_open_minutes",
        "has_started",
        "is_completed",
    }
    completed_row = df.loc[df["task_id"] == 1].iloc[0]
    assert pytest.approx(completed_row["start_delay_minutes"], rel=1e-3) == 30.0
    assert pytest.approx(completed_row["completion_time_minutes"], rel=1e-3) == 30.0
    assert completed_row["has_started"] == 1
    pending_row = df.loc[df["task_id"] == 2].iloc[0]
    assert pending_row["has_started"] == 0
    assert pending_row["is_completed"] == 0


def test_summarise_user_returns_counts_and_tip():
    now = datetime.utcnow()
    completed = _make_task_namespace(
        id=1,
        status="Completed",
        assigned_email="user@example.com",
        created_at=now - timedelta(hours=5),
        start_time=now - timedelta(hours=4, minutes=45),
        end_time=now - timedelta(hours=4, minutes=15),
    )
    pending = _make_task_namespace(
        id=2,
        status="Pending",
        assigned_email="user@example.com",
        created_at=now - timedelta(hours=2),
    )

    summary = _summarise_user([completed, pending])
    assert summary["tasks_completed"] == 1
    assert summary["tasks_pending"] == 1
    assert summary["average_completion_minutes"] == pytest.approx(30.0, rel=1e-3)
    assert "Nice progress" in summary["suggestion"]


def test_compute_admin_insights_aggregates_metrics():
    now = datetime.utcnow()
    completed = _make_task_namespace(
        id=1,
        assigned_email="alice@example.com",
        status="Completed",
        created_at=now - timedelta(hours=6),
        start_time=now - timedelta(hours=5, minutes=30),
        end_time=now - timedelta(hours=4, minutes=30),
    )
    pending_same_user = _make_task_namespace(
        id=2,
        assigned_email="alice@example.com",
        status="Pending",
        created_at=now - timedelta(hours=3),
    )
    in_progress_other = _make_task_namespace(
        id=3,
        assigned_email="bob@example.com",
        status="In Progress",
        created_at=now - timedelta(hours=2),
        start_time=now - timedelta(hours=1, minutes=30),
    )

    insights = compute_admin_insights([completed, pending_same_user, in_progress_other])
    assert insights["overall_completion_rate"] == pytest.approx(1 / 3, rel=1e-3)
    assert insights["average_completion_time"] == pytest.approx(60.0, rel=1e-3)
    assert {metric["assigned_email"] for metric in insights["user_metrics"]} == {
        "alice@example.com",
        "bob@example.com",
    }
    for metric in insights["user_metrics"]:
        assert "risk_score" in metric
        assert metric["segment"] in {"Highly Engaged", "Average", "Needs Attention"} or metric["segment"].startswith("Segment ")
