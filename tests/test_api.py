from datetime import datetime, timedelta

from app import Task


def test_health_check(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_api_tasks_returns_tasks_and_insights(client, test_app, make_task):
    now = datetime.utcnow()
    make_task(
        assigned_email="alice@example.com",
        status="Completed",
        created_at=now - timedelta(hours=4),
        start_time=now - timedelta(hours=3, minutes=30),
        end_time=now - timedelta(hours=3),
    )
    make_task(
        assigned_email="alice@example.com",
        status="Pending",
        created_at=now - timedelta(hours=2),
    )
    make_task(
        assigned_email="bob@example.com",
        status="Pending",
        created_at=now - timedelta(hours=1),
    )

    response = client.get("/api/tasks")
    assert response.status_code == 200
    payload = response.get_json()
    assert len(payload["tasks"]) == 3
    assert set(payload["insights"].keys()) == {
        "user_metrics",
        "overall_completion_rate",
        "average_completion_time",
        "at_risk_users",
    }
    assert {task["assigned_email"] for task in payload["tasks"]} == {
        "alice@example.com",
        "bob@example.com",
    }


def test_api_user_tasks_filters_by_email(client, make_task):
    now = datetime.utcnow()
    make_task(assigned_email="user@example.com", created_at=now - timedelta(hours=1))
    make_task(assigned_email="other@example.com", created_at=now - timedelta(hours=2))

    response = client.get("/api/user/user@example.com/tasks")
    assert response.status_code == 200
    payload = response.get_json()
    assert len(payload["tasks"]) == 1
    assert payload["tasks"][0]["assigned_email"] == "user@example.com"
    assert payload["summary"]["tasks_pending"] == 1


def test_api_start_task_transitions_status(client, test_app, make_task):
    task = make_task(status="Pending")

    response = client.post(f"/api/tasks/{task.id}/start")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["task"]["status"] == "In Progress"
    assert payload["task"]["start_time"] is not None

    with test_app.app_context():
        persisted = Task.query.get(task.id)
        assert persisted.status == "In Progress"
        assert persisted.start_time is not None
        assert persisted.start_ip == "127.0.0.1"


def test_api_complete_task_marks_finished(client, test_app, make_task):
    task = make_task(status="Pending")

    response = client.post(f"/api/tasks/{task.id}/complete")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["task"]["status"] == "Completed"
    assert payload["task"]["end_time"] is not None

    with test_app.app_context():
        persisted = Task.query.get(task.id)
        assert persisted.status == "Completed"
        assert persisted.end_time is not None
        assert persisted.end_ip == "127.0.0.1"
        assert persisted.start_time is not None
        assert persisted.start_ip == "127.0.0.1"
