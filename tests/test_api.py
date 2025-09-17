from datetime import datetime, timedelta

from app import Task, EmailSettings


def test_health_check(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_admin_dashboard_requires_auth(client):
    response = client.get("/admin")
    assert response.status_code == 401
    assert 'WWW-Authenticate' in response.headers


def test_admin_dashboard_with_auth(client, admin_headers):
    response = client.get("/admin", headers=admin_headers)
    assert response.status_code == 200


def test_api_tasks_requires_auth(client):
    response = client.get("/api/tasks")
    assert response.status_code == 401


def test_api_tasks_returns_tasks_and_insights(client, test_app, make_task, admin_headers):
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

    response = client.get("/api/tasks", headers=admin_headers)
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
    assert all("user_token" in task for task in payload["tasks"])


def test_api_user_tasks_requires_token(client, make_task):
    make_task(assigned_email="user@example.com")
    response = client.get("/api/user/user@example.com/tasks")
    assert response.status_code == 401


def test_api_user_tasks_filters_by_email(client, make_task):
    now = datetime.utcnow()
    user_task = make_task(assigned_email="user@example.com", created_at=now - timedelta(hours=1))
    make_task(assigned_email="other@example.com", created_at=now - timedelta(hours=2))

    response = client.get(
        f"/api/user/user@example.com/tasks?token={user_task.user_token}"
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert len(payload["tasks"]) == 1
    assert payload["tasks"][0]["assigned_email"] == "user@example.com"
    assert payload["summary"]["tasks_pending"] == 1


def test_api_start_task_requires_token(client, make_task):
    task = make_task(status="Pending")
    response = client.post(f"/api/tasks/{task.id}/start")
    assert response.status_code == 401


def test_api_start_task_transitions_status(client, test_app, make_task):
    task = make_task(status="Pending")

    response = client.post(
        f"/api/tasks/{task.id}/start",
        json={"token": task.user_token},
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["task"]["status"] == "In Progress"
    assert payload["task"]["start_time"] is not None

    with test_app.app_context():
        persisted = Task.query.get(task.id)
        assert persisted.status == "In Progress"
        assert persisted.start_time is not None
        assert persisted.start_ip == "127.0.0.1"


def test_api_complete_task_requires_token(client, make_task):
    task = make_task(status="Pending")
    response = client.post(f"/api/tasks/{task.id}/complete")
    assert response.status_code == 401


def test_api_complete_task_marks_finished(client, test_app, make_task):
    task = make_task(status="Pending")

    response = client.post(
        f"/api/tasks/{task.id}/complete",
        json={"token": task.user_token},
    )
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


def test_user_dashboard_requires_valid_token(client):
    response = client.get("/user/test@example.com")
    assert response.status_code == 403


def test_user_dashboard_allows_valid_token(client, make_task):
    task = make_task(assigned_email="user@example.com")
    response = client.get(f"/user/user@example.com?token={task.user_token}")
    assert response.status_code == 200

def test_update_email_settings_requires_auth(client):
    response = client.post('/admin/email_settings', data={'mail_server': 'smtp.example.com'})
    assert response.status_code == 401


def test_update_email_settings_persists(client, admin_headers):
    response = client.post(
        '/admin/email_settings',
        data={
            'mail_server': 'smtp.example.com',
            'mail_port': '2525',
            'mail_use_tls': 'on',
            'mail_username': 'smtp-user',
            'mail_password': 'smtp-pass',
            'mail_default_sender': 'Alerts Bot <alerts@example.com>',
        },
        headers=admin_headers,
        follow_redirects=False,
    )
    assert response.status_code == 302
    with client.application.app_context():
        settings = EmailSettings.query.get(1)
        assert settings.mail_server == 'smtp.example.com'
        assert settings.mail_port == 2525
        assert settings.use_tls is True
        assert settings.use_ssl is False
        assert settings.username == 'smtp-user'
        assert settings.password == 'smtp-pass'
        assert settings.default_sender == 'Alerts Bot <alerts@example.com>'


def test_test_email_settings_flow(client, admin_headers):
    response = client.post(
        '/admin/email_settings/test',
        data={'test_recipient': 'user@example.com'},
        headers=admin_headers,
        follow_redirects=False,
    )
    assert response.status_code == 302


def test_admin_email_task_sends_link(client, make_task, admin_headers):
    task = make_task(assigned_email='user@example.com')
    response = client.post(f'/admin/tasks/{task.id}/email', headers=admin_headers, follow_redirects=False)
    assert response.status_code == 302
