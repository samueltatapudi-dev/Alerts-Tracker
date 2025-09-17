import base64
import os
from datetime import datetime

import pytest

os.environ.setdefault('SECRET_KEY', 'test-secret')
os.environ.setdefault('ADMIN_USERNAME', 'admin')
os.environ.setdefault('ADMIN_PASSWORD', 'password')

from app import app, db, Task, generate_user_token, invalidate_insights_cache  # noqa: E402


def _basic_auth_header(username: str, password: str) -> dict:
    token = base64.b64encode(f"{username}:{password}".encode('utf-8')).decode('utf-8')
    return {'Authorization': f'Basic {token}'}


@pytest.fixture
def test_app(tmp_path):
    db_path = tmp_path / "test.db"
    app.config.update(
        TESTING=True,
        SQLALCHEMY_DATABASE_URI=f"sqlite:///{db_path}",
        WTF_CSRF_ENABLED=False,
        MAIL_SUPPRESS_SEND=True,
        ADMIN_INSIGHTS_CACHE_SECONDS=0,
    )
    with app.app_context():
        db.session.remove()
        db.drop_all()
        db.create_all()
        invalidate_insights_cache()
        yield app
        db.session.remove()
        db.drop_all()
        invalidate_insights_cache()


@pytest.fixture
def client(test_app):
    return test_app.test_client()


@pytest.fixture
def app_context(test_app):
    with test_app.app_context():
        yield


@pytest.fixture
def admin_headers(test_app):
    return _basic_auth_header(test_app.config['ADMIN_USERNAME'], test_app.config['ADMIN_PASSWORD'])


@pytest.fixture
def make_task(test_app):
    def _create(**kwargs):
        now = kwargs.pop("created_at", datetime.utcnow())
        task = Task(
            title=kwargs.pop("title", "Sample Task"),
            description=kwargs.pop("description", "Do something"),
            assigned_email=kwargs.pop("assigned_email", "user@example.com"),
            status=kwargs.pop("status", "Pending"),
            created_at=now,
            start_time=kwargs.pop("start_time", None),
            end_time=kwargs.pop("end_time", None),
            start_ip=kwargs.pop("start_ip", None),
            end_ip=kwargs.pop("end_ip", None),
        )
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {', '.join(kwargs.keys())}")
        with test_app.app_context():
            db.session.add(task)
            db.session.commit()
            db.session.refresh(task)
            invalidate_insights_cache()
            return task

    return _create


@pytest.fixture
def user_token():
    def _token(email: str) -> str:
        return generate_user_token(email)
    return _token
