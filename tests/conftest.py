import pytest
from datetime import datetime

from app import app, db, Task


@pytest.fixture
def test_app(tmp_path):
    db_path = tmp_path / "test.db"
    app.config.update(
        TESTING=True,
        SQLALCHEMY_DATABASE_URI=f"sqlite:///{db_path}",
        WTF_CSRF_ENABLED=False,
        MAIL_SUPPRESS_SEND=True,
    )
    with app.app_context():
        db.session.remove()
        db.drop_all()
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(test_app):
    return test_app.test_client()


@pytest.fixture
def app_context(test_app):
    with test_app.app_context():
        yield


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
            return task

    return _create
