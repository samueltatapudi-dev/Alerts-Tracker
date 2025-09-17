import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template, request, url_for, flash
from flask_mail import Mail, Message
from flask_sqlalchemy import SQLAlchemy
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'super-secret-key')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///tasks.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAIL_SERVER'] = os.environ.get('MAIL_SERVER', 'localhost')
app.config['MAIL_PORT'] = int(os.environ.get('MAIL_PORT', 25))
app.config['MAIL_USE_TLS'] = os.environ.get('MAIL_USE_TLS', 'false').lower() == 'true'
app.config['MAIL_USE_SSL'] = os.environ.get('MAIL_USE_SSL', 'false').lower() == 'true'
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD')
app.config['MAIL_SUPPRESS_SEND'] = os.environ.get('MAIL_SUPPRESS_SEND', 'false').lower() == 'true'
app.config['MAIL_DEFAULT_SENDER'] = os.environ.get('MAIL_DEFAULT_SENDER', 'no-reply@alerts.local')
app.config['APP_BASE_URL'] = os.environ.get('APP_BASE_URL')
app.config['PREFERRED_URL_SCHEME'] = os.environ.get('PREFERRED_URL_SCHEME', 'https')

STATUSES = ('Pending', 'In Progress', 'Completed')

db = SQLAlchemy(app)
mail = Mail(app)


class Task(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(120), nullable=False)
    description = db.Column(db.Text, nullable=False)
    assigned_email = db.Column(db.String(120), nullable=False, index=True)
    status = db.Column(db.String(32), nullable=False, default='Pending')
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    start_time = db.Column(db.DateTime, nullable=True)
    start_ip = db.Column(db.String(64), nullable=True)
    end_time = db.Column(db.DateTime, nullable=True)
    end_ip = db.Column(db.String(64), nullable=True)

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'assigned_email': self.assigned_email,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'start_ip': self.start_ip,
            'end_ip': self.end_ip,
            'user_url': build_user_url(self.assigned_email),
        }

    def __repr__(self) -> str:
        return f'<Task {self.id} {self.title}>'


def get_client_ip() -> str:
    header = request.headers.get('X-Forwarded-For', '')
    if header:
        return header.split(',')[0].strip()
    return request.remote_addr or 'unknown'


def build_user_url(email: str) -> str:
    normalized = email.strip().lower()
    base_url = app.config.get('APP_BASE_URL')
    if base_url:
        return f"{base_url.rstrip('/')}/user/{normalized}"
    try:
        return url_for('user_dashboard', email=normalized, _external=True)
    except RuntimeError:
        root = getattr(request, 'url_root', None)
        if root:
            return f"{root.rstrip('/')}/user/{normalized}"
    server_name = app.config.get('SERVER_NAME')
    if server_name:
        scheme = app.config.get('PREFERRED_URL_SCHEME', 'http')
        return f"{scheme}://{server_name.rstrip('/')}/user/{normalized}"
    return f"http://localhost:5000/user/{normalized}"


def send_task_email(task: Task) -> None:
    if not task.assigned_email:
        return
    link = build_user_url(task.assigned_email)
    subject = f"New Task Assigned: {task.title}"
    body = (
        f"Hello,\n\n"
        f"You have a new task assigned.\n"
        f"Title: {task.title}\n"
        f"Description: {task.description}\n\n"
        f"You can work on the task at {link}.\n\n"
        f"Best regards,\nAlerts & Tracker"
    )
    try:
        msg = Message(subject=subject, recipients=[task.assigned_email], body=body)
        mail.send(msg)
        app.logger.info('Queued task email to %s with link %s', task.assigned_email, link)
    except Exception as exc:  # pragma: no cover - log without failing request
        app.logger.warning('Email delivery failed for %s: %s', task.assigned_email, exc)


def _build_dataframe(tasks: List[Task]) -> pd.DataFrame:
    rows = []
    now = datetime.utcnow()
    for task in tasks:
        start_time = task.start_time
        end_time = task.end_time
        start_delay = None
        completion_time = None
        time_open = (end_time or now) - task.created_at
        if start_time:
            start_delay = (start_time - task.created_at).total_seconds() / 60.0
        if start_time and end_time:
            completion_time = (end_time - start_time).total_seconds() / 60.0
        rows.append({
            'task_id': task.id,
            'assigned_email': task.assigned_email,
            'status': task.status,
            'start_delay_minutes': start_delay,
            'completion_time_minutes': completion_time,
            'time_open_minutes': time_open.total_seconds() / 60.0,
            'has_started': 1 if start_time else 0,
            'is_completed': 1 if task.status == 'Completed' else 0,
        })
    if not rows:
        return pd.DataFrame(columns=['task_id', 'assigned_email', 'status', 'start_delay_minutes',
                                     'completion_time_minutes', 'time_open_minutes',
                                     'has_started', 'is_completed'])
    return pd.DataFrame(rows)


def _train_completion_model(df: pd.DataFrame) -> Optional[LogisticRegression]:
    if df.empty:
        return None
    if df['is_completed'].nunique() < 2:
        return None
    features = df[['start_delay_minutes', 'time_open_minutes', 'has_started']].fillna(0.0)
    labels = df['is_completed']
    try:
        model = LogisticRegression(max_iter=1000)
        model.fit(features, labels)
        return model
    except Exception as exc:  # pragma: no cover - defensive guard
        app.logger.warning('Logistic regression training failed: %s', exc)
        return None


def _cluster_users(user_feature_rows: List[Dict[str, Any]]) -> Dict[str, str]:
    if not user_feature_rows:
        return {}
    data = np.array([
        [row['completion_rate'], row['avg_completion_time'] or 0.0,
         row['avg_start_delay'] or 0.0, row['open_tasks']]
        for row in user_feature_rows
    ])
    n_clusters = min(3, len(user_feature_rows))
    if n_clusters < 1:
        return {}
    try:
        kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
        assignments = kmeans.fit_predict(data)
    except Exception as exc:  # pragma: no cover - fallback if clustering fails
        app.logger.warning('KMeans clustering failed: %s', exc)
        return {row['assigned_email']: 'Average' for row in user_feature_rows}

    centers = kmeans.cluster_centers_
    ordered_centers = sorted(range(len(centers)), key=lambda idx: centers[idx][0], reverse=True)
    labels = ['Highly Engaged', 'Average', 'Needs Attention']
    cluster_label_map = {}
    for rank, cluster_idx in enumerate(ordered_centers):
        label = labels[rank] if rank < len(labels) else f'Segment {rank + 1}'
        cluster_label_map[cluster_idx] = label
    return {row['assigned_email']: cluster_label_map.get(assignments[idx], 'Average')
            for idx, row in enumerate(user_feature_rows)}


def compute_admin_insights(tasks: List[Task]) -> Dict[str, object]:
    df = _build_dataframe(tasks)
    metrics = {
        'user_metrics': [],
        'overall_completion_rate': 0.0,
        'average_completion_time': 0.0,
        'at_risk_users': [],
    }
    if df.empty:
        return metrics

    completion_rate_overall = df['is_completed'].mean()
    metrics['overall_completion_rate'] = round(float(completion_rate_overall), 3)

    completed = df[df['status'] == 'Completed']
    metrics['average_completion_time'] = round(float(completed['completion_time_minutes'].mean()) if not completed.empty else 0.0, 2)

    user_groups = df.groupby('assigned_email')
    feature_rows = []
    risk_scores = {}

    completion_model = _train_completion_model(df)
    now = datetime.utcnow()
    for email, group in user_groups:
        completion_rate = group['is_completed'].mean()
        avg_completion = group['completion_time_minutes'].mean() if 'completion_time_minutes' in group else None
        avg_start_delay = group['start_delay_minutes'].mean() if 'start_delay_minutes' in group else None
        open_tasks = int((group['status'] != 'Completed').sum())
        feature_rows.append({
            'assigned_email': email,
            'completion_rate': float(completion_rate) if not np.isnan(completion_rate) else 0.0,
            'avg_completion_time': float(avg_completion) if avg_completion is not None and not np.isnan(avg_completion) else None,
            'avg_start_delay': float(avg_start_delay) if avg_start_delay is not None and not np.isnan(avg_start_delay) else None,
            'open_tasks': open_tasks,
        })

        risk_score = 0.0
        if completion_model is not None:
            open_group = group[group['status'] != 'Completed']
            if not open_group.empty:
                features = open_group[['start_delay_minutes', 'time_open_minutes', 'has_started']].fillna(0.0)
                probabilities = completion_model.predict_proba(features)[:, 1]
                risk_score = float(1.0 - probabilities.mean())
        if risk_score == 0.0 and open_tasks:
            oldest_open = group[group['status'] != 'Completed']['time_open_minutes'].max()
            if not np.isnan(oldest_open) and oldest_open > 12 * 60:
                risk_score = min(1.0, oldest_open / (24 * 60))
        risk_scores[email] = round(risk_score, 3)

    clusters = _cluster_users(feature_rows)

    for row in feature_rows:
        email = row['assigned_email']
        user_info = {
            'assigned_email': email,
            'completion_rate': round(row['completion_rate'], 3),
            'avg_completion_time': round(row['avg_completion_time'], 2) if row['avg_completion_time'] is not None else None,
            'avg_start_delay': round(row['avg_start_delay'], 2) if row['avg_start_delay'] is not None else None,
            'open_tasks': row['open_tasks'],
            'segment': clusters.get(email, 'Average'),
            'risk_score': risk_scores.get(email, 0.0),
        }
        user_info['at_risk'] = user_info['risk_score'] >= 0.6
        metrics['user_metrics'].append(user_info)

    metrics['user_metrics'].sort(key=lambda item: item['risk_score'], reverse=True)
    metrics['at_risk_users'] = [item['assigned_email'] for item in metrics['user_metrics'] if item['at_risk']]
    return metrics


def _summarise_user(tasks: List[Task]) -> Dict[str, object]:
    if not tasks:
        return {
            'average_completion_minutes': None,
            'tasks_completed': 0,
            'tasks_pending': 0,
            'suggestion': 'No tasks yet. Once you start working, personalised tips will appear.',
        }
    df = _build_dataframe(tasks)
    completed = df[df['status'] == 'Completed']
    avg_completion = float(completed['completion_time_minutes'].mean()) if not completed.empty else None
    pending_count = int((df['status'] != 'Completed').sum())
    completed_count = int((df['status'] == 'Completed').sum())
    suggestion = generate_user_tip(avg_completion, pending_count, completed_count, df)
    return {
        'average_completion_minutes': round(avg_completion, 2) if avg_completion is not None else None,
        'tasks_completed': completed_count,
        'tasks_pending': pending_count,
        'suggestion': suggestion,
    }


def generate_user_tip(avg_completion: Optional[float], pending: int, completed: int, df: pd.DataFrame) -> str:
    if avg_completion is None and completed == 0:
        return 'Once you complete a task, we will track your pace and share tailored suggestions.'
    avg_start_delay = df['start_delay_minutes'].mean()
    if pending > completed and pending >= 3:
        return 'You have several pending tasks. Try breaking them into smaller chunks to build momentum.'
    if avg_start_delay and avg_start_delay > 120:
        return 'You tend to start tasks a bit late. Starting within two hours can improve completion rates.'
    if avg_completion and avg_completion > 240:
        return 'Your tasks take longer than average. Consider blocking focused time to finish faster.'
    if pending == 0:
        return 'Great job staying on top of your tasks! Keep maintaining this pace.'
    return 'Nice progress. Keep reviewing priorities daily to maintain steady completion.'


@app.route('/')
def home():
    return redirect(url_for('admin_dashboard'))


@app.route('/admin', methods=['GET'])
def admin_dashboard():
    tasks = Task.query.order_by(Task.created_at.desc()).all()
    insights = compute_admin_insights(tasks)
    return render_template('admin_dashboard.html', tasks=tasks, insights=insights)


@app.route('/admin/create_task', methods=['POST'])
def create_task():
    title = request.form.get('title', '').strip()
    description = request.form.get('description', '').strip()
    assigned_email = request.form.get('assigned_email', '').strip().lower()

    if not title or not description or not assigned_email:
        flash('All fields are required.', 'danger')
        return redirect(url_for('admin_dashboard'))

    task = Task(title=title, description=description, assigned_email=assigned_email, status='Pending')
    db.session.add(task)
    db.session.commit()
    send_task_email(task)
    flash('Task created successfully.', 'success')
    return redirect(url_for('admin_dashboard'))


@app.route('/user/<path:email>', methods=['GET'])
def user_dashboard(email: str):
    normalized = email.strip().lower()
    return render_template('user_dashboard.html', email=normalized)


@app.route('/api/tasks', methods=['GET'])
def api_tasks():
    tasks = Task.query.order_by(Task.created_at.desc()).all()
    insights = compute_admin_insights(tasks)
    return jsonify({
        'tasks': [task.to_dict() for task in tasks],
        'insights': insights,
    })


@app.route('/api/user/<path:email>/tasks', methods=['GET'])
def api_user_tasks(email: str):
    normalized = email.strip().lower()
    tasks = Task.query.filter_by(assigned_email=normalized).order_by(Task.created_at.desc()).all()
    summary = _summarise_user(tasks)
    return jsonify({
        'tasks': [task.to_dict() for task in tasks],
        'summary': summary,
    })


@app.route('/api/tasks/<int:task_id>/start', methods=['POST'])
def api_start_task(task_id: int):
    task = Task.query.get_or_404(task_id)
    if not task.start_time:
        task.start_time = datetime.utcnow()
        task.start_ip = get_client_ip()
    task.status = 'In Progress'
    db.session.commit()
    return jsonify({'message': 'Task started', 'task': task.to_dict()})


@app.route('/api/tasks/<int:task_id>/complete', methods=['POST'])
def api_complete_task(task_id: int):
    task = Task.query.get_or_404(task_id)
    task.status = 'Completed'
    if not task.start_time:
        task.start_time = datetime.utcnow()
        task.start_ip = task.start_ip or get_client_ip()
    task.end_time = datetime.utcnow()
    task.end_ip = get_client_ip()
    db.session.commit()
    return jsonify({'message': 'Task completed', 'task': task.to_dict()})


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
