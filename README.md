# Alerts & Tracker

Alerts & Tracker is a Flask-based productivity assistant that lets admins assign work, track execution in real time, and give users personalised nudges powered by lightweight analytics.

- **Tech stack:** Flask, SQLAlchemy, pandas, scikit-learn, Bootstrap, vanilla JS.
- **Data store:** SQLite (`instance/tasks.db` by default).
- **Email:** Flask-Mail; configurable SMTP provider.

## Prerequisites

- Python 3.10+ (3.13 tested)
- `pip` and `virtualenv`
- An SMTP server or debug listener (for outbound email)
- Optional: speakers/headphones (user dashboard plays audio reminders)

## 1. Set Up & Run

```bash
# 1. Clone or copy the project, then from the repo root:
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env  # edit values as needed

# 3. Launch the app
python app.py
```

The server listens on http://127.0.0.1:5000 with `debug=True`. The first start creates `instance/tasks.db` and the new `user_group` tables automatically.

### SMTP Quick Options

| Scenario | Settings |
| --- | --- |
| Local development (no real sends) | `python3 -m smtpd -c DebuggingServer -n localhost:1025` in another terminal, then set `MAIL_SERVER=localhost`, `MAIL_PORT=1025`, `MAIL_SUPPRESS_SEND=false` |
| Gmail with 2FA | Use an App Password, set `MAIL_USE_TLS=true`, `MAIL_PORT=587` |
| Provider requiring SSL | Set `MAIL_USE_SSL=true`, `MAIL_USE_TLS=false`, typically `MAIL_PORT=465` |

Edit `.env` (auto-loaded via `python-dotenv`) with:

```
SECRET_KEY=replace-me
MAIL_SERVER=smtp.yourprovider.com
MAIL_PORT=587
MAIL_USE_TLS=true
MAIL_USERNAME=alerts@example.com
MAIL_PASSWORD=your-app-password
MAIL_DEFAULT_SENDER=Alerts Bot <alerts@example.com>
MAIL_SUPPRESS_SEND=false
APP_BASE_URL=https://demo.yourdomain.com
PREFERRED_URL_SCHEME=https
```

> Tip: restart `python app.py` whenever `.env` changes.

## 2. Admin Workflow

1. **Create groups** (left column → _Create User Group_).
   - Paste multiple emails (comma, semicolon, or new line separated).
   - Saved groups are listed with member emails and a quick **Use** button.
2. **Assign tasks** (left column → _Create Task_).
   - Enter a title & description.
   - Add ad-hoc recipients in the textarea **and/or** highlight one or more groups in the multiselect (hold Cmd/Ctrl for multi-select). Click **Clear** to deselect all.
   - Submit: the app fans out _one task per user_, emails each assignee, and shows the new entries in **Current Tasks**.
3. **Share user dashboards**.
   - Every task row exposes the personalised URL plus a **Copy Link** button.
4. **Monitor insights**.
   - The “Completion Rate”, “Avg Completion Time”, and “AI Insights” cards auto-refresh every 10 seconds.
   - The “User Engagement Snapshot” sorts users by risk score (logistic regression + heuristics) and flags at-risk users.

## 3. User Experience

- Users visit `APP_BASE_URL/user/<email>` (link is included in the notification email and the admin Share column).
- Dashboards auto-refresh every 7 seconds.
- Buttons: **Start** moves tasks to “In Progress” and records IP/time; **Complete** closes them and saves end IP/time.
- Audio cues:
  - A chime plays when a brand-new task arrives.
  - As long as unfinished tasks remain, a reminder sound repeats every 5 seconds. Finish all tasks to silence it (browsers may require a user interaction to enable audio).
- The “Performance Summary” panel surfaces average completion time, counts, and adaptive AI suggestions.

## 4. API Surface

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/api/health` | Simple uptime probe (`{"status": "ok"}`) |
| `GET` | `/api/tasks` | All tasks + aggregated insights for admins |
| `GET` | `/api/user/<email>/tasks` | Tasks for a specific user + summary data |
| `POST` | `/api/tasks/<id>/start` | Mark a task “In Progress”, capture start IP/time |
| `POST` | `/api/tasks/<id>/complete` | Mark a task “Completed”, capture end IP/time |

Responses are JSON; all task payloads include `user_url` for direct linking.

## 5. Development Notes

- Database lives under `instance/`; delete `instance/tasks.db` for a clean slate (recreated on launch).
- Templates: `templates/admin_dashboard.html`, `templates/user_dashboard.html`
- Front-end logic: `static/js/app.js`
- Static audio/css: `static/audio/alert.wav`, `static/css/`
- ML helpers live in `app.py` (`compute_admin_insights`, `_cluster_users`, etc.).
- No migrations are wired up; if you change models, either rebuild the SQLite file or add Alembic.

## 6. Troubleshooting & FAQs

- **No emails?** Ensure SMTP credentials are correct and check the Flask log for “Email delivery failed …”. Some providers require enabling less-secure apps or app passwords.
- **Links wrong?** Set `APP_BASE_URL` to the public host (e.g. tunnel URL) so generated links and emails resolve correctly.
- **Audio silent?** Most browsers block autoplay until the user interacts with the page. Ask users to click once to unblock.
- **Group changes missing?** Restart the server after introducing the grouping feature so the new tables are created (`db.create_all()` runs on boot).
- **Reset demo data?** Delete `instance/tasks.db` and restart; all tables repopulate from scratch.

## 7. Next Ideas

- Add user authentication / RBAC around admin endpoints.
- Replace SQLite with Postgres + Alembic migrations for production.
- Integrate transactional email provider webhooks (SendGrid, SES) to track bounces.
- Extend reminder settings (per-task cadence, snooze) or hook in SMS using Twilio.

Happy hacking!
