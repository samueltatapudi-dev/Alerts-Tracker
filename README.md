# Alerts & Tracker

A Flask dashboard for assigning work, tracking progress, and nudging users with ML-powered insights.

## Quick Start

1. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` (or create `.env`) and fill in your mail settings (see below).
4. Launch the server:
   ```bash
   python app.py
   ```
5. Visit http://127.0.0.1:5000 to open the admin dashboard. Task links now use the configured base URL.

### Working with Groups

- Use the **Create User Group** card on the admin dashboard to name a cohort and paste multiple user emails (comma/new-line separated).
- When creating a task, paste multiple emails directly or highlight one or more groups in the multiselect (or click **Use** beside a group)—one task per user is generated automatically and each person receives their own email.

## Email Configuration

The app sends emails via Flask-Mail. Provide SMTP credentials in environment variables or a `.env` file:

```
MAIL_SERVER=smtp.yourprovider.com
MAIL_PORT=587
MAIL_USE_TLS=true
MAIL_USERNAME=someone@example.com
MAIL_PASSWORD=your-app-password
MAIL_DEFAULT_SENDER=Alerts Bot <someone@example.com>
MAIL_SUPPRESS_SEND=false
APP_BASE_URL=https://your-demo-host
```

Notes:
- `APP_BASE_URL` is optional but recommended for correct deep links in emails and the API.
- For TLS (port 587) set `MAIL_USE_TLS=true`. For SSL (port 465) set `MAIL_USE_SSL=true` and leave TLS false.
- Gmail requires an app password when two-factor authentication is enabled.
- To test locally without sending real messages, run `python3 -m smtpd -c DebuggingServer -n localhost:1025` in another terminal, then use `MAIL_SERVER=localhost`, `MAIL_PORT=1025`, and keep `MAIL_SUPPRESS_SEND=false`.

`python-dotenv` now loads `.env` automatically on startup, so restarting the server after edits is enough.

## Features

- Admin dashboard to create tasks, monitor assignments, and copy shareable user URLs.
- Group management so you can re-use cohorts of users when creating tasks.
- Auto-refreshing insights summarizing completion rates, risk levels, and ML clustering.
- User dashboard with live updates, start/complete actions, and audio alerts for new tasks.
- REST API (`/api/tasks`, `/api/user/<email>/tasks`) powering the UI and available for integrations.
- Health check at `/api/health` for deployment probes.

## Troubleshooting

- If emails do not arrive, check application logs for "Email delivery failed" warnings.
- Verify that firewall rules allow outbound SMTP traffic from your environment.
- When testing in staging environments, set `APP_BASE_URL` to the public URL so links in emails open the correct page.
- When you add new user groups, restart the app once so SQLAlchemy can create the tables (or delete `instance/tasks.db` to start fresh in dev).
- Delete `instance/tasks.db` if you want a fresh database (the schema recreates itself on startup).
