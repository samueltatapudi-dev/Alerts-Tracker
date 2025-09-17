const formatDateTime = (isoString) => {
  if (!isoString) return '—';
  const date = new Date(isoString);
  if (Number.isNaN(date.getTime())) return '—';
  return date.toLocaleString();
};

const fetchJSON = async (url, options = {}) => {
  const response = await fetch(url, options);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || 'Request failed');
  }
  return response.json();
};

const renderAdminTasks = (tasks) => {
  const table = document.getElementById('adminTasksTable');
  if (!table) return;
  const tbody = table.querySelector('tbody');
  if (!tasks.length) {
    tbody.innerHTML = '<tr><td colspan="7" class="text-center text-muted">No tasks created yet.</td></tr>';
    return;
  }
  const rows = tasks.map((task) => `
    <tr>
      <td>${task.title}</td>
      <td>${task.assigned_email}</td>
      <td><span class="badge text-bg-primary">${task.status}</span></td>
      <td>${formatDateTime(task.start_time)}</td>
      <td>${task.start_ip || '—'}</td>
      <td>${formatDateTime(task.end_time)}</td>
      <td>${task.end_ip || '—'}</td>
    </tr>
  `);
  tbody.innerHTML = rows.join('');
};

const renderAdminInsights = (insights) => {
  if (!insights) return;
  const completionRateEl = document.getElementById('insightCompletionRate');
  const completionTimeEl = document.getElementById('insightCompletionTime');
  const riskUsersEl = document.getElementById('insightRiskUsers');
  const usersTable = document.getElementById('adminUsersTable');

  if (completionRateEl) {
    const rate = insights.overall_completion_rate ? (insights.overall_completion_rate * 100).toFixed(1) : '0.0';
    completionRateEl.textContent = `${rate}%`;
  }
  if (completionTimeEl) {
    completionTimeEl.textContent = insights.average_completion_time ? `${insights.average_completion_time.toFixed(2)} mins` : 'N/A';
  }
  if (riskUsersEl) {
    riskUsersEl.textContent = insights.at_risk_users && insights.at_risk_users.length ? insights.at_risk_users.join(', ') : 'None';
  }
  if (usersTable) {
    const tbody = usersTable.querySelector('tbody');
    if (!insights.user_metrics || !insights.user_metrics.length) {
      tbody.innerHTML = '<tr><td colspan="7" class="text-center text-muted">No data yet.</td></tr>';
      return;
    }
    const rows = insights.user_metrics.map((row) => {
      const riskPercent = (row.risk_score * 100).toFixed(1);
      const classes = row.at_risk ? 'risk-high' : '';
      const completionPercent = (row.completion_rate * 100).toFixed(1);
      const avgCompletion = row.avg_completion_time != null ? row.avg_completion_time.toFixed(2) : '—';
      const avgDelay = row.avg_start_delay != null ? row.avg_start_delay.toFixed(2) : '—';
      return `
        <tr class="${classes}">
          <td>${row.assigned_email}</td>
          <td>${completionPercent}%</td>
          <td>${avgCompletion}</td>
          <td>${avgDelay}</td>
          <td>${row.open_tasks}</td>
          <td><span class="badge text-bg-secondary badge-segment">${row.segment}</span></td>
          <td>${riskPercent}%</td>
        </tr>
      `;
    });
    tbody.innerHTML = rows.join('');
  }
};

const initAdminDashboard = () => {
  const poll = async () => {
    try {
      const data = await fetchJSON('/api/tasks');
      renderAdminTasks(data.tasks);
      renderAdminInsights(data.insights);
    } catch (error) {
      console.error('Failed to refresh admin dashboard', error);
    } finally {
      window.adminDashboardTimer = window.setTimeout(poll, 10000);
    }
  };

  if (window.adminDashboardTimer) {
    window.clearTimeout(window.adminDashboardTimer);
  }
  poll();
};

const renderUserTasks = (table, tasks, knownTaskIds, audioEl, options = {}) => {
  const tbody = table.querySelector('tbody');
  const suppressAudio = options.suppressAudio ?? false;
  if (!tasks.length) {
    tbody.innerHTML = '<tr><td colspan="4" class="text-center text-muted">No tasks assigned yet.</td></tr>';
    return;
  }
  let hasNewTask = false;
  const rows = tasks.map((task) => {
    if (!knownTaskIds.has(task.id)) {
      hasNewTask = true;
      knownTaskIds.add(task.id);
    }
    const canStart = task.status === 'Pending';
    const canComplete = task.status === 'In Progress';
    const buttons = `
      <div class="d-flex gap-2">
        <button class="btn btn-sm btn-outline-primary" data-action="start" data-task="${task.id}" ${canStart ? '' : 'disabled'}>Start</button>
        <button class="btn btn-sm btn-success" data-action="complete" data-task="${task.id}" ${canComplete ? '' : 'disabled'}>Complete</button>
      </div>
    `;
    return `
      <tr data-task-id="${task.id}">
        <td>${task.title}</td>
        <td>${task.description}</td>
        <td><span class="badge text-bg-primary">${task.status}</span></td>
        <td>${buttons}</td>
      </tr>
    `;
  });
  tbody.innerHTML = rows.join('');
  if (hasNewTask && audioEl && !suppressAudio) {
    audioEl.currentTime = 0;
    audioEl.play().catch((error) => console.warn('Audio playback blocked', error));
  }
};

const renderUserSummary = (summary) => {
  if (!summary) return;
  const avgEl = document.getElementById('summaryAvgCompletion');
  const pendingEl = document.getElementById('summaryPending');
  const completedEl = document.getElementById('summaryCompleted');
  const suggestionEl = document.getElementById('summarySuggestion');
  if (avgEl) {
    avgEl.textContent = summary.average_completion_minutes != null ? `${summary.average_completion_minutes.toFixed(2)} mins` : '—';
  }
  if (pendingEl) {
    pendingEl.textContent = summary.tasks_pending ?? 0;
  }
  if (completedEl) {
    completedEl.textContent = summary.tasks_completed ?? 0;
  }
  if (suggestionEl) {
    suggestionEl.textContent = summary.suggestion || 'Keep up the good work!';
  }
};

const initUserDashboard = (email) => {
  const table = document.getElementById('userTasksTable');
  if (!table) return;
  const audioEl = document.getElementById('newTaskAudio');
  const knownTaskIds = new Set();

  const handleAction = async (taskId, action) => {
    try {
      await fetchJSON(`/api/tasks/${taskId}/${action}`, { method: 'POST' });
      await poll();
    } catch (error) {
      console.error(`Failed to ${action} task`, error);
      alert('Something went wrong. Please try again.');
    }
  };

  table.addEventListener('click', (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) return;
    const action = target.dataset.action;
    const taskId = target.dataset.task;
    if (action && taskId) {
      handleAction(taskId, action);
    }
  });

  let isInitialRender = true;

  const poll = async () => {
    try {
      const data = await fetchJSON(`/api/user/${encodeURIComponent(email)}/tasks`);
      renderUserTasks(table, data.tasks, knownTaskIds, audioEl, { suppressAudio: isInitialRender });
      renderUserSummary(data.summary);
      isInitialRender = false;
    } catch (error) {
      console.error('Failed to refresh user dashboard', error);
    } finally {
      window.userDashboardTimer = window.setTimeout(poll, 7000);
    }
  };

  if (window.userDashboardTimer) {
    window.clearTimeout(window.userDashboardTimer);
  }
  poll();
};

window.initAdminDashboard = initAdminDashboard;
window.initUserDashboard = initUserDashboard;
