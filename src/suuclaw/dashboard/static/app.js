import { ArchitectureView } from './architecture.js';

/* SuuClaw workbench. Every state below is derived from a persisted run or event. */
const $ = (id) => document.getElementById(id);
const $$ = (query, root = document) => [...root.querySelectorAll(query)];
const escapeHTML = (value) => String(value ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const icon = (name, className = '') => `<svg class="icon ${className}" aria-hidden="true"><use href="#i-${name}"/></svg>`;
const TERMINAL = new Set(['completed', 'failed', 'interrupted', 'cancelled']);
const THEMES = { tide: { label: '潮汐', color: '#edf4f7' }, abyss: { label: '深航', color: '#0c1d32' }, bloom: { label: '花间', color: '#f4eff7' } };
const STATUS = { idle: '待运行', running: '进行中', completed: '已完成', failed: '执行失败', interrupted: '已中断', waiting: '等待确认', delegated: '等待专家', rejected: '验证未通过', ended: '已结束', pending: '待开始', in_progress: '进行中', blocked: '受阻' };
const state = {
  perspective: 'architecture', graphMode: 'overview', pathSteps: [], transitions: new Set(), lastGraphNode: '__start__', transfer: null,
  bootstrap: null, sessions: [], session: null, run: null, events: [], seq: 0,
  nodes: new Map(), tools: new Map(), approvals: new Map(), todos: [], checks: [], results: [],
  planSummary: '', outcome: null, activeNode: '', selectedNode: '', lastModelMessage: '',
  currentView: 'workflow', tab: 'events', filter: 'all', stream: null, reconnectTimer: null,
  generation: 0, renderPending: false, pending: false, files: [], fileGeneration: 0,
  connected: false, initializing: true, drafts: new Map(), pendingApprovals: new Set(), autoScroll: true,
};
const welcomeHTML = $('chat-messages').innerHTML;
const readPreference = (key, fallback = '') => { try { return localStorage.getItem(key) ?? fallback; } catch { return fallback; } };
const savePreference = (key, value) => { try { localStorage.setItem(key, value); } catch { /* Private browsing may deny storage. */ } };
const json = (value) => { try { return JSON.stringify(value, null, 2); } catch { return String(value); } };
const clip = (value, length = 130) => { const text = String(value ?? '').replace(/\s+/g, ' ').trim(); return text.length > length ? text.slice(0, length) + '…' : text; };
const nodeID = (id) => ({ searchAgent: 'search', codeAgent: 'code', search_agent: 'search', code_agent: 'code' }[id] || id || '');
const nodeInfo = (id) => state.bootstrap?.topology?.nodes?.find((node) => node.id === nodeID(id));
const nodeLabel = (id) => nodeInfo(id)?.label || id || '工作流';
const AI_TITLES = { __start__: 'Input', intent_router: 'Intent Router', chat_responder: 'Chat Responder', planner: 'Supervisor', context_monitor: 'Context Monitor', context_compressor: 'Context Compaction', verifier: 'Verifier', final: 'Final Response', __end__: 'End', search: 'Search Agent', code: 'Code Agent' };
const nodeTitle = (id) => nodeInfo(id)?.title || AI_TITLES[nodeID(id)] || id || 'Workflow';
const graphMotion = { queue: [], next: null, expiry: null, edges: new Map(), nodes: new Map() };
const reduceMotion = () => matchMedia('(prefers-reduced-motion: reduce)').matches;
const nodeIcon = (node) => ({ boundary: 'chat', router: 'flow', agent: 'chat', supervisor: 'spark', context: 'memory', verifier: 'shield', output: 'check', specialist: node.id === 'search' ? 'search' : 'code' }[node.kind] || 'cpu');
const isRunning = () => !!state.run && !TERMINAL.has(state.run.status);
const currentSessionRunning = () => !!state.session?.active_run_id || isRunning();
const displayTime = (value, full = false) => {
  const date = new Date(value);
  if (!Number.isFinite(date.getTime())) return '';
  return new Intl.DateTimeFormat('zh-CN', { timeZone: 'Asia/Shanghai', ...(full ? { month: 'short', day: 'numeric' } : {}), hour: '2-digit', minute: '2-digit', hour12: false }).format(date);
};
const duration = (ms) => {
  if (!Number.isFinite(Number(ms))) return '—';
  const seconds = Math.max(0, Number(ms)) / 1000;
  if (seconds < 1) return `${Math.round(seconds * 1000)} 毫秒`;
  if (seconds < 60) return `${seconds < 10 ? seconds.toFixed(1) : Math.round(seconds)} 秒`;
  return `${Math.floor(seconds / 60)} 分 ${Math.floor(seconds % 60)} 秒`;
};
const bytes = (value) => value < 1024 ? `${value} B` : value < 1048576 ? `${(value / 1024).toFixed(1)} KB` : `${(value / 1048576).toFixed(1)} MB`;

async function api(path, options = {}) {
  const response = await fetch(path, { cache: 'no-store', ...options, headers: { ...(options.body ? { 'Content-Type': 'application/json' } : {}), ...options.headers } });
  let result;
  try { result = await response.json(); } catch { throw new Error(`服务返回了无法读取的内容（${response.status}）。`); }
  if (!response.ok) {
    const error = new Error(result?.error?.message || `请求未完成（${response.status}）。`);
    error.code = result?.error?.code;
    error.status = response.status;
    throw error;
  }
  return result;
}
function readableError(error) {
  if (['model_not_configured', 'model_unconfigured', 'missing_api_key'].includes(error.code)) return '尚未配置模型。请在本地 .env 中填写模型信息，然后重启工作台。';
  if (error.code === 'run_in_progress' || error.code === 'session_busy') return '这段对话的任务还在进行，请等待它结束后再发送。';
  if (error.code === 'message_too_long') return '内容超过 16,000 个字符，请拆分成更短的任务。';
  if (error.code === 'server_busy') return '当前可用的任务槽位已满，稍后再试。';
  if (error.code === 'approval_closed') return '这次确认已经结束，请查看最新的执行动态。';
  if (error instanceof TypeError) return '无法连接本地服务，请确认 SuuClaw 工作台仍在运行。';
  return error.message || '操作未完成，请重试。';
}
function toast(message) {
  const element = document.createElement('div');
  element.className = 'toast';
  element.textContent = message;
  $('toast-region').append(element);
  setTimeout(() => element.remove(), 5000);
}
function announce(message) { $('screen-announcement').textContent = message; }
function setConnection(status, label) {
  $('connection-status').className = `connection-status ${status}`;
  $('connection-status').lastElementChild.textContent = label;
}
function setHTML(element, html) {
  if (element._content === html) return;
  const opened = new Set($$('details[open][data-detail]', element).map((item) => item.dataset.detail));
  element.innerHTML = html;
  element._content = html;
  $$('details[data-detail]', element).forEach((item) => { if (opened.has(item.dataset.detail)) item.open = true; });
}
function emptyState(name, title, description) {
  return `<div class="empty-state">${icon(name)}<strong>${escapeHTML(title)}</strong><p>${escapeHTML(description)}</p></div>`;
}
function disclosure(title, value, key) {
  return `<details class="detail-disclosure" data-detail="${escapeHTML(key)}"><summary>${icon('chevron')}${escapeHTML(title)}</summary><pre>${escapeHTML(typeof value === 'string' ? value : json(value))}</pre></details>`;
}

// Deliberately small Markdown renderer. All source text is escaped; raw HTML is never used.
function inlineMarkdown(text) {
  const tokens = [];
  let escaped = escapeHTML(text).replace(/`([^`]+)`/g, (_, code) => { tokens.push(`<code>${code}</code>`); return `\u0000${tokens.length - 1}\u0000`; });
  escaped = escaped.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, (_, label, href) => `<a href="${href}" target="_blank" rel="noopener noreferrer">${label}</a>`);
  escaped = escaped.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  return escaped.replace(/\u0000(\d+)\u0000/g, (_, index) => tokens[Number(index)]);
}
function markdown(content) {
  const lines = String(content || '').replace(/\r\n/g, '\n').split('\n');
  let html = '', paragraph = [], list = [], listType = '', code = null, language = '';
  const flushParagraph = () => { if (paragraph.length) { html += `<p>${paragraph.map(inlineMarkdown).join('<br>')}</p>`; paragraph = []; } };
  const flushList = () => { if (list.length) { html += `<${listType}>${list.map((item) => `<li>${inlineMarkdown(item)}</li>`).join('')}</${listType}>`; list = []; listType = ''; } };
  const flushCode = () => { html += `<div class="code-block"><header><span>${escapeHTML(language || '代码')}</span><button class="message-copy" data-action="copy-code" type="button">${icon('copy')}复制</button></header><pre><code>${escapeHTML(code.join('\n'))}</code></pre></div>`; code = null; };
  for (const line of lines) {
    if (/^\s*```/.test(line)) {
      flushParagraph(); flushList();
      if (code !== null) flushCode(); else { code = []; language = line.replace(/^\s*```/, '').trim(); }
      continue;
    }
    if (code !== null) { code.push(line); continue; }
    if (!line.trim()) { flushParagraph(); flushList(); continue; }
    const heading = line.match(/^#{1,6}\s+(.+)$/);
    if (heading) { flushParagraph(); flushList(); html += `<h3>${inlineMarkdown(heading[1])}</h3>`; continue; }
    const bullet = line.match(/^\s*(?:([-*])|\d+[.)])\s+(.+)$/);
    if (bullet) { flushParagraph(); const type = bullet[1] ? 'ul' : 'ol'; if (listType && listType !== type) flushList(); listType = type; list.push(bullet[2]); continue; }
    flushList(); paragraph.push(line);
  }
  flushParagraph(); flushList(); if (code !== null) flushCode();
  return html;
}

function applyTheme(theme) {
  if (!THEMES[theme]) theme = 'tide';
  document.documentElement.dataset.theme = theme;
  $('theme-name').textContent = THEMES[theme].label;
  document.querySelector('meta[name="theme-color"]').content = THEMES[theme].color;
  $$('[data-theme-choice]').forEach((button) => button.setAttribute('aria-pressed', String(button.dataset.themeChoice === theme)));
  savePreference('suuclaw.theme', theme);
}
function setView(view) {
  if (!['workflow', 'tools', 'files', 'history'].includes(view)) return;
  closeExpandedGraphs();
  state.currentView = view;
  $$('.view').forEach((element) => { element.hidden = element.id !== `view-${view}`; element.classList.toggle('active', !element.hidden); });
  $$('[data-view]').forEach((button) => { const active = button.dataset.view === view; button.classList.toggle('active', active); if (active) button.setAttribute('aria-current', 'page'); else button.removeAttribute('aria-current'); });
  $('page-title').textContent = { workflow: '工作流', tools: '工具箱', files: '工作文件', history: '执行记录' }[view];
  document.title = `SuuClaw · ${$('page-title').textContent}`;
  $('main-scroll').scrollTop = 0;
  closeSidebar();
  if (view === 'files') loadFiles();
  if (view === 'tools') renderTools();
  if (view === 'history') renderHistory();
}
function setChat(open) {
  $('app-shell').classList.toggle('chat-hidden', !open);
  $('chat-toggle').setAttribute('aria-expanded', String(open));
  if (open) $('message-input').focus({ preventScroll: true });
  else $('chat-toggle').focus({ preventScroll: true });
  if (matchMedia('(min-width: 901px)').matches) savePreference('suuclaw.chat-open', String(open));
}
function closeSidebar() {
  $('sidebar').classList.remove('open'); $('sidebar-scrim').hidden = true; $('menu-toggle').setAttribute('aria-expanded', 'false');
}
function resetRun(run = null) {
  closeStream(); clearGraphMotion();
  state.pathSteps = []; state.transitions = new Set(); state.lastGraphNode = '__start__'; state.transfer = null;
  state.run = run; state.events = []; state.seq = 0; state.nodes = new Map(); state.tools = new Map(); state.approvals = new Map();
  state.todos = []; state.checks = []; state.results = []; state.planSummary = ''; state.outcome = null; state.activeNode = ''; state.selectedNode = ''; state.lastModelMessage = '';
  $('inspector-panel').hidden = true;
}
function closeStream() {
  if (state.stream) state.stream.close();
  state.stream = null;
  if (state.reconnectTimer) clearTimeout(state.reconnectTimer);
  state.reconnectTimer = null;
}
function saveDraft() { state.drafts.set(state.session?.id || 'new', $('message-input').value); }
function restoreDraft() { $('message-input').value = state.drafts.get(state.session?.id || 'new') || ''; resizeComposer(); updateComposer(); }
async function refreshSessions() {
  const response = await api('/api/sessions');
  state.sessions = response.sessions || [];
  const selected = state.sessions.find((session) => session.id === state.session?.id);
  if (selected && state.session) {
    state.session.title = selected.title;
    state.session.active_run_id = selected.active_run_id;
    updateComposer();
  }
  renderSessions();
}
function renderSessions() {
  setHTML($('sessions-list'), state.sessions.length ? state.sessions.map((session) => `<button class="session-item ${state.session?.id === session.id ? 'selected' : ''}" data-session="${escapeHTML(session.id)}" title="${escapeHTML(session.title || '新对话')}" ${state.session?.id === session.id ? 'aria-current="true"' : ''}>${icon('chat')}<span class="session-item-title">${escapeHTML(session.title || '新对话')}</span>${session.active_run_id ? '<i class="session-dot" aria-label="正在执行"></i>' : ''}</button>`).join('') : '<p class="sidebar-empty">还没有对话。<br>从一个小想法开始吧。</p>');
}
async function selectSession(id) {
  saveDraft(); closeStream();
  const generation = ++state.generation;
  $('composer-error').hidden = true;
  try {
    const session = await api(`/api/sessions/${encodeURIComponent(id)}`);
    if (generation !== state.generation) return;
    state.session = session;
    resetRun();
    savePreference('suuclaw.session', id);
    restoreDraft(); renderSessions(); renderRunSelect(); renderHistory(); renderChat(); scheduleRender();
    $('file-preview').hidden = true; state.files = [];
    closeSidebar();
    const selected = session.active_run_id || session.runs?.at(-1)?.id;
    if (selected) await selectRun(selected, generation);
    if (state.currentView === 'files') loadFiles();
  } catch (error) { if (generation === state.generation) toast(readableError(error)); }
}
async function newSession() {
  if (state.initializing || state.newSessionPending || state.pending) return;
  state.newSessionPending = true;
  saveDraft();
  try {
    const session = await api('/api/sessions', { method: 'POST', body: '{}' });
    await refreshSessions();
    await selectSession(session.id);
    setView('workflow'); setChat(true); toast('新对话已准备好');
  } catch (error) { toast(readableError(error)); }
  finally { state.newSessionPending = false; }
}
async function selectRun(id, generation = state.generation) {
  closeStream();
  const sessionId = state.session?.id;
  const selection = Symbol(id);
  state.runSelection = selection;
  try {
    const run = await api(`/api/runs/${encodeURIComponent(id)}`);
    if (generation !== state.generation || sessionId !== state.session?.id || state.runSelection !== selection) return;
    const persistedStatus = run.status;
    resetRun(run);
    for (const event of run.events || []) applyEvent(event, true);
    // Event append precedes metadata finalization. A terminal frame is the newer
    // authority when this response caught that small persistence window.
    const terminalEvent = state.events.findLast((event) => event.type === 'run_end');
    state.run.status = terminalEvent?.status || persistedStatus;
    if (TERMINAL.has(state.run.status) && state.session?.active_run_id === id) state.session.active_run_id = null;
    renderRunSelect(); scheduleRender();
    if (!TERMINAL.has(run.status)) connectStream(id, generation);
    else setConnection('connected', '本地已连接');
  } catch (error) { if (generation === state.generation) toast(readableError(error)); }
}
function renderRunSelect() {
  const runs = state.session?.runs || [];
  const items = [...runs];
  if (state.run && !items.some((run) => run.id === state.run.id)) items.push(state.run);
  setHTML($('run-select'), items.length ? [...items].reverse().map((run, index) => `<option value="${escapeHTML(run.id)}">${index === 0 ? '最近一轮' : displayTime(run.created_at, true)} · ${STATUS[run.status] || run.status || '执行'}</option>`).join('') : '<option value="">尚无执行记录</option>');
  $('run-select').value = state.run?.id || '';
  $('run-select').disabled = !items.length;
}
async function sendMessage(event) {
  event.preventDefault();
  const message = $('message-input').value.trim();
  if (!message || state.initializing || state.pending) return;
  if (!state.bootstrap?.configured) { showSetup(); return; }
  if (currentSessionRunning()) { toast('这段对话仍有任务在进行。可以新建另一段对话。'); return; }
  state.pending = true; updateComposer(); $('composer-error').hidden = true;
  try {
    if (!state.session) {
      state.session = await api('/api/sessions', { method: 'POST', body: '{}' });
      state.session.messages = []; state.session.runs = [];
      savePreference('suuclaw.session', state.session.id);
      ++state.generation;
    }
    const sessionId = state.session.id;
    const run = await api(`/api/sessions/${encodeURIComponent(sessionId)}/runs`, { method: 'POST', body: JSON.stringify({ message }) });
    if (state.session?.id !== sessionId) return;
    run.id = run.id || run.run_id; run.message = run.message || message;
    state.session.messages ||= []; state.session.runs ||= [];
    state.session.messages.push({ role: 'user', content: message, at: run.created_at || new Date().toISOString(), run_id: run.id });
    state.session.runs.push(run); state.session.active_run_id = run.id;
    state.drafts.delete(sessionId); state.drafts.delete('new'); $('message-input').value = ''; resizeComposer();
    resetRun(run); state.autoScroll = true;
    renderRunSelect(); scheduleRender(); setView('workflow');
    connectStream(run.id, state.generation);
    refreshSessions().catch(() => {});
    announce('任务已开始');
  } catch (error) {
    $('composer-error').textContent = readableError(error); $('composer-error').hidden = false;
  } finally { state.pending = false; updateComposer(); }
}
function connectStream(id, generation, attempt = 0) {
  if (generation !== state.generation || state.run?.id !== id || TERMINAL.has(state.run.status)) return;
  closeStream();
  setConnection('connecting', attempt ? '正在重连' : '连接实时动态');
  const stream = new EventSource(`/api/runs/${encodeURIComponent(id)}/events?after=${state.seq}`);
  state.stream = stream;
  stream.onopen = () => { if (state.stream === stream) { setConnection('connected', '实时同步中'); state.connected = true; } };
  stream.onmessage = (message) => {
    if (generation !== state.generation || state.run?.id !== id || state.stream !== stream) return;
    try {
      const event = JSON.parse(message.data);
      applyEvent(event);
      scheduleRender();
      if (event.type === 'run_end') { closeStream(); setConnection('connected', '本地已连接'); finishRun(id, generation); }
    } catch (error) { console.warn('SuuClaw received an unreadable event', error); }
  };
  stream.onerror = () => {
    if (generation !== state.generation || state.stream !== stream) return;
    stream.close(); state.stream = null;
    if (TERMINAL.has(state.run?.status)) { setConnection('connected', '本地已连接'); return; }
    setConnection('connecting', '连接中断，重连中');
    state.reconnectTimer = setTimeout(() => connectStream(id, generation, attempt + 1), Math.min(1000 * 2 ** attempt, 10000));
  };
}
async function finishRun(id, generation) {
  state.session.active_run_id = null;
  announce(state.run.status === 'completed' ? '本轮任务已结束' : '本轮任务需要关注');
  updateComposer();
  try {
    const sessionId = state.session.id;
    const session = await api(`/api/sessions/${encodeURIComponent(sessionId)}`);
    if (generation !== state.generation || state.run?.id !== id || state.session?.id !== sessionId) return;
    // The run_end frame can arrive just before the final store update. Preserve the
    // locally received answer when the first metadata read is slightly behind it.
    for (const message of state.session.messages || []) {
      if (message.run_id === id && !session.messages?.some((item) => item.role === message.role && item.run_id === id)) (session.messages ||= []).push(message);
    }
    session.active_run_id = null;
    state.session = session;
    const savedRun = session.runs?.find((run) => run.id === id);
    if (savedRun && TERMINAL.has(savedRun.status)) state.run = { ...state.run, ...savedRun };
    renderRunSelect(); renderHistory(); scheduleRender();
    refreshSessions().catch(() => {});
    if (state.currentView === 'files') loadFiles();
  } catch { /* The received events remain available if a metadata refresh fails. */ }
}

function ensureNode(id) {
  id = nodeID(id);
  if (!state.nodes.has(id)) state.nodes.set(id, { id, status: 'idle', visits: 0, started: null, elapsed: 0, output: null, lastMessage: '', instruction: '', lastError: '' });
  return state.nodes.get(id);
}
function finishNode(id, event, status = 'completed') {
  const node = ensureNode(id);
  if (node.started !== null) node.elapsed += Math.max(0, (event.elapsed_ms || 0) - node.started);
  node.started = null; node.status = status;
  if (state.activeNode === nodeID(id)) state.activeNode = '';
}
function applyStateData(data) {
  if (!data || typeof data !== 'object') return;
  if (Array.isArray(data.todos)) state.todos = data.todos.filter((row) => row && typeof row === 'object');
  if (typeof data.plan_summary === 'string') state.planSummary = data.plan_summary;
  if (Array.isArray(data.verification_checks)) state.checks = data.verification_checks;
  if (Array.isArray(data.verification_results)) state.results = data.verification_results;
}
function applySuccessfulTool(tool) {
  const args = tool.args || {};
  if (tool.name === 'todo_write') { if (Array.isArray(args.todos)) state.todos = args.todos.map((todo, index) => typeof todo === 'string' ? { id: String(index + 1), content: todo, status: 'pending' } : { status: 'pending', ...todo }); state.planSummary = args.plan_summary || ''; }
  if (tool.name === 'todo_update') { const todo = state.todos.find((item) => String(item.id) === String(args.todo_id)); if (todo) Object.assign(todo, { status: args.status, note: args.note || '' }); }
  if (tool.result && typeof tool.result === 'object') applyStateData(tool.result);
}
function applyEvent(event, replay = false) {
  const seq = Number(event.seq);
  if (Number.isFinite(seq) && seq <= state.seq) return;
  if (Number.isFinite(seq)) state.seq = seq;
  state.events.push(event);
  const id = nodeID(event.node);
  switch (event.type) {
    case 'run_start': {
      const start = ensureNode('__start__'); start.status = 'completed'; start.visits = 1;
      if (state.run) state.run.status = 'running';
      state.pathSteps.push({ id: '__start__', seq, at: event.elapsed_ms || 0 });
      break;
    }
    case 'node_start': {
      const node = ensureNode(id);
      const resuming = node.status === 'delegated' || node.status === 'running';
      const previousStep = state.pathSteps.at(-1)?.id;
      // Returning to the supervisor is a resumption within the same superstep.
      // Its wall-clock duration includes delegation; visits are not model turns.
      if (!resuming) { node.visits += 1; node.started = event.elapsed_ms || 0; node.startSeq = seq; }
      node.status = 'running'; state.activeNode = id;
      if (nodeInfo(id)?.parent) {
        const parent = ensureNode(nodeInfo(id).parent);
        parent.status = 'delegated';
        recordGraphTransfer(parent.id, id, event, replay);
      } else if (previousStep && nodeInfo(previousStep)?.parent === id) {
        recordGraphTransfer(id, previousStep, event, replay, true);
      } else if (state.lastGraphNode !== id) {
        recordGraphTransfer(state.lastGraphNode, id, event, replay);
      }
      if (!nodeInfo(id)?.parent) state.lastGraphNode = id;
      if (previousStep !== id) state.pathSteps.push({ id, seq, at: event.elapsed_ms || 0, resumed: resuming });
      break;
    }
    case 'node_output': {
      const node = ensureNode(id); node.output = event.data;
      const status = event.data?.error ? 'failed' : id === 'verifier' && event.data?.passed === false ? 'rejected' : 'completed';
      finishNode(id, event, status); applyStateData(event.data);
      break;
    }
    case 'node_summary': {
      const node = ensureNode(id); node.lastMessage = event.content || '';
      const visitTools = [...state.tools.values()].filter((tool) => tool.node === id && (tool.seq || 0) >= (node.startSeq || 0));
      const searchesFailed = id === 'search' && visitTools.length > 0 && visitTools.every((tool) => tool.status === 'failed');
      const bad = event.error || event.exhausted || event.interrupted || searchesFailed;
      if (nodeInfo(id)?.kind === 'specialist') finishNode(id, event, bad ? 'failed' : 'completed');
      if (bad) node.lastError = event.error || (event.interrupted ? '执行被中断' : searchesFailed ? '本次搜索请求均未成功' : '达到本轮调用上限');
      break;
    }
    case 'ai_message': {
      const node = ensureNode(id); node.lastMessage = event.content || '';
      if (event.content) state.lastModelMessage = event.content;
      break;
    }
    case 'handoff': {
      const from = nodeID(event.from || id || 'planner'), to = nodeID(event.to);
      ensureNode(from).status = 'delegated';
      ensureNode(to).instruction = event.instruction || '';
      state.activeNode = '';
      recordGraphTransfer(from, to, event, replay);
      break;
    }
    case 'tool_call': {
      const key = event.id || `${seq}-${event.name}`;
      state.tools.set(key, { key, id: event.id, name: event.name, node: id, args: event.args, status: 'running', start: event.elapsed_ms || 0, at: event.at, seq });
      break;
    }
    case 'search_results':
    case 'tool_result': {
      const name = event.name || (event.type === 'search_results' ? 'web_search' : '工具');
      let tool = state.tools.get(event.id);
      if (!tool) tool = [...state.tools.values()].reverse().find((item) => item.name === name && item.status === 'running');
      if (!tool) { const key = event.id || `${seq}-${name}`; tool = { key, name, node: id, args: null, start: null }; state.tools.set(key, tool); }
      const failed = event.ok === false || !!event.error;
      const result = event.type === 'search_results' ? { query: event.query, answer: event.answer, sources: event.sources, result_count: event.result_count, error: event.error } : event.result ?? event.output ?? event.error;
      Object.assign(tool, { status: failed ? 'failed' : 'completed', result, elapsed: tool.start === null ? null : Math.max(0, (event.elapsed_ms || 0) - tool.start) });
      if (!failed) applySuccessfulTool(tool);
      break;
    }
    case 'error': if (id) { ensureNode(id).lastError = event.message || ''; ensureNode(id).status = 'failed'; if (state.activeNode === id) state.activeNode = ''; } break;
    case 'approval_requested': state.approvals.set(event.approval_id, { ...event, resolved: false }); break;
    case 'approval_resolved': {
      const approval = state.approvals.get(event.approval_id);
      if (approval) Object.assign(approval, event, { resolved: true });
      state.pendingApprovals.delete(event.approval_id);
      break;
    }
    case 'final_answer': {
      state.outcome = event;
      const end = ensureNode('__end__'); end.status = event.error || event.interrupted ? 'failed' : event.passed === false ? 'rejected' : 'completed'; end.visits = 1;
      recordGraphTransfer(state.lastGraphNode, '__end__', event, replay);
      state.pathSteps.push({ id: '__end__', seq, at: event.elapsed_ms || 0 });
      if (event.content && state.session) {
        const messages = state.session.messages ||= [];
        if (!messages.some((item) => item.role === 'assistant' && item.run_id === state.run?.id)) messages.push({ role: 'assistant', content: event.content, at: event.at, run_id: state.run?.id });
      }
      break;
    }
    case 'run_end': {
      if (state.run) { state.run.status = event.status || 'completed'; state.run.finished_at = event.at; state.run.duration_ms = event.elapsed_ms; if (event.error) state.run.error = event.error; }
      const bad = event.status === 'failed' || event.status === 'interrupted';
      for (const node of state.nodes.values()) if (['running', 'delegated'].includes(node.status)) finishNode(node.id, event, bad ? 'failed' : 'ended');
      if (bad) ensureNode('__end__').status = 'failed';
      for (const tool of state.tools.values()) if (tool.status === 'running') tool.status = bad ? 'failed' : 'ended';
      if (!replay && state.session?.active_run_id === state.run?.id) state.session.active_run_id = null;
      state.activeNode = ''; state.transfer = null;
      break;
    }
  }
}

function scheduleRender() {
  if (state.renderPending) return;
  state.renderPending = true;
  requestAnimationFrame(() => { state.renderPending = false; renderHeader(); renderGraphState(); architectureView.render(state); renderActivity(); renderInspector(); renderChat(); updateComposer(); if (state.currentView === 'tools') renderTools(); if (state.currentView === 'history') renderHistory(); });
}
function elapsedRun() {
  if (!state.run) return 0;
  if (isRunning()) { const started = new Date(state.run.created_at).getTime(); if (Number.isFinite(started)) return Date.now() - started; }
  return state.run.duration_ms ?? state.run.elapsed_ms ?? state.events.at(-1)?.elapsed_ms ?? 0;
}
function renderHeader() {
  const run = state.run;
  $('run-heading').textContent = run ? clip(run.message || run.task || state.session?.messages?.find((item) => item.role === 'user' && item.run_id === run.id)?.content || state.session?.title || '本轮任务', 70) : '给想法一条行动的路径。';
  const pendingApproval = [...state.approvals.values()].some((approval) => !approval.resolved) && isRunning();
  let status = run?.status || 'idle'; if (pendingApproval) status = 'waiting';
  $('run-status').className = `run-status status-${status}`;
  $('run-status').innerHTML = `<i></i>${escapeHTML(run ? STATUS[status] || status : '准备就绪')}`;
  $('run-description').textContent = !run ? '发起一个任务，观察 SuuClaw 如何理解、规划、执行与验证。' : isRunning() ? pendingApproval ? '有一个操作需要你确认，请在右侧对话中查看详情。' : `正在${state.activeNode ? `「${nodeLabel(state.activeNode)}」` : '处理任务'}。下方路径与动态会随执行更新。` : run.error ? clip(run.error, 180) : state.outcome?.passed === false ? '本轮执行已结束，验证未通过。可在验证结果中查看详情。' : '本轮执行记录已保存。选择节点或动态，检查具体过程。';
  $('graph-mode').textContent = !run ? '真实工作流结构' : isRunning() ? '与任务实时同步' : '历史执行路径';
  $('graph-subtitle').textContent = state.graphMode === 'complete' ? '全部条件与返回回路' : '编排、执行与验证';
  $('run-elapsed').hidden = !run; $('run-tools').hidden = !run;
  if (run) { $('run-elapsed').innerHTML = `${icon('clock')}${duration(elapsedRun())}`; $('run-tools').textContent = `${state.tools.size} 次工具调用`; }
  $('nav-live').hidden = !isRunning();
  $('chat-live').hidden = !isRunning();
  const last = state.events.at(-1);
  $('chat-live-text').textContent = pendingApproval ? '等待操作确认' : last?.type === 'heartbeat' ? `仍在工作，已等待 ${Math.round(last.quiet_for || 0)} 秒` : last?.type === 'retry' ? `模型连接重试 ${last.attempt || ''}` : state.activeNode ? `${nodeLabel(state.activeNode)}正在工作` : '正在准备任务';
  $('chat-presence').textContent = isRunning() ? pendingApproval ? '有一个操作需要确认' : '正在认真做事' : state.bootstrap?.configured ? '准备接住你的想法' : '等待连接模型';
}

// The graph is a presentation of API topology, never a second workflow definition.
// Named routes only supply geometry. Unknown real edges still get a visible fallback.
const GRAPH_LAYOUT = {
  __start__: [64, 81], intent_router: [224, 81], chat_responder: [474, 81], __end__: [777, 81],
  planner: [188, 293], context_monitor: [445, 293], verifier: [688, 293], final: [674, 172],
  search: [137, 486], code: [319, 486], context_compressor: [610, 486],
};
const GRAPH_ROUTES = {
  '__start__:intent_router': ['M89 81H131', 109, 69],
  'intent_router:chat_responder': ['M317 81H381', 349, 68],
  'chat_responder:__end__': ['M567 81H752', 655, 68],
  'intent_router:planner': ['M224 118V174Q224 185 212 185H200Q188 185 188 197V254', 230, 175],
  'planner:context_monitor': ['M281 293H352', 316, 279],
  'context_monitor:verifier': ['M538 293H595', 566, 279],
  'verifier:context_monitor': ['M688 332V359Q688 369 677 369H456Q445 369 445 358V332', 568, 389],
  'context_monitor:planner': ['M407 332V346Q407 357 395 357H201Q188 357 188 344V332', 298, 347, 'secondary'],
  'context_monitor:final': ['M484 254V190Q484 172 504 172H581', 524, 158],
  'context_monitor:context_compressor': ['M508 332V371Q508 382 522 382H790Q804 382 804 396V472Q804 486 790 486H718', 751, 371],
  'context_compressor:verifier': ['M718 470H806Q820 470 820 456V307Q820 293 805 293H781', 809, 339, 'secondary'],
  'context_compressor:planner': ['M568 525V561Q568 577 550 577H38Q24 577 24 561V307Q24 293 40 293H95', 307, 568, 'secondary'],
  'context_compressor:final': ['M677 447V421Q677 408 692 408H810Q828 408 828 392V186Q828 172 812 172H767', 775, 213, 'secondary'],
  'final:__end__': ['M767 172H800Q814 172 814 158V96Q814 81 802 81', 795, 144],
  'planner:search': ['M95 309H64Q51 309 51 324V471Q51 486 62 486', 73, 366],
  'planner:code': ['M234 332V360Q234 374 250 374H414Q428 374 428 388V472Q428 486 412 486H394', 325, 364],
};
function edgeKey(edge) { return `${edge.source}:${edge.target}`; }
function edgeCaption(edge) { return edge.caption || (edge.kind === 'delegation' ? '按需委派' : edge.label || (edge.conditional ? '按条件路由' : '传递状态')); }
function edgeExists(from, to) { return state.bootstrap?.topology?.edges?.some((edge) => edge.source === from && edge.target === to); }
function clearGraphMotion() {
  clearTimeout(graphMotion.next); clearTimeout(graphMotion.expiry);
  graphMotion.next = null; graphMotion.expiry = null; graphMotion.queue = [];
  graphMotion.edges.clear(); graphMotion.nodes.clear();
}
function recordGraphTransfer(from, to, event, replay, returning = false) {
  if (!edgeExists(from, to)) return; // Never light an inferred, nonexistent shortcut.
  const key = `${from}:${to}`;
  state.transitions.add(key);
  if (returning) state.transitions.add(`${to}:${from}`);
  const transfer = { key, from, to, returning, seq: event.seq, destination: returning ? from : to };
  // handoff and the following specialist node_start describe the same transfer.
  const duplicate = state.transfer?.key === key && state.transfer.returning === returning && event.type === 'node_start' && nodeInfo(to)?.parent;
  state.transfer = transfer;
  if (!replay && !duplicate) {
    graphMotion.queue.push(transfer);
    if (graphMotion.queue.length > 8) graphMotion.queue.splice(0, graphMotion.queue.length - 8);
    if (!graphMotion.next) playGraphMotion();
  }
}
function playGraphMotion() {
  const transfer = graphMotion.queue.shift();
  if (!transfer) { graphMotion.next = null; return; }
  const until = Date.now() + (reduceMotion() ? 700 : 1400);
  graphMotion.edges.set(transfer.key, { ...transfer, until });
  graphMotion.nodes.set(transfer.destination, until);
  renderGraphState();
  clearTimeout(graphMotion.expiry);
  graphMotion.expiry = setTimeout(() => { graphMotion.edges.forEach((pulse, key) => { if (pulse.until <= Date.now()) graphMotion.edges.delete(key); }); graphMotion.nodes.forEach((expiry, key) => { if (expiry <= Date.now()) graphMotion.nodes.delete(key); }); renderGraphState(); }, 1450);
  graphMotion.next = setTimeout(() => { graphMotion.next = null; playGraphMotion(); }, reduceMotion() ? 100 : 520);
}
function buildGraph() {
  const topology = state.bootstrap?.topology, graph = $('workflow-graph');
  $('graph-loading').hidden = true;
  if (!topology?.nodes?.length) { $('graph-loading').hidden = false; $('graph-loading').innerHTML = '<p>暂时无法读取工作流结构，请刷新重试。</p>'; return; }
  graph.setAttribute('viewBox', '0 0 840 610');
  const nodes = topology.nodes.map((node, index) => ({ ...node, x: GRAPH_LAYOUT[node.id]?.[0] ?? Number(node.x) ?? 130 + index % 3 * 220, y: GRAPH_LAYOUT[node.id]?.[1] ?? Number(node.y) ?? 560 }));
  const map = new Map(nodes.map((node) => [node.id, node]));
  let html = '<defs>';
  for (const [name, color] of [['idle', '--edge'], ['active', '--accent'], ['visited', '--success'], ['recent', '--warm'], ['failed', '--danger']]) html += `<marker id="arrow-${name}" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="${name === 'active' ? 13 : 11}" markerHeight="${name === 'active' ? 13 : 11}" orient="auto-start-reverse" markerUnits="userSpaceOnUse"><path d="M0 0L10 5L0 10Z" fill="var(${color})"/></marker>`;
  html += '</defs><g class="graph-groups" aria-hidden="true"><rect class="workflow-group" x="20" y="219" width="800" height="338" rx="18"/><text class="group-title" x="38" y="241">Agent workflow</text><text class="group-subtitle" x="179" y="241">编排、执行与验证</text><rect class="specialist-group" x="42" y="402" width="368" height="137" rx="12"/><text class="group-title" x="59" y="425">ReAct Specialists</text><text class="group-subtitle" x="59" y="441">主管按需委派，执行后返回结果</text><rect class="context-group" x="451" y="402" width="340" height="137" rx="12"/><text class="group-title" x="468" y="425">Context budget</text><text class="group-subtitle" x="468" y="441">达到条件时压缩，随后恢复原定步骤</text></g>';
  for (const [index, edge] of (topology.edges || []).entries()) {
    const from = map.get(edge.source), to = map.get(edge.target);
    if (!from || !to) continue;
    const key = edgeKey(edge), route = GRAPH_ROUTES[key];
    const path = route?.[0] || `M${from.x} ${from.y + 40}C${from.x} ${from.y + 90} ${to.x} ${to.y - 90} ${to.x} ${to.y - 40}`;
    const x = route?.[1] ?? (from.x + to.x) / 2, y = route?.[2] ?? (from.y + to.y) / 2;
    const secondary = edge.emphasis === 'secondary' || route?.[3] === 'secondary';
    html += `<g class="edge-group ${secondary ? 'secondary-route' : 'primary-route'} ${edge.kind === 'delegation' ? 'delegation' : ''}" data-route="${escapeHTML(key)}" data-source="${escapeHTML(edge.source)}" data-target="${escapeHTML(edge.target)}"><title>${escapeHTML(`${nodeTitle(edge.source)} → ${nodeTitle(edge.target)}：${edgeCaption(edge)}`)}</title><path id="route-${index}" class="graph-edge ${edge.conditional ? 'conditional' : ''}" d="${path}" marker-end="url(#arrow-idle)"/><path class="edge-travel" d="${path}" pathLength="100"/><text class="edge-label" text-anchor="middle" x="${x}" y="${y}">${escapeHTML(key === 'final:__end__' ? '' : edge.kind === 'delegation' ? 'delegate' : edgeCaption(edge))}</text></g>`;
  }
  for (const node of nodes) {
    const boundary = node.kind === 'boundary', width = node.kind === 'specialist' ? 150 : node.id === 'context_compressor' ? 216 : 186;
    html += `<g class="node ${boundary ? 'boundary' : ''}" data-node="${escapeHTML(node.id)}" transform="translate(${node.x},${node.y})" tabindex="0" role="button" aria-label="${escapeHTML(nodeTitle(node.id))}，${escapeHTML(node.label)}，未经过"><title>${escapeHTML(node.description)}</title>`;
    if (boundary) html += `<circle class="node-halo" r="31"/><circle class="node-body" r="25"/><use href="#i-${node.id === '__end__' ? 'check' : 'chat'}" class="node-symbol" x="-10" y="-10" width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/><text class="node-label" text-anchor="middle" x="0" y="47">${escapeHTML(nodeTitle(node.id))}</text><text class="boundary-subtitle" text-anchor="middle" x="0" y="63">${escapeHTML(node.label)}</text>`;
    else html += `<rect class="node-halo" x="${-width / 2 - 5}" y="-44" width="${width + 10}" height="88" rx="16"/><rect class="node-body" x="${-width / 2}" y="-39" width="${width}" height="78" rx="11"/><text class="node-label" x="${-width / 2 + 14}" y="-13">${escapeHTML(nodeTitle(node.id))}</text><text class="node-subtitle" x="${-width / 2 + 14}" y="7">${escapeHTML(node.subtitle || node.label)}</text><circle class="node-status-dot" cx="${-width / 2 + 18}" cy="24" r="3"/><text class="node-runtime" x="${-width / 2 + 28}" y="28">未经过</text><use href="#i-${nodeIcon(node)}" class="node-symbol" x="${width / 2 - 29}" y="3" width="17" height="17" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/>`;
    html += '</g>';
  }
  graph.innerHTML = html;
  renderTopologyList(); renderGraphState();
}
function observedTransitions() { return new Set(state.transitions); }
function renderGraphState() {
  const now = Date.now(), graph = $('workflow-graph');
  graph.classList.toggle('complete-topology', state.graphMode === 'complete');
  $$('[data-node]', graph).forEach((element) => {
    const id = element.dataset.node, node = state.nodes.get(id), status = node?.status || 'idle';
    element.classList.remove('running', 'completed', 'failed', 'rejected', 'delegated', 'ended', 'selected', 'recent');
    if (status !== 'idle') element.classList.add(status);
    element.classList.toggle('recent', (graphMotion.nodes.get(id) || 0) > now);
    element.classList.toggle('selected', id === state.selectedNode);
    element.setAttribute('aria-label', `${nodeTitle(id)}，${nodeLabel(id)}，${STATUS[status] || status}，点击查看详情`);
    if (id === '__end__') element.querySelector('.node-symbol')?.setAttribute('href', ['failed', 'rejected', 'interrupted'].includes(status) ? '#i-alert' : '#i-check');
    const runtime = element.querySelector('.node-runtime');
    if (runtime) runtime.textContent = !node ? '未经过' : `${STATUS[status] || status}${node.visits > 1 ? ` / 第 ${node.visits} 次` : ''}`;
  });
  $$('.edge-group', graph).forEach((group) => {
    const key = group.dataset.route, pulse = graphMotion.edges.get(key), recent = pulse && pulse.until > now;
    const transfer = state.transfer?.key === key ? state.transfer : null;
    const active = !!transfer && isRunning() && state.nodes.get(transfer.destination)?.status === 'running';
    const returning = active ? transfer.returning : recent ? pulse.returning : false;
    const related = group.dataset.source === state.selectedNode || group.dataset.target === state.selectedNode;
    const visited = state.transitions.has(key);
    const failed = visited && ['failed', 'rejected', 'interrupted'].includes(state.nodes.get(group.dataset.target)?.status);
    group.classList.toggle('failed-route', failed);
    group.classList.toggle('visited', visited); group.classList.toggle('active', active); group.classList.toggle('recent', !!recent); group.classList.toggle('returning', !!returning); group.classList.toggle('related', related);
    const path = group.querySelector('.graph-edge');
    const color = failed ? 'failed' : active ? 'active' : recent ? 'recent' : visited ? 'visited' : 'idle';
    path.setAttribute(returning ? 'marker-start' : 'marker-end', `url(#arrow-${color})`);
    path.removeAttribute(returning ? 'marker-end' : 'marker-start');
  });
  renderExecutionFocus();
}
function renderExecutionFocus() {
  const runningNode = state.activeNode && state.nodes.get(state.activeNode)?.status === 'running' ? state.activeNode : [...state.nodes.values()].find((node) => node.status === 'running')?.id;
  const waitingApproval = isRunning() && [...state.approvals.values()].some((approval) => !approval.resolved);
  const liveTool = [...state.tools.values()].reverse().find((tool) => tool.status === 'running');
  const focus = $('execution-focus');
  focus.classList.toggle('live', isRunning() && !!runningNode && !waitingApproval);
  focus.classList.toggle('needs-attention', waitingApproval || state.run?.status === 'failed' || state.run?.status === 'interrupted' || state.outcome?.passed === false);
  $('focus-title').textContent = waitingApproval ? 'Approval required' : runningNode && isRunning() ? nodeTitle(runningNode) : !state.run ? 'Ready' : isRunning() ? 'Routing' : state.outcome?.passed === false ? 'Review required' : state.run.status === 'completed' ? 'Run completed' : state.run.status === 'failed' ? 'Run failed' : 'Run interrupted';
  $('focus-description').textContent = waitingApproval ? '等待你在对话栏确认操作' : liveTool && isRunning() ? `${nodeLabel(liveTool.node)}正在调用 ${liveTool.name}` : runningNode && isRunning() ? `${nodeLabel(runningNode)}正在执行，状态来自实际事件` : !state.run ? '等待任务，节点会按实际执行点亮' : isRunning() ? '上一步已返回，等待下一条执行事件' : '已保留实际路径；历史记录不会自动播放动画';
  $('focus-counter').textContent = state.run ? `${state.events.length} 条事件 / ${state.tools.size} 次工具调用` : '本轮尚未开始';
  const steps = state.pathSteps;
  const html = steps.length ? steps.map((step, index) => `<button class="path-step ${index === steps.length - 1 && isRunning() ? 'current' : ''}" data-inspect="${escapeHTML(step.id)}" title="${escapeHTML(nodeLabel(step.id))}，+${duration(step.at)}${step.resumed ? '，从委派中恢复' : ''}">${escapeHTML(nodeTitle(step.id))}${step.resumed ? '<small>返回</small>' : ''}</button>${index < steps.length - 1 ? icon('arrow') : ''}`).join('') : '<span class="path-placeholder">任务开始后显示，不预设执行顺序</span>';
  const area = $('path-steps'), atEnd = area.scrollWidth - area.scrollLeft - area.clientWidth < 50;
  setHTML(area, html); if (atEnd) area.scrollLeft = area.scrollWidth;
}
function renderTopologyList() {
  const edges = state.bootstrap?.topology?.edges || [];
  $('topology-count').textContent = `${edges.length} 条实际连接`;
  setHTML($('topology-connections'), edges.map((edge) => `<div class="topology-row"><button data-inspect="${escapeHTML(edge.source)}">${escapeHTML(nodeTitle(edge.source))}</button>${icon('arrow')}<button data-inspect="${escapeHTML(edge.target)}">${escapeHTML(nodeTitle(edge.target))}</button><span>${escapeHTML(edgeCaption(edge))}${edge.label && edge.kind !== 'delegation' ? `<code>${escapeHTML(edge.label)}</code>` : ''}</span></div>`).join(''));
}
function revealInMain(element) {
  const scroller = $('main-scroll');
  const top = scroller.scrollTop + element.getBoundingClientRect().top - scroller.getBoundingClientRect().top - 16;
  scroller.scrollTo({ top: Math.max(0, top), behavior: reduceMotion() ? 'auto' : 'smooth' });
}
function inspectNode(id) {
  setPerspective('workflow');
  closeExpandedGraphs();
  state.selectedNode = id; renderGraphState(); renderInspector();
  revealInMain($('inspector-panel'));
}
function renderInspector() {
  const definition = nodeInfo(state.selectedNode);
  $('inspector-panel').hidden = state.perspective !== 'workflow' || !definition;
  if (!definition) return;
  const node = state.nodes.get(definition.id), tools = [...state.tools.values()].filter((tool) => tool.node === definition.id);
  $('inspector-title').textContent = `${nodeTitle(definition.id)} / ${definition.subtitle || definition.label}`;
  $('inspector-icon').innerHTML = icon(nodeIcon(definition)); $('inspector-status').textContent = STATUS[node?.status || 'idle'];
  let html = `<p class="inspector-description">${escapeHTML(definition.description)}</p>`;
  if (definition.parent) html += '<p class="inspector-note">此专家运行于 Supervisor 的一次编排步骤内部，完成后返回主管；不是独立的并行分支。</p>';
  if (node) html += `<div class="inspector-metrics"><span>节点进入<strong>${node.visits} 次</strong></span><span>观测耗时<strong>${duration(node.elapsed + (node.started !== null && isRunning() ? Math.max(0, elapsedRun() - node.started) : 0))}</strong>${definition.id === 'planner' ? '<small>含委派等待</small>' : ''}</span><span>工具调用<strong>${tools.length} 次</strong></span></div>`;
  if (node?.lastError) html += `<p class="inspector-error">${escapeHTML(node.lastError)}</p>`;
  const outgoing = (state.bootstrap?.topology?.edges || []).filter((edge) => edge.source === definition.id);
  html += '<div class="inspector-grid"><section class="inspector-section"><h4>Input <span>已记录输入</span></h4>';
  const task = state.run?.message || state.run?.task;
  html += node?.instruction ? `<p class="inspector-record">${escapeHTML(node.instruction)}</p><small>Supervisor 发出的实际委派指令</small>` : task ? `<p class="inspector-record">${escapeHTML(clip(task, 650))}</p><small>本轮用户任务；事件流未记录完整节点上下文</small>` : '<p class="inspector-placeholder">运行后显示本轮任务或实际委派指令。</p>';
  html += '</section><section class="inspector-section"><h4>Routing <span>可达下游</span></h4>';
  html += definition.parent ? `<button class="inspector-route" data-inspect="${escapeHTML(definition.parent)}"><strong>返回 ${escapeHTML(nodeTitle(definition.parent))}</strong><span>委派内返回 / 将结果交回任务编排</span>${icon('chevron')}</button>` : outgoing.length ? outgoing.map((edge) => `<button class="inspector-route" data-inspect="${escapeHTML(edge.target)}"><strong>${escapeHTML(nodeTitle(edge.target))}</strong><span>${escapeHTML(edgeCaption(edge))}</span>${icon('chevron')}</button>`).join('') : '<p class="inspector-placeholder">这个节点没有下游连接。</p>';
  html += '</section></div><section class="inspector-section"><h4>Output <span>实际返回</span></h4>';
  if (node?.lastMessage) html += `<div class="inspector-output message-content">${markdown(node.lastMessage)}</div>`;
  if (node?.output) html += disclosure('结构化节点数据', node.output, `node-data-${definition.id}`);
  if (!node?.lastMessage && !node?.output) html += '<p class="inspector-placeholder">尚未收到这个节点的输出。</p>';
  html += '</section>';
  if (tools.length) html += `<section class="inspector-section"><h4>Tool calls <span>工具调用与结果</span></h4>${tools.map((tool) => disclosure(`${tool.name} / ${STATUS[tool.status]}${tool.elapsed != null ? ` / ${duration(tool.elapsed)}` : ''}`, { input: tool.args, output: tool.result }, `node-tool-${tool.key}`)).join('')}</section>`;
  setHTML($('inspector-content'), html);
}

function eventPresentation(event) {
  const label = nodeLabel(event.node);
  const descriptions = {
    run_start: ['spark', '任务已开始', '事件将按实际发生顺序记录。'],
    node_start: ['flow', `${label}开始工作`, nodeInfo(event.node)?.description || ''],
    node_output: ['check', `${label}返回结果`, event.data?.plan_summary || '可展开查看节点返回的数据。', 'success'],
    node_summary: ['check', `${label}结束本次处理`, clip(event.content), event.error ? 'error' : 'success'],
    ai_message: ['cpu', `${label}收到模型回复`, event.tool_calls?.length ? `准备调用：${event.tool_calls.join('、')}` : clip(event.content)],
    tool_call: ['tools', `调用 ${event.name || '工具'}`, `${label}${event.args?.path ? ` / ${event.args.path}` : ''}`],
    search_results: [event.ok === false ? 'alert' : 'search', event.ok === false ? 'Web Search 查询失败' : 'Web Search 返回来源', `${event.result_count ?? event.sources?.length ?? 0} 条来源 / ${clip(event.query || event.error)}`, event.ok === false ? 'error' : 'success'],
    memory: ['memory', `${label}组装三层记忆`, 'Rules / Working Memory / History Summary，可展开查看本次快照。'],
    memory_snapshot: ['memory', `${label}更新上下文`, '可展开查看实际上下文记录。'],
    tool_result: [event.ok === false ? 'alert' : 'check', `${event.name || '工具'}${event.ok === false ? '未成功' : '已返回'}`, `${label} / ${event.ok === false ? '查看结果以了解原因' : '结果已交回工作流'}`, event.ok === false ? 'error' : 'success'],
    handoff: ['arrow', `委派给${nodeLabel(nodeID(event.to))}`, clip(event.instruction)],
    heartbeat: ['clock', `${label}仍在工作`, `距离上次动态 ${Math.round(event.quiet_for || 0)} 秒。这是存活信号，并不代表任务进度。`],
    retry: ['refresh', '模型连接正在重试', `第 ${event.attempt || '?'} 次 / ${clip(event.message)}`, 'warning'],
    error: ['alert', `${label}遇到问题`, event.message || '查看事件详情。', 'error'],
    approval_requested: ['shield', '等待你确认操作', event.risk_reason || event.command, 'warning'],
    approval_resolved: [event.approved ? 'check' : 'shield', event.approved ? '操作已获准' : '操作未获准', event.reason || '', event.approved ? 'success' : 'warning'],
    final_answer: ['check', '已生成本轮答复', event.passed === false ? '验证未通过，请检查验证结果。' : clip(event.content), event.error ? 'error' : 'success'],
    run_end: [event.status === 'completed' ? 'check' : 'alert', `本轮${STATUS[event.status] || '已结束'}`, event.error || '运行记录已保存在本地。', event.status === 'completed' ? 'success' : 'error'],
  };
  return descriptions[event.type] || ['info', event.type || '工作流事件', clip(event.message || event.content || '')];
}
function renderActivity() {
  $('events-count').textContent = state.events.length;
  $('plan-count').textContent = state.todos.length;
  $('verification-count').textContent = state.checks.length + state.results.length;
  $('event-filter-label').hidden = state.tab !== 'events';
  let html;
  if (state.tab === 'events') {
    let events = state.events.filter((event) => state.filter === 'all' || state.filter === 'tools' && ['tool_call', 'tool_result', 'search_results', 'handoff'].includes(event.type) || state.filter === 'nodes' && ['node_start', 'node_output', 'node_summary'].includes(event.type) || state.filter === 'memory' && ['memory', 'memory_snapshot'].includes(event.type) || state.filter === 'errors' && (['error', 'retry', 'approval_requested'].includes(event.type) || ['tool_result', 'search_results'].includes(event.type) && event.ok === false));
    if (!events.length) html = emptyState('flow', state.events.length ? '暂时没有这类动态' : '行动发生时，这里会亮起来', state.events.length ? '试试其他动态筛选。' : '发起任务后，模型回复、工具调用与节点变化会按顺序出现在这里。');
    else {
      const total = events.length; events = events.slice(-120).reverse();
      html = events.map((event) => {
        const [name, title, description, style = ''] = eventPresentation(event);
        return `<article class="event-row ${style}"><span class="event-icon">${icon(name)}</span><div><div class="event-title">${escapeHTML(title)}</div>${description ? `<span class="event-subtitle">${escapeHTML(clip(description, 185))}</span>` : ''}</div><time class="event-time" title="${escapeHTML(displayTime(event.at, true))}">+${duration(event.elapsed_ms || 0)}</time><div class="event-details">${disclosure('事件详情', event, `event-${event.seq}`)}</div></article>`;
      }).join('');
      if (total > 120) html += `<div class="empty-state"><p>显示最近 120 条动态，共 ${total} 条。切换筛选可聚焦具体类型。</p></div>`;
    }
  } else if (state.tab === 'plan') {
    html = state.todos.length ? (state.planSummary ? `<p class="plan-summary">${escapeHTML(state.planSummary)}</p>` : '') + state.todos.map((todo, index) => `<div class="todo-row ${escapeHTML(todo.status || 'pending')}"><span class="todo-mark">${todo.status === 'completed' ? icon('check') : index + 1}</span><div class="todo-content">${escapeHTML(todo.content || todo.title || todo.description || '')}${todo.note ? `<span class="todo-note">${escapeHTML(todo.note)}</span>` : ''}</div><span class="todo-state">${escapeHTML(STATUS[todo.status] || '待开始')}</span></div>`).join('') : emptyState('check', '主管尚未发布计划', '需要多步处理的任务会在这里显示计划与实际完成状态；简单对话可能不经过规划。');
  } else {
    const verified = state.nodes.has('verifier') || state.checks.length || state.results.length;
    html = state.outcome ? `<div class="verification-outcome ${verified && state.outcome.passed === true ? 'passed' : verified && state.outcome.passed === false ? 'failed' : ''}">${icon(verified && state.outcome.passed ? 'shield' : 'info')} ${!verified ? '本轮未经过独立验证' : state.outcome.passed === true ? '本轮验证通过' : state.outcome.passed === false ? '本轮验证未通过' : '本轮未记录验证结论'}${verified && state.outcome.attempts != null ? ` / ${state.outcome.attempts} 次尝试` : ''}</div>` : '';
    html += state.checks.map((check, index) => `<article class="event-row ${check.passed ? 'success' : 'error'}"><span class="event-icon">${icon(check.passed ? 'check' : 'alert')}</span><div><p class="event-title">${escapeHTML(check.name || `验收项 ${index + 1}`)}</p><p class="event-subtitle">${escapeHTML(check.detail || '')}</p></div><span class="event-time">${check.passed ? '通过' : '未通过'}</span></article>`).join('');
    html += state.results.map((result, index) => `<article class="event-row ${result.ok ? 'success' : 'error'}"><span class="event-icon">${icon(result.ok ? 'check' : 'alert')}</span><div><p class="event-title">${escapeHTML(result.command)}</p><p class="event-subtitle">退出码：${escapeHTML(result.exit_code ?? '未返回')}</p></div><span class="event-time">${result.ok ? '通过' : '未通过'}</span><div class="event-details">${disclosure('命令输出', { stdout: result.stdout, stderr: result.stderr }, `verify-${index}`)}</div></article>`).join('');
    if (!state.checks.length && !state.results.length) html += emptyState('shield', '尚无逐项验证记录', '验证器运行后会在这里显示验收判断与命令结果。');
  }
  setHTML($('activity-content'), html);
}

function messageHTML(message, index) {
  const user = message.role === 'user';
  return `<article class="message ${user ? 'user' : 'assistant'}"><div class="message-meta">${user ? '<span class="message-avatar">你</span>' : '<img src="/static/suuclaw-mark.svg" alt="">'}<strong>${user ? '你' : 'SuuClaw'}</strong><time>${escapeHTML(displayTime(message.at))}</time></div><div class="message-content">${markdown(message.content)}</div>${!user ? `<button class="message-copy" type="button" data-action="copy-message" data-index="${index}">${icon('copy')}复制回答</button>` : ''}</article>`;
}
function toolChatHTML() {
  const tools = [...state.tools.values()];
  if (!tools.length) return '';
  let html = `<div class="run-chat-summary">${icon('tools')}<span>本轮 ${tools.length} 次工具调用</span><button class="text-button" data-action="show-tools" type="button">查看全部动态</button></div>`;
  html += tools.slice(-5).map((tool) => `<details class="chat-tool ${tool.status}" data-detail="chat-tool-${escapeHTML(tool.key)}"><summary>${icon(tool.status === 'running' ? 'tools' : tool.status === 'failed' ? 'alert' : 'check')}<span>${escapeHTML(tool.name)}</span><small>${tool.status === 'running' ? '执行中' : tool.elapsed != null ? duration(tool.elapsed) : STATUS[tool.status]}</small></summary><pre>${escapeHTML(json({ input: tool.args, ...(tool.result !== undefined ? { output: tool.result } : {}) }))}</pre></details>`).join('');
  return html;
}
function approvalsHTML() {
  return [...state.approvals.values()].map((approval) => {
    const expired = approval.expires_at && new Date(approval.expires_at).getTime() < Date.now();
    const actionable = !approval.resolved && isRunning() && !expired;
    return `<section class="approval-card"><h4>${icon('shield')}${approval.resolved ? approval.approved ? '已允许这次操作' : '这次操作未获准' : expired ? '确认已过期' : !isRunning() ? '这次确认已结束' : '这个操作需要你确认'}</h4><p>${escapeHTML(approval.risk_reason || '检查命令内容后，决定是否允许这一次操作。')}</p><pre>${escapeHTML(approval.command)}</pre>${actionable ? `<div class="approval-actions"><button class="primary-button" data-approval="${escapeHTML(approval.approval_id)}" data-approved="true" ${state.pendingApprovals.has(approval.approval_id) ? 'disabled' : ''}>允许这一次</button><button class="secondary-button" data-approval="${escapeHTML(approval.approval_id)}" data-approved="false" ${state.pendingApprovals.has(approval.approval_id) ? 'disabled' : ''}>拒绝</button></div>` : `<p>${escapeHTML(approval.reason || '决定已记录在本轮动态中。')}</p>`}</section>`;
  }).join('');
}
function renderChat() {
  const area = $('chat-messages');
  const atBottom = state.autoScroll || area.scrollHeight - area.scrollTop - area.clientHeight < 75;
  const messages = state.session?.messages || [];
  if (!messages.length && !state.run) { setHTML(area, welcomeHTML); return; }
  let html = '', extrasAdded = false;
  messages.forEach((message, index) => {
    if (message.role === 'assistant' && message.run_id === state.run?.id && !extrasAdded) { html += toolChatHTML() + approvalsHTML(); extrasAdded = true; }
    html += messageHTML(message, index);
  });
  if (!extrasAdded) html += toolChatHTML() + approvalsHTML();
  if (isRunning() && state.lastModelMessage && !state.outcome) html += `<details class="chat-tool" data-detail="live-model-reply"><summary>${icon('cpu')}<span>${escapeHTML(nodeLabel(state.activeNode))}的最近回复</span></summary><div class="message-content" style="padding:12px">${markdown(state.lastModelMessage)}</div></details>`;
  if (state.run?.status === 'failed' && !state.outcome?.content) html += `<div class="approval-card"><h4>${icon('alert')}本轮未能完成</h4><p>${escapeHTML(state.run.error || state.events.findLast((event) => event.type === 'error')?.message || '在工作流动态中查看错误详情，然后重新发送任务。')}</p></div>`;
  setHTML(area, html);
  if (atBottom) area.scrollTop = area.scrollHeight;
}
function updateComposer() {
  const input = $('message-input');
  const busy = state.pending || currentSessionRunning();
  input.disabled = state.initializing;
  $('send-button').disabled = state.initializing || busy || !input.value.trim() || !state.bootstrap;
  $('send-button').title = busy ? '当前对话仍有任务进行中' : state.bootstrap?.configured ? '发送任务' : '查看模型配置方法';
  input.placeholder = state.initializing ? '正在恢复对话…' : busy ? '任务进行中，可以先写下下一步想法…' : '把想法交给 SuuClaw…';
}
function resizeComposer() { const input = $('message-input'); input.style.height = 'auto'; input.style.height = `${Math.min(input.scrollHeight, 180)}px`; }
async function approve(id, approved) {
  if (!state.run || state.pendingApprovals.has(id)) return;
  state.pendingApprovals.add(id); renderChat();
  try {
    await api(`/api/runs/${encodeURIComponent(state.run.id)}/approvals/${encodeURIComponent(id)}`, { method: 'POST', body: JSON.stringify({ approved }) });
    toast(approved ? '已允许这一次操作' : '已拒绝这次操作');
  } catch (error) { state.pendingApprovals.delete(id); toast(readableError(error)); renderChat(); }
}

function renderTools() {
  const query = $('tool-search').value.trim().toLowerCase();
  const tools = (state.bootstrap?.tools || []).filter((tool) => `${tool.name} ${tool.description}`.toLowerCase().includes(query));
  const uses = [...state.tools.values()];
  setHTML($('tools-list'), tools.length ? tools.map((tool) => {
    const calls = uses.filter((item) => item.name === tool.name), latest = calls.at(-1);
    const name = /search/.test(tool.name) ? 'search' : /file|directory|list/.test(tool.name) ? 'files' : /command|shell|code/.test(tool.name) ? 'code' : /todo/.test(tool.name) ? 'check' : 'tools';
    return `<article class="tool-card"><span class="tool-card-icon">${icon(name)}</span><div><h3>${escapeHTML(tool.name)}</h3><p>${escapeHTML(tool.description)}</p>${tool.signature ? disclosure('参数说明', tool.signature, `signature-${tool.name}`) : ''}${latest ? disclosure('最近一次调用', { input: latest.args, output: latest.result }, `last-tool-${tool.name}`) : ''}</div><span class="tool-state ${latest?.status || ''}">${latest ? `${STATUS[latest.status]} / ${calls.length} 次` : '本轮未调用'}</span></article>`;
  }).join('') : emptyState('tools', query ? '没有找到匹配的工具' : '暂时没有可用工具', query ? '试试工具名称或用途中的其他关键词。' : '连接本地服务后，这里会显示实际注册的工具。'));
}
async function loadFiles() {
  const id = state.session?.id, generation = ++state.fileGeneration;
  $('file-location').textContent = state.session?.workspace || '';
  if (!id) { setHTML($('files-list'), emptyState('files', '先开始一段对话', '每段对话拥有自己的工作区，任务创建的文件会显示在这里。')); return; }
  try {
    const response = await api(`/api/sessions/${encodeURIComponent(id)}/files`);
    if (generation !== state.fileGeneration || id !== state.session?.id) return;
    state.files = response.files || [];
    setHTML($('files-list'), state.files.length ? state.files.map((file) => `<button class="file-row" data-file="${escapeHTML(file.path)}">${icon(/\.(py|js|ts|html|css|json|sh|ps1)$/.test(file.path) ? 'code' : 'files')}<span><strong>${escapeHTML(file.name || file.path)}</strong><small>${escapeHTML(file.path)}${file.modified_at ? ` / ${escapeHTML(displayTime(file.modified_at, true))}` : ''}</small></span><span class="file-size">${bytes(file.size || 0)}</span>${icon('chevron')}</button>`).join('') + (response.truncated ? '<p class="file-notice">文件较多，当前显示部分文件。</p>' : '') : emptyState('files', '工作区还是一张白纸', '让 SuuClaw 创建一个文件，完成后刷新这里查看成果。'));
  } catch (error) { if (generation === state.fileGeneration) setHTML($('files-list'), emptyState('alert', '文件列表未能读取', readableError(error))); }
}
async function previewFile(path) {
  if (!state.session) return;
  const sessionId = state.session.id;
  try {
    const endpoint = `/api/sessions/${encodeURIComponent(sessionId)}/file?path=${encodeURIComponent(path)}`;
    const file = await api(endpoint);
    if (state.session?.id !== sessionId) return;
    $('file-preview-name').textContent = file.path || file.name;
    $('file-content').textContent = file.binary ? '此文件无法以文本预览。可下载后使用相应应用打开。' : file.content || '';
    $('file-notice').hidden = !file.truncated && !file.binary;
    $('file-notice').textContent = file.binary ? '二进制文件' : '文件较大，这里只显示前一部分。下载可查看完整内容。';
    $('file-download').href = endpoint + '&download=1';
    $('file-preview').hidden = false;
    revealInMain($('file-preview'));
  } catch (error) { toast(readableError(error)); }
}
function renderHistory() {
  const runs = state.session?.runs || [];
  setHTML($('history-list'), runs.length ? [...runs].reverse().map((run) => `<button class="history-card" data-run="${escapeHTML(run.id)}"><div><h3>${escapeHTML(clip(run.message || run.task || state.session?.messages?.find((message) => message.role === 'user' && message.run_id === run.id)?.content || '一轮任务', 80))}</h3><p>${escapeHTML(displayTime(run.created_at, true))}${run.duration_ms != null || run.elapsed_ms != null ? ` / ${duration(run.duration_ms ?? run.elapsed_ms)}` : ''} / ${escapeHTML(String(run.id).slice(0, 8))}</p></div><span class="run-status status-${escapeHTML(run.status)}"><i></i>${escapeHTML(STATUS[run.status] || run.status)}</span>${icon('chevron')}</button>`).join('') : emptyState('history', '还没有需要回看的执行', '发起任务后，每一轮执行都会留在当前对话中。左侧可以切换其他对话。'));
}
function showDialog(title, html) { $('dialog-title').textContent = title; $('dialog-content').innerHTML = html; $('info-dialog').showModal(); }
function setupHTML() {
  return '<p>在启动 SuuClaw 的项目目录中配置 <code>.env</code>。模型密钥保存在本地，工作台不会让你在网页中填写密钥。</p><pre><code>OPENAI_API_KEY=你的模型密钥\nOPENAI_MODEL=你的模型名称\n# 兼容服务可填写对应地址\nOPENAI_BASE_URL=https://api.openai.com/v1</code></pre><p>保存后重启工作台，再刷新页面。当前项目的 <code>.env.example</code> 提供了配置示例。</p><button class="primary-button" id="reload-bootstrap">重新检查连接</button>';
}
function showSetup() { showDialog('连接你的模型', setupHTML()); }
function showWorkspace() {
  showDialog('本地工作空间', `<p>每段对话在独立工作区运行。模型通过已注册的工具完成任务，过程保存在本地。</p><dl><dt>工作区根目录</dt><dd>${escapeHTML(state.bootstrap?.workspace || '尚未连接')}</dd><dt>当前对话目录</dt><dd>${escapeHTML(state.session?.workspace || '创建对话后生成')}</dd><dt>模型</dt><dd>${escapeHTML(state.bootstrap?.model?.name || '尚未配置')}</dd><dt>模型状态</dt><dd>${state.bootstrap?.configured ? '已配置（可发起任务验证连接）' : '尚未配置'}</dd></dl>${!state.bootstrap?.configured ? setupHTML() : '<p>切换模型后需要重启本地工作台。模型实际可用性以调用结果为准。</p>'}`);
}
async function loadBootstrap() {
  const bootstrap = await api('/api/bootstrap');
  state.bootstrap = bootstrap;
  if (bootstrap.configured == null) bootstrap.configured = !!bootstrap.model?.configured;
  $('config-banner').hidden = !!bootstrap.configured;
  $('composer-model').lastElementChild.textContent = bootstrap.model?.name || '尚未配置模型';
  $('composer-model').title = bootstrap.model?.name || '尚未配置模型';
  $('workspace-label').textContent = String(bootstrap.workspace || '').split(/[\\/]/).filter(Boolean).at(-1) || '本地工作区';
  $('workspace-label').title = bootstrap.workspace || '';
  $('tool-count').textContent = bootstrap.tools?.length || '';
  buildGraph(); renderTools(); scheduleRender();
  state.connected = true; setConnection('connected', '本地已连接');
}

// UI wiring stays in one place; every mutation goes through the same run/session state.
const architectureView = new ArchitectureView($('architecture-root'), {
  onInspectNode: inspectNode,
  onOpenView: (view) => { if (view === 'workflow') setPerspective('workflow'); setView(view); },
});
function closeExpandedGraphs() {
  for (const [panel, button, label] of [['architecture-panel', 'architecture-fullscreen', '系统全景'], ['execution-panel', 'graph-fullscreen', '工作流']]) {
    $(panel).classList.remove('expanded');
    $(button).setAttribute('aria-expanded', 'false');
    $(button).setAttribute('aria-label', `展开${label}`); $(button).title = `展开${label}`;
  }
}
function setPerspective(perspective) {
  state.perspective = perspective === 'workflow' ? 'workflow' : 'architecture';
  closeExpandedGraphs();
  for (const mode of ['architecture', 'workflow']) $('perspective-' + mode).setAttribute('aria-pressed', String(state.perspective === mode));
  $('architecture-panel').hidden = state.perspective !== 'architecture';
  $('execution-panel').hidden = state.perspective !== 'workflow';
  $('perspective-description').textContent = state.perspective === 'architecture' ? '入口、记忆与工具如何协作' : '节点实际经过的路径与条件';
  savePreference('suuclaw.perspective', state.perspective);
  renderInspector(); architectureView.render(state);
}
$('perspective-architecture').addEventListener('click', () => setPerspective('architecture'));
$('perspective-workflow').addEventListener('click', () => setPerspective('workflow'));
$('architecture-fullscreen').addEventListener('click', () => {
  const expanded = $('architecture-panel').classList.toggle('expanded');
  $('architecture-fullscreen').setAttribute('aria-expanded', String(expanded));
  $('architecture-fullscreen').setAttribute('aria-label', expanded ? '收起系统全景' : '展开系统全景');
  $('architecture-fullscreen').title = expanded ? '收起系统全景（Esc）' : '展开系统全景';
});
$$('[data-theme-choice]').forEach((button) => button.addEventListener('click', () => { applyTheme(button.dataset.themeChoice); $('theme-picker').open = false; toast(`已切换为「${THEMES[button.dataset.themeChoice].label}」`); }));
$$('[data-view]').forEach((button) => button.addEventListener('click', () => setView(button.dataset.view)));
$('new-session').addEventListener('click', newSession);
$('chat-new').addEventListener('click', newSession);
$('refresh-sessions').addEventListener('click', () => refreshSessions().catch((error) => toast(readableError(error))));
$('sessions-list').addEventListener('click', (event) => { const button = event.target.closest('[data-session]'); if (button) selectSession(button.dataset.session); });
$('chat-toggle').addEventListener('click', () => setChat($('app-shell').classList.contains('chat-hidden')));
$('chat-close').addEventListener('click', () => setChat(false));
$('menu-toggle').addEventListener('click', () => { const open = !$('sidebar').classList.contains('open'); $('sidebar').classList.toggle('open', open); $('sidebar-scrim').hidden = !open; $('menu-toggle').setAttribute('aria-expanded', String(open)); });
$('sidebar-scrim').addEventListener('click', closeSidebar);
$('composer').addEventListener('submit', sendMessage);
$('message-input').addEventListener('input', () => { resizeComposer(); updateComposer(); });
$('message-input').addEventListener('keydown', (event) => { if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); $('composer').requestSubmit(); } });
$('run-select').addEventListener('change', () => { if ($('run-select').value) selectRun($('run-select').value); });
$('graph-overview').addEventListener('click', () => setGraphMode('overview'));
$('graph-complete').addEventListener('click', () => setGraphMode('complete'));
function setGraphMode(mode) { state.graphMode = mode; $('graph-overview').setAttribute('aria-pressed', String(mode === 'overview')); $('graph-complete').setAttribute('aria-pressed', String(mode === 'complete')); renderHeader(); renderGraphState(); }
for (const container of [$('path-steps'), $('topology-connections'), $('inspector-content')]) container.addEventListener('click', (event) => { const button = event.target.closest('[data-inspect]'); if (button) inspectNode(button.dataset.inspect); });
$('workflow-graph').addEventListener('click', (event) => { const node = event.target.closest('[data-node]'); if (node) inspectNode(node.dataset.node); });
$('workflow-graph').addEventListener('keydown', (event) => { if (event.key === 'Enter' || event.key === ' ') { const node = event.target.closest('[data-node]'); if (node) { event.preventDefault(); inspectNode(node.dataset.node); } } });
$('close-inspector').addEventListener('click', () => { state.selectedNode = ''; $('inspector-panel').hidden = true; renderGraphState(); });
$('graph-fit').addEventListener('click', () => {
  const fit = $('graph-viewport').classList.toggle('fit');
  $('graph-viewport').scrollTo({ left: 0, top: 0, behavior: 'smooth' });
  $('graph-fit').setAttribute('aria-pressed', String(fit));
  $('graph-fit').setAttribute('aria-label', fit ? '恢复清晰尺寸' : '适应画布');
  $('graph-fit').title = fit ? '恢复清晰尺寸' : '适应画布';
});
$('graph-fullscreen').addEventListener('click', () => { const panel = $('execution-panel'); const expanded = panel.classList.toggle('expanded'); $('graph-fullscreen').setAttribute('aria-expanded', String(expanded)); $('graph-fullscreen').setAttribute('aria-label', expanded ? '收起工作流' : '展开工作流'); $('graph-fullscreen').title = expanded ? '收起工作流（Esc）' : '展开工作流'; });
$$('[data-tab]').forEach((button, index) => {
  button.tabIndex = index === 0 ? 0 : -1;
  button.addEventListener('click', () => { state.tab = button.dataset.tab; $$('[data-tab]').forEach((item) => { item.classList.toggle('active', item === button); item.setAttribute('aria-selected', String(item === button)); item.tabIndex = item === button ? 0 : -1; }); $('activity-content').setAttribute('aria-labelledby', button.id); renderActivity(); });
  button.addEventListener('keydown', (event) => {
    const tabs = $$('[data-tab]');
    let next = index;
    if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
    else if (event.key === 'ArrowLeft') next = (index + tabs.length - 1) % tabs.length;
    else if (event.key === 'Home') next = 0;
    else if (event.key === 'End') next = tabs.length - 1;
    else return;
    event.preventDefault(); tabs[next].click(); tabs[next].focus({ preventScroll: true });
  });
});
$('event-filter').addEventListener('change', () => { state.filter = $('event-filter').value; renderActivity(); });
$('tool-search').addEventListener('input', renderTools);
$('refresh-files').addEventListener('click', loadFiles);
$('files-list').addEventListener('click', (event) => { const button = event.target.closest('[data-file]'); if (button) previewFile(button.dataset.file); });
$('close-file').addEventListener('click', () => { $('file-preview').hidden = true; });
$('history-list').addEventListener('click', (event) => { const button = event.target.closest('[data-run]'); if (button) { setView('workflow'); selectRun(button.dataset.run); } });
$('jump-live').addEventListener('click', () => { setView('workflow'); if (matchMedia('(max-width: 900px)').matches) setChat(false); });
$('workspace-info').addEventListener('click', showWorkspace);
$('show-setup').addEventListener('click', showSetup);
$('close-dialog').addEventListener('click', () => $('info-dialog').close());
$('info-dialog').addEventListener('click', (event) => { if (event.target === $('info-dialog')) { const rect = $('info-dialog').getBoundingClientRect(); if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) $('info-dialog').close(); } });
$('dialog-content').addEventListener('click', async (event) => { if (event.target.closest('#reload-bootstrap')) { try { await loadBootstrap(); $('info-dialog').close(); toast(state.bootstrap.configured ? '模型信息已读取' : '服务尚未读取到模型配置，请保存配置并重启'); } catch (error) { toast(readableError(error)); } } });
$('chat-messages').addEventListener('scroll', () => { const area = $('chat-messages'); state.autoScroll = area.scrollHeight - area.scrollTop - area.clientHeight < 75; }, { passive: true });
$('chat-messages').addEventListener('click', async (event) => {
  const prompt = event.target.closest('[data-prompt]');
  if (prompt) { $('message-input').value = prompt.dataset.prompt; resizeComposer(); updateComposer(); $('message-input').focus({ preventScroll: true }); return; }
  const approval = event.target.closest('[data-approval]');
  if (approval) { approve(approval.dataset.approval, approval.dataset.approved === 'true'); return; }
  const action = event.target.closest('[data-action]');
  if (!action) return;
  if (action.dataset.action === 'show-tools') { state.filter = 'tools'; $('event-filter').value = 'tools'; $('tab-events').click(); setView('workflow'); if (matchMedia('(max-width: 900px)').matches) setChat(false); return; }
  const text = action.dataset.action === 'copy-code' ? action.closest('.code-block').querySelector('code').textContent : state.session?.messages?.[Number(action.dataset.index)]?.content;
  if (text != null) { try { await navigator.clipboard.writeText(text); toast('已复制'); } catch { toast('浏览器未允许复制，请选择文本手动复制。'); } }
});
document.addEventListener('keydown', (event) => {
  const editing = /INPUT|TEXTAREA|SELECT/.test(document.activeElement?.tagName) || document.activeElement?.isContentEditable;
  if (event.key === 'Escape') { closeSidebar(); $('theme-picker').open = false; closeExpandedGraphs(); }
  if (!editing && !event.ctrlKey && !event.metaKey && !event.altKey && event.key.toLowerCase() === 'n') { event.preventDefault(); newSession(); }
  if (!editing && event.key === '/' && state.currentView === 'tools') { event.preventDefault(); $('tool-search').focus({ preventScroll: true }); }
});
document.addEventListener('click', (event) => { if (!$('theme-picker').contains(event.target)) $('theme-picker').open = false; });

const resizer = $('dock-resizer');
function setDockWidth(width) {
  const maximum = Math.max(320, Math.min(640, innerWidth - 610));
  width = Math.max(320, Math.min(maximum, Number(width) || 382));
  document.documentElement.style.setProperty('--chat-width', `${width}px`);
  resizer.setAttribute('aria-valuenow', String(Math.round(width)));
  resizer.setAttribute('aria-valuemax', String(Math.round(maximum)));
  savePreference('suuclaw.dock-width', String(width));
}
resizer.addEventListener('pointerdown', (event) => {
  if (event.button !== 0) return;
  event.preventDefault(); resizer.setPointerCapture(event.pointerId); document.body.classList.add('resizing');
});
resizer.addEventListener('pointermove', (event) => { if (resizer.hasPointerCapture(event.pointerId)) setDockWidth(innerWidth - event.clientX); });
const endResize = () => document.body.classList.remove('resizing');
resizer.addEventListener('pointerup', endResize); resizer.addEventListener('lostpointercapture', endResize); resizer.addEventListener('pointercancel', endResize);
resizer.addEventListener('keydown', (event) => { if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') { event.preventDefault(); setDockWidth(Number(resizer.getAttribute('aria-valuenow')) + (event.key === 'ArrowLeft' ? 20 : -20)); } });
window.addEventListener('resize', () => { if (innerWidth > 900 && document.documentElement.style.getPropertyValue('--chat-width')) setDockWidth(parseFloat(document.documentElement.style.getPropertyValue('--chat-width'))); });
window.addEventListener('beforeunload', () => { closeStream(); clearGraphMotion(); });
setInterval(() => { if (isRunning()) { renderHeader(); renderExecutionFocus(); architectureView.render(state); if (state.selectedNode) renderInspector(); if (state.approvals.size) renderChat(); } }, 1000);
setInterval(() => { if (state.sessions.some((session) => session.active_run_id)) refreshSessions().catch(() => {}); }, 12000);

async function start() {
  $$('svg.icon').forEach((element) => element.setAttribute('aria-hidden', 'true'));
  applyTheme(readPreference('suuclaw.theme', 'tide'));
  setPerspective(readPreference('suuclaw.perspective', 'architecture'));
  const savedWidth = Number(readPreference('suuclaw.dock-width')); if (savedWidth && innerWidth > 900) setDockWidth(savedWidth);
  if (innerWidth <= 900 || readPreference('suuclaw.chat-open') === 'false') { $('app-shell').classList.add('chat-hidden'); $('chat-toggle').setAttribute('aria-expanded', 'false'); }
  renderActivity(); renderHistory(); updateComposer();
  const results = await Promise.allSettled([loadBootstrap(), refreshSessions()]);
  if (results[0].status === 'rejected') {
    setConnection('offline', '无法连接服务');
    $('graph-loading').hidden = false; $('graph-loading').innerHTML = '<p>工作流未能加载，请确认本地服务仍在运行后刷新页面。</p>';
    $('workspace-label').textContent = '服务未连接'; $('composer-model').lastElementChild.textContent = '服务未连接';
    toast(readableError(results[0].reason));
  }
  if (results[1].status === 'rejected') { setHTML($('sessions-list'), '<p class="sidebar-empty">对话未能读取。<br>点击上方刷新重试。</p>'); }
  else {
    const savedSession = readPreference('suuclaw.session');
    const session = state.sessions.find((item) => item.id === savedSession) || state.sessions[0];
    if (session) await selectSession(session.id);
  }
  state.initializing = false;
  updateComposer();
}
start();
