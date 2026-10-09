/* SuuClaw system map. Structure is static; activity comes only from observed state. */
const NODE_NAMES = { intent_router: 'Intent Router', chat_responder: 'Chat Responder', planner: 'Supervisor', search: 'Search Agent', code: 'Code Agent', verifier: 'Verifier', context_monitor: 'Context Monitor', context_compressor: 'Context Compaction', final: 'Final Response' };
const normalizeNode = (id) => ({ searchAgent: 'search', codeAgent: 'code', search_agent: 'search', code_agent: 'code' }[id] || id || '');
const collection = (value) => Array.isArray(value) ? value : value && typeof value.values === 'function' ? [...value.values()] : value && typeof value === 'object' ? Object.values(value) : [];
const asText = (value) => value == null ? '' : String(value);
const brief = (value, limit = 80) => { const text = asText(value).replace(/\s+/g, ' ').trim(); return text.length > limit ? text.slice(0, limit) + '…' : text; };
const escapeText = (value) => asText(value).replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));
const has = (value, key) => value != null && Object.prototype.hasOwnProperty.call(value, key);
const formatValue = (value) => { if (typeof value === 'string') return value; try { return JSON.stringify(value, null, 2) ?? ''; } catch { return asText(value); } };
const action = (kind, target, label) => ({ kind, target, label });
const fact = (label, value) => ({ label, value: value == null || value === '' ? '尚未观测' : asText(value) });
const section = (title, value) => ({ title, value });
const TERMINAL = new Set(['completed', 'failed', 'interrupted', 'cancelled']);

function observedTools(state, events) {
  const existing = collection(state.tools);
  if (existing.length) return existing.map((tool) => ({ ...tool, node: normalizeNode(tool.node) }));
  const tools = new Map();
  for (const event of events) {
    if (event.type === 'tool_call') {
      const id = event.id || `call-${event.seq}-${tools.size}`;
      tools.set(id, { id, key: id, name: event.name || '工具', node: normalizeNode(event.node), args: event.args, status: 'running', seq: event.seq });
    } else if (event.type === 'tool_result' || event.type === 'search_results') {
      const name = event.name || (event.type === 'search_results' ? 'web_search' : '工具');
      const match = tools.get(event.id) || [...tools.values()].reverse().find((tool) => tool.name === name && tool.status === 'running');
      const id = match?.id || event.id || `result-${event.seq}-${tools.size}`;
      const result = event.type === 'search_results' ? { query: event.query, answer: event.answer, sources: event.sources, result_count: event.result_count, error: event.error } : event.result ?? event.output ?? event.error;
      tools.set(id, { ...match, id, key: id, name, node: normalizeNode(event.node || match?.node), result, status: event.ok === false || event.error ? 'failed' : 'completed' });
    }
  }
  return [...tools.values()];
}

function observedApprovals(state, events) {
  const existing = collection(state.approvals);
  if (existing.length) return existing.map((approval) => ({ ...approval }));
  const approvals = new Map();
  for (const event of events) {
    if (event.type === 'approval_requested') approvals.set(event.approval_id, { ...event, resolved: false });
    if (event.type === 'approval_resolved') approvals.set(event.approval_id, { ...approvals.get(event.approval_id), ...event, resolved: true });
  }
  return [...approvals.values()];
}

/** Pure projection of a dashboard run. It neither starts work nor infers hidden model calls. */
export function deriveArchitecture(state = {}) {
  const events = Array.isArray(state.events) ? state.events : [];
  const run = state.run || null;
  const lastEnd = [...events].reverse().find((event) => event.type === 'run_end');
  const runStatus = lastEnd?.status || run?.status || 'idle';
  const running = runStatus === 'running' && !TERMINAL.has(runStatus);
  const runId = run?.id || run?.run_id || null;
  const eventCount = events.length;
  const nodeRows = collection(state.nodes);
  const activeNodes = running ? nodeRows.filter((node) => node.status === 'running').map((node) => normalizeNode(node.id)).filter(Boolean) : [];
  const nodeObserved = (id) => nodeRows.some((node) => normalizeNode(node.id) === id && node.status !== 'idle') || events.some((event) => normalizeNode(event.node) === id);
  const memoryEvents = events.filter((event) => event.type === 'memory' && event.memory && typeof event.memory === 'object');
  const lastMemory = memoryEvents.at(-1);
  const snapshot = lastMemory?.memory || null;
  const memory = { snapshot, sourceNode: normalizeNode(lastMemory?.node) || null, sequence: lastMemory?.seq ?? null, count: memoryEvents.length };
  const rules = snapshot?.rules;
  const working = snapshot?.working_memory;
  const history = snapshot?.history_summary_store;
  const toolRows = observedTools(state, events);
  const approvalRows = observedApprovals(state, events);
  const pendingTools = toolRows.filter((tool) => ['running', 'pending'].includes(tool.status));
  const tools = {
    available: collection(state.bootstrap?.tools).length,
    total: toolRows.length, observed: toolRows.length,
    succeeded: toolRows.filter((tool) => ['completed', 'succeeded'].includes(tool.status)).length,
    failed: toolRows.filter((tool) => ['failed', 'rejected'].includes(tool.status)).length,
    pending: running ? pendingTools.length : 0,
    rows: toolRows,
  };
  const unresolved = approvalRows.filter((approval) => !approval.resolved);
  const approvals = {
    total: approvalRows.length, pending: running ? unresolved.length : 0,
    approved: approvalRows.filter((approval) => approval.resolved && approval.approved === true).length,
    denied: approvalRows.filter((approval) => approval.resolved && approval.approved === false).length,
    unresolved: unresolved.length, rows: approvalRows,
  };
  const lastApproval = approvalRows.at(-1);
  const started = events.some((event) => event.type === 'run_start') || !!run;
  const final = [...events].reverse().find((event) => event.type === 'final_answer') || state.outcome;
  const answerMessage = collection(state.session?.messages).find((message) => runId && message.run_id === runId && message.role === 'assistant');
  const answer = final?.content || answerMessage?.content || '';
  const verified = nodeObserved('verifier') || collection(state.checks).length > 0 || collection(state.results).length > 0;
  const answerFailed = ['failed', 'interrupted', 'cancelled'].includes(runStatus) || !!final?.error || !!final?.interrupted || final?.passed === false;
  const model = [...events].reverse().find((event) => event.type === 'run_start' && event.model)?.model || state.bootstrap?.model?.name || '未配置模型';
  const configured = !!(state.bootstrap?.model?.configured ?? state.bootstrap?.configured);
  const workspace = state.session?.workspace || state.bootstrap?.workspace || '';
  const runtimeObserved = nodeRows.some((node) => node.id !== '__start__' && node.id !== '__end__' && node.status !== 'idle') || events.some((event) => ['node_start', 'node_output', 'ai_message', 'handoff'].includes(event.type));
  const runtimeFailed = !running && answerFailed && runtimeObserved;
  const runtimeStatus = approvals.pending ? 'waiting' : activeNodes.length ? 'active' : runtimeFailed ? 'failed' : runtimeObserved ? 'observed' : 'static';
  const currentNames = activeNodes.map((id) => NODE_NAMES[id] || id);
  const toolActive = tools.pending > 0 && !approvals.pending;
  const toolStatus = approvals.pending && tools.pending ? 'waiting' : toolActive ? 'active' : tools.failed ? 'failed' : tools.total ? 'observed' : 'static';
  const responseActive = running && !approvals.pending && activeNodes.some((id) => ['chat_responder', 'final'].includes(id));
  const responseStatus = responseActive ? 'active' : answerFailed ? 'failed' : answer ? 'observed' : 'static';
  const compression = activeNodes.includes('context_compressor');
  const observedCompression = events.filter((event) => event.type === 'node_output' && normalizeNode(event.node) === 'context_compressor').at(-1);
  const compressionRows = observedCompression?.data?.compression_events || history?.compression_events || [];
  const workspaceToolActive = toolActive && pendingTools.some((tool) => /file|directory|bash|shell|grep|glob|notepad/i.test(tool.name || ''));
  const activeToolNames = [...new Set(pendingTools.map((tool) => tool.name))].join('、');
  const runtimeSummary = approvals.pending ? '等待操作确认' : toolActive ? `${brief(currentNames[0] || 'Agent', 18)} / 工具执行中` : currentNames.length ? currentNames.join('、') : runtimeObserved ? '查看已观测执行阶段' : 'Supervisor / 专家 / 验证';
  const approvalStatus = approvals.pending ? 'waiting' : lastApproval?.resolved && lastApproval.approved === false ? 'failed' : approvals.total ? 'observed' : 'static';
  const node = (id, title, subtitle, description, status, observed, summary, facts = [], sections = [], actions = []) => ({ id, title, subtitle, description, status, active: status === 'active', observed: !!observed, summary, facts, sections, actions });
  const nodes = [
    node('gateway', 'Gateway', '请求入口', '浏览器发送消息，创建本轮任务，再持续接收执行事件。CLI 与飞书也是项目入口，但它们的运行不会出现在这个浏览器会话里。', activeNodes.includes('intent_router') ? 'active' : started ? 'observed' : 'static', started, 'Browser 会话', [fact('当前观察范围', '本页 Browser 会话'), fact('其他入口', 'CLI、Feishu（本页不监控）'), fact('会话', state.session?.title || state.session?.id)], [], [action('node', 'intent_router', '查看 Intent Router')]),
    node('context', 'Context Assembly', '上下文组装', '在节点构造提示词时，将工作区规则、当前任务状态和历史摘要组装为上下文。这里展示实际收到的 memory 快照；上下文组装没有独立的持续运行信号。', snapshot ? 'observed' : 'static', snapshot, snapshot ? `最近用于 ${NODE_NAMES[memory.sourceNode] || memory.sourceNode}` : '按节点读取三层上下文', [fact('实际快照', memory.count ? `${memory.count} 次组装观测` : '暂无 memory 事件'), fact('最近使用节点', NODE_NAMES[memory.sourceNode] || memory.sourceNode), fact('事件序号', memory.sequence)], snapshot ? [section('最近的上下文快照', snapshot)] : [], [action('view', 'workflow', '查看执行路径')]),
    node('runtime', 'Agent Runtime', '编排与模型交互', 'Intent Router 判断入口路径；Supervisor 编排任务并委派 Search / Code 专家，Verifier 检查结果。这里的活动表示实际节点正在执行，不代表模型提供方正处于推理阶段。', runtimeStatus, runtimeObserved, runtimeSummary, [fact('配置模型', model), fact('当前阶段', approvals.pending ? '等待操作确认' : currentNames.join('、') || (runtimeObserved ? '本轮阶段已结束' : '尚未开始')), fact('观测边界', '未提供模型计费或请求耗时')], [section('当前节点状态', nodeRows.filter((row) => row.id !== '__start__' && row.id !== '__end__').map((row) => ({ node: row.id, status: row.status, visits: row.visits, elapsed_ms: row.elapsed })))], [action('node', 'planner', 'Supervisor / 任务编排'), action('node', 'chat_responder', 'Chat / 直接回复'), action('node', 'search', 'Search / 搜索专家'), action('node', 'code', 'Code / 执行专家'), action('node', 'verifier', 'Verifier / 结果验证')]),
    node('tools', 'Tool Runtime', '工具执行', '接收 Agent 的工具调用，将实际结果交回 Agent。搜索返回、文件读写和命令执行都来自本轮事件；可用工具列表与实际调用记录分开统计。', toolStatus, tools.total > 0, approvals.pending && tools.pending ? '调用暂停，等待确认' : toolActive ? brief(activeToolNames, 29) : tools.total ? `${tools.total} 次调用 / ${tools.failed ? `${tools.failed} 次失败` : `${tools.succeeded} 次成功`}` : '尚无本轮工具调用', [fact('本轮实际调用', tools.total), fact('已完成 / 失败', `${tools.succeeded} / ${tools.failed}`), fact('进行中', tools.pending), fact('注册工具', `${tools.available} 个可用能力`)], toolRows.length ? [section('本轮工具记录', toolRows.map((row) => ({ name: row.name, node: row.node, status: row.status, args: row.args, result: row.result })))] : [], [action('view', 'tools', '打开工具箱')]),
    node('response', 'Response', '回复与结果', '将直接回复或经过任务执行后的最终答复返回对话。运行结束不等于验证通过；失败、中断和未通过的结果会保留原状态。', responseStatus, !!answer || !!final, responseActive ? '正在生成回复' : answerFailed ? '本轮未成功完成' : answer ? '已收到最终回复' : '等待实际答复', [fact('本轮状态', ({ idle: '尚未开始', running: '进行中', completed: '已完成', failed: '失败', interrupted: '已中断', cancelled: '已取消' }[runStatus] || runStatus)), fact('验证结果', !verified ? started ? '本轮未经过独立验证' : '尚未观测验证' : final?.passed === true ? '通过' : final?.passed === false ? '未通过' : '本轮未提供验证结果')], [...(answer ? [section('最终回复', answer)] : []), ...(final?.error || run?.error ? [section('错误信息', final?.error || run.error)] : [])], [action('view', 'history', '查看执行记录')]),
    node('rules', 'Rules', '工作区规则', '规则作为工作区上下文的一部分传入节点。这里只显示本次组装实际提供的规则，不把未配置的规则当作已经生效。', rules ? 'observed' : 'static', rules, rules ? `${Array.isArray(rules.rules) ? rules.rules.length : 0} 条规则 / 工作区作用域` : '等待上下文快照', [fact('作用域', rules?.scope), fact('存储来源', rules?.storage), fact('规则数', Array.isArray(rules?.rules) ? rules.rules.length : null)], rules ? [section('规则快照', rules)] : []),
    node('working', 'Working Memory', '当前任务状态', '保留当前任务、计划、待办、研究结果和验证信息，在节点之间传递。它是工作流状态，不意味着已经写入某个待办文件。', working ? 'observed' : 'static', working, working ? `${Array.isArray(working.todos) ? working.todos.length : 0} 项待办 / 随状态组装` : '任务、计划、研究与验证', [fact('任务', brief(working?.task, 120)), fact('计划', brief(working?.plan_summary, 120)), fact('待办', Array.isArray(working?.todos) ? `${working.todos.length} 项` : null)], working ? [section('任务状态快照', working)] : [], [action('view', 'workflow', '查看计划与执行')]),
    node('history', 'History Summary', '历史摘要与笔记', '读取会话摘要和工作区笔记，为后续节点补充历史上下文。摘要文本可以来自会话组装；是否存在文件与是否有摘要文本分别展示。压缩记录也不等于持久化成功。', compression ? 'active' : history ? 'observed' : 'static', history, compression ? '正在压缩上下文' : history ? '摘要 / 笔记 / 压缩记录' : '按需读取与上下文压缩', [fact('历史摘要文件', history ? `${history.history_path || '路径未提供'} / ${history.history_exists === true ? '观测到存在' : history.history_exists === false ? '观测时不存在' : '未观测存在状态'}` : null), fact('摘要文本', history ? history.history_summary ? '快照含摘要文本' : '快照中为空' : null), fact('笔记文件', history ? `${history.notepad_path || '路径未提供'} / ${history.notepad_exists === true ? '观测到存在' : history.notepad_exists === false ? '观测时不存在' : '未观测存在状态'}` : null)], [...(history ? [section('历史摘要快照', history)] : []), ...(compressionRows.length ? [section('压缩记录（含原始说明）', compressionRows)] : [])], [action('node', 'context_compressor', '查看 Context Compaction'), action('view', 'files', '查看工作文件')]),
    node('trace', 'Trace', '事件记录', '本轮事件按顺序进入工作台，支持执行记录回看。这里统计前端已经接收的事件，不据此推断模型用量。', eventCount ? 'observed' : 'static', eventCount > 0, eventCount ? `已观测 ${eventCount} 条事件` : '按序记录与历史回看', [fact('本轮已观测事件', eventCount), fact('最后序号', events.at(-1)?.seq), fact('最新事件', events.at(-1)?.type)], events.length ? [section('最近事件', events.slice(-5))] : [], [action('view', 'history', '打开执行记录')]),
    node('checkpoints', 'Checkpoints', '恢复检查点', '核心运行时支持检查点与恢复。检查点写入目前属于内部记录，未通过本页事件接口上报，因此这里展示系统能力，不判断某一轮是否已经保存。', 'static', false, '保存状态未观测', [fact('支持能力', '运行状态检查点 / 恢复'), fact('本页观测', '未提供检查点保存事件')]),
    node('workspace', 'Workspace', '会话工作区', '工具在会话工作目录中读写文件。工作文件视图列出实际发现的文件；显示工作目录不代表本轮已经产生文件。', workspaceToolActive ? 'active' : state.session?.workspace ? 'observed' : 'static', !!state.session?.workspace, workspaceToolActive ? '工作区工具正在执行' : state.session?.workspace ? '当前会话独立工作区' : '文件、命令与执行产物', [fact('当前目录', workspace), fact('范围', state.session?.workspace ? '当前 Dashboard 会话' : '工作台配置目录')], [], [action('view', 'files', '打开工作文件')]),
    node('approval', 'Approval', '操作确认', '需要确认的工具操作会暂停，直到收到实际决议。确认在对话面板中完成；拒绝操作会保留为拒绝，不会被标成执行成功。', approvalStatus, approvals.total > 0, approvals.pending ? `${approvals.pending} 项等待确认` : lastApproval?.resolved && lastApproval.approved === false ? '最近操作未获准' : approvals.approved ? `${approvals.approved} 项已获准` : approvals.unresolved ? '本轮已结束 / 决议未观测' : '按需确认操作', [fact('等待确认', approvals.pending), fact('已获准 / 已拒绝', `${approvals.approved} / ${approvals.denied}`), fact('最近操作', brief(lastApproval?.command, 160))], approvalRows.length ? [section('操作确认记录', approvalRows)] : []),
  ];
  const nodeById = Object.fromEntries(nodes.map((item) => [item.id, item]));
  const results = events.filter((event) => ['tool_result', 'search_results'].includes(event.type));
  const lastLoop = [...events].reverse().find((event) => ['tool_call', 'tool_result', 'search_results', 'ai_message', 'handoff', 'node_start', 'node_output', 'node_summary'].includes(event.type));
  const observeActive = running && !approvals.pending && !tools.pending && !!lastLoop && ['tool_result', 'search_results'].includes(lastLoop.type) && activeNodes.includes(normalizeNode(lastLoop.node));
  const edge = (id, source, target, observed, active = false, waiting = false, failed = false) => ({ id, source, target, observed: !!observed, active: running && !!active, status: running && waiting ? 'waiting' : running && active ? 'active' : failed ? 'failed' : observed ? 'observed' : 'static' });
  const edges = [
    edge('request', 'gateway', 'context', started, activeNodes.includes('intent_router')),
    edge('assemble', 'context', 'runtime', snapshot),
    edge('answer', 'runtime', 'response', !!answer, responseActive, false, !!answer && answerFailed),
    edge('reply', 'response', 'gateway', !!answer, false, false, !!answer && answerFailed),
    edge('act', 'runtime', 'tools', tools.total > 0, toolActive, !!approvals.pending),
    edge('observe', 'tools', 'runtime', results.length > 0, observeActive, false, lastLoop?.type === 'tool_result' && lastLoop?.ok === false),
    edge('confirm', 'tools', 'approval', approvals.total > 0, false, !!approvals.pending),
    edge('decision', 'approval', 'tools', approvals.approved + approvals.denied > 0, false, false, lastApproval?.resolved && lastApproval.approved === false),
    edge('rules-context', 'rules', 'context', !!rules),
    edge('working-context', 'working', 'context', !!working),
    edge('history-context', 'history', 'context', !!history),
    edge('memory-context', 'working', 'context', snapshot),
    edge('events', 'runtime', 'trace', eventCount > 0),
    edge('files', 'tools', 'workspace', toolRows.some((tool) => /file|directory|bash|shell|grep|glob|notepad/i.test(tool.name || '')), workspaceToolActive),
  ];
  return { running, runId, runStatus, eventCount, nodes, nodeById, edges, memory, tools, approvals, runtime: { activeNodes, model, configured }, workspace };
}

const LAYOUT = {
  gateway: { x: 24, y: 102, width: 118, height: 86, kind: 'primary' },
  context: { x: 169, y: 102, width: 178, height: 86, kind: 'primary' },
  runtime: { x: 383, y: 102, width: 190, height: 90, kind: 'primary' },
  response: { x: 606, y: 102, width: 123, height: 86, kind: 'primary' },
  tools: { x: 383, y: 250, width: 190, height: 80, kind: 'primary' },
  approval: { x: 606, y: 250, width: 123, height: 80, kind: 'support' },
  rules: { x: 40, y: 456, width: 200, height: 106, kind: 'memory' },
  working: { x: 260, y: 456, width: 218, height: 106, kind: 'memory' },
  history: { x: 498, y: 456, width: 216, height: 106, kind: 'memory' },
  trace: { x: 769, y: 111, width: 143, height: 104, kind: 'service' },
  checkpoints: { x: 769, y: 283, width: 143, height: 104, kind: 'service' },
  workspace: { x: 769, y: 458, width: 143, height: 104, kind: 'service' },
};
const PATHS = {
  request: 'M142 141H169', assemble: 'M347 141H383', answer: 'M573 141H606',
  reply: 'M668 102V51Q668 40 657 40H94Q83 40 83 51V102',
  act: 'M430 192V250', observe: 'M525 250V192',
  confirm: 'M573 273H606', decision: 'M606 309H573',
  'rules-context': 'M140 456V393Q140 380 153 380H258',
  'working-context': 'M369 456V393Q369 380 356 380H258',
  'history-context': 'M606 456V393Q606 380 593 380H258',
  'memory-context': 'M258 380V188',
  events: 'M573 174H581Q590 174 590 184V217Q590 226 601 226H741Q751 226 751 214V173H769',
  files: 'M573 322H589Q599 322 599 335V344Q599 358 612 358H737Q751 358 751 372V510H769',
};
const STATUS_NAMES = { static: '系统能力', observed: '已有观测', active: '正在执行', waiting: '等待确认', failed: '查看异常' };
const MARKER_COLORS = { static: '--edge', observed: '--success', active: '--accent', waiting: '--warning', failed: '--danger' };
let instanceCounter = 0;

/** A local, keyboard-operable diagram with its own inspector. No document scrolling. */
export class ArchitectureView {
  constructor(container, { onInspectNode, onOpenView } = {}) {
    if (!container) throw new Error('ArchitectureView requires a container.');
    this.container = container;
    this.onInspectNode = onInspectNode;
    this.onOpenView = onOpenView;
    this.selected = null;
    this.runKey = undefined;
    this.data = null;
    this.detailHTML = '';
    this.prefix = `suu-architecture-${++instanceCounter}`;
    this.handleClick = (event) => {
      const close = event.target.closest('[data-architecture-close]');
      if (close) { this.closeInspector(true); return; }
      const actionButton = event.target.closest('[data-architecture-action]');
      if (actionButton) {
        const current = this.data?.nodeById[this.selected];
        const chosen = current?.actions[Number(actionButton.dataset.architectureAction)];
        if (chosen?.kind === 'node') this.onInspectNode?.(chosen.target);
        if (chosen?.kind === 'view') this.onOpenView?.(chosen.target);
        return;
      }
      const node = event.target.closest('[data-architecture-node]');
      if (node) this.inspect(node.dataset.architectureNode);
    };
    this.handleKey = (event) => {
      if (event.key === 'Escape' && this.selected) { event.preventDefault(); this.closeInspector(true); }
      if (event.key === 'Enter' || event.key === ' ') {
        const node = event.target.closest('[data-architecture-node]');
        if (node) { event.preventDefault(); this.inspect(node.dataset.architectureNode); }
      }
    };
    container.classList.add('suu-architecture');
    container.innerHTML = this.shell();
    this.stage = container.querySelector('.architecture-stage');
    this.inspector = container.querySelector('.architecture-inspector');
    this.inspectorBody = container.querySelector('.architecture-inspector-body');
    this.scope = container.querySelector('[data-architecture-scope]');
    this.model = container.querySelector('[data-architecture-model]');
    this.nodeElements = new Map([...container.querySelectorAll('[data-architecture-node]')].map((element) => [element.dataset.architectureNode, element]));
    this.edgeElements = new Map([...container.querySelectorAll('[data-architecture-edge]')].map((element) => [element.dataset.architectureEdge, element]));
    container.addEventListener('click', this.handleClick);
    container.addEventListener('keydown', this.handleKey);
  }

  shell() {
    const empty = deriveArchitecture();
    const markers = Object.entries(MARKER_COLORS).map(([status, variable]) => `<marker id="${this.prefix}-${status}" viewBox="0 0 12 12" markerWidth="${status === 'active' ? 13 : 11}" markerHeight="${status === 'active' ? 13 : 11}" refX="10.5" refY="6" orient="auto" markerUnits="userSpaceOnUse"><path d="M1 1L11 6L1 11Z" style="fill:var(${variable})"/></marker>`).join('');
    const edges = empty.edges.map((edge) => `<g class="architecture-edge" data-architecture-edge="${edge.id}" data-status="static" data-active="false"><path class="architecture-edge-line" d="${PATHS[edge.id]}" ${edge.id.endsWith('-context') && edge.id !== 'memory-context' ? '' : `marker-end="url(#${this.prefix}-static)"`}/><path class="architecture-edge-travel" d="${PATHS[edge.id]}" pathLength="100"/></g>`).join('');
    const nodes = empty.nodes.map((node) => {
      const box = LAYOUT[node.id];
      const side = ['service', 'support'].includes(box.kind);
      const titleY = side ? 29 : 28;
      return `<g class="architecture-node architecture-node-${box.kind}" data-architecture-node="${node.id}" data-status="static" data-active="false" role="button" tabindex="0" aria-label="${node.title}，${node.subtitle}，查看详情" aria-expanded="false" transform="translate(${box.x} ${box.y})"><title>${node.title} / ${node.subtitle}</title><rect class="architecture-node-halo" x="-4" y="-4" width="${box.width + 8}" height="${box.height + 8}"/><rect class="architecture-node-surface" width="${box.width}" height="${box.height}"/><circle class="architecture-node-dot" cx="${box.width - 13}" cy="13" r="3"/><text class="architecture-node-title" x="14" y="${titleY}">${node.title}</text><text class="architecture-node-subtitle" x="14" y="${titleY + 21}">${node.subtitle}</text><text class="architecture-node-summary" x="14" y="${box.height - 14}" data-architecture-summary>${escapeText(node.summary)}</text></g>`;
    }).join('');
    return `<div class="architecture-scope"><span><i aria-hidden="true"></i><span data-architecture-scope>Browser 会话 / 系统结构</span></span><span class="architecture-model" data-architecture-model></span></div><div class="architecture-stage"><div class="architecture-canvas-scroll" role="region" aria-label="系统全景图，可选择模块查看详情" tabindex="0"><svg class="architecture-svg" viewBox="0 0 940 608" role="group" aria-labelledby="${this.prefix}-title ${this.prefix}-description"><title id="${this.prefix}-title">SuuClaw Agent Architecture 系统全景</title><desc id="${this.prefix}-description">请求从 Browser 入口进入上下文组装和 Agent Runtime，再返回回复。Agent 与工具通过 Act 和 Observe 循环工作，三层记忆提供上下文，运行服务记录事件并管理工作区。选择模块查看真实数据。</desc><defs>${markers}</defs><g class="architecture-groups" aria-hidden="true"><rect class="architecture-loop-group" x="368" y="66" width="220" height="282"/><text class="architecture-group-title" x="384" y="87">Agent / Tool Loop</text><rect class="architecture-memory-group" x="24" y="407" width="705" height="182"/><text class="architecture-group-title" x="40" y="431">Layered Memory</text><text class="architecture-group-subtitle" x="202" y="431">三层上下文</text><rect class="architecture-services-group" x="753" y="66" width="175" height="523"/><text class="architecture-group-title" x="769" y="87">Runtime Services</text><path class="architecture-service-link" d="M840 229V268M840 401V444"/></g><g aria-hidden="true">${edges}<text class="architecture-link-label" x="350" y="30" text-anchor="middle">reply to browser / 返回回复</text><text class="architecture-loop-label" x="420" y="221" text-anchor="end">Act</text><text class="architecture-loop-label" x="536" y="221">Observe</text><text class="architecture-link-label" x="273" y="319">assemble</text><text class="architecture-link-note" x="273" y="338">按节点组装</text><text class="architecture-link-label" x="685" y="218" text-anchor="middle">events</text><text class="architecture-link-label" x="676" y="350" text-anchor="middle">workspace I/O</text><g class="architecture-other-entry"><rect x="24" y="205" width="47" height="23"/><text x="47.5" y="221" text-anchor="middle">CLI</text><rect x="78" y="205" width="64" height="23"/><text x="110" y="221" text-anchor="middle">Feishu</text><text class="architecture-link-note" x="25" y="247">其他入口 / 本页不监控</text></g></g><g>${nodes}</g></svg></div><aside class="architecture-inspector" aria-label="系统模块详情" hidden><button class="architecture-inspector-close" type="button" aria-label="关闭模块详情" data-architecture-close><svg viewBox="0 0 20 20" aria-hidden="true"><path d="M5 5L15 15M15 5L5 15"/></svg></button><div class="architecture-inspector-body"></div></aside></div><div class="architecture-legend"><span><i class="is-static"></i>系统关系</span><span><i class="is-observed"></i>本轮已观测</span><span><i class="is-active"></i>实际活动</span><span><i class="is-waiting"></i>等待确认</span><span><i class="is-failed"></i>异常 / 拒绝</span><small>选择模块查看职责与数据<span class="architecture-mobile-hint">，横向滑动查看全图</span></small></div>`;
  }

  render(state = {}) {
    const data = deriveArchitecture(state);
    const key = `${state.session?.id || ''}/${data.runId || ''}`;
    if (this.runKey !== key) { this.closeInspector(false); this.runKey = key; }
    this.data = data;
    this.container.dataset.running = String(data.running);
    this.scope.textContent = data.running ? data.approvals.pending ? 'Browser 会话 / 等待操作确认' : 'Browser 会话 / 正在观察本轮任务' : data.runId ? 'Browser 会话 / 本轮历史观测' : 'Browser 会话 / 系统结构';
    this.model.textContent = brief(data.runtime.model, 36);
    this.model.title = data.runtime.model;
    for (const node of data.nodes) {
      const element = this.nodeElements.get(node.id);
      element.dataset.status = node.status;
      element.dataset.active = String(node.active);
      element.setAttribute('aria-expanded', String(this.selected === node.id));
      element.setAttribute('aria-label', `${node.title}，${node.subtitle}，${STATUS_NAMES[node.status]}，${node.summary}。查看详情`);
      const available = LAYOUT[node.id].width - 28;
      const max = Math.floor(available / 6.0);
      // Fit mixed scripts deterministically; full values remain in the inspector and title.
      let measured = 0, summary = '';
      for (const char of node.summary) { const width = char.charCodeAt(0) > 255 ? 1.7 : .95; if (measured + width > max - 1) { summary += '…'; break; } summary += char; measured += width; }
      element.querySelector('[data-architecture-summary]').textContent = summary;
      element.querySelector('title').textContent = `${node.title} / ${node.subtitle}\n${node.summary}`;
    }
    for (const edge of data.edges) {
      const element = this.edgeElements.get(edge.id);
      element.dataset.status = edge.status;
      element.dataset.active = String(edge.active);
      const path = element.querySelector('.architecture-edge-line');
      if (path.hasAttribute('marker-end')) path.setAttribute('marker-end', `url(#${this.prefix}-${edge.status})`);
    }
    if (this.selected) this.renderInspector();
  }

  inspect(id) {
    if (!this.data?.nodeById[id]) return;
    this.selected = id;
    this.inspector.hidden = false;
    for (const [nodeId, element] of this.nodeElements) element.setAttribute('aria-expanded', String(nodeId === id));
    this.renderInspector();
    this.container.querySelector('[data-architecture-close]').focus({ preventScroll: true });
  }

  closeInspector(restoreFocus = false) {
    const previous = this.selected;
    this.selected = null;
    if (this.inspector) this.inspector.hidden = true;
    if (this.nodeElements) for (const element of this.nodeElements.values()) element.setAttribute('aria-expanded', 'false');
    if (restoreFocus && previous) this.nodeElements.get(previous)?.focus({ preventScroll: true });
  }

  renderInspector() {
    const node = this.data?.nodeById[this.selected];
    if (!node) return;
    const html = `<div class="architecture-detail-heading"><span class="architecture-detail-status" data-status="${node.status}"><i></i>${STATUS_NAMES[node.status]}</span><h3>${escapeText(node.title)}</h3><p>${escapeText(node.subtitle)}</p></div><p class="architecture-detail-description">${escapeText(node.description)}</p><dl class="architecture-facts">${node.facts.map((item) => `<div><dt>${escapeText(item.label)}</dt><dd>${escapeText(item.value)}</dd></div>`).join('')}</dl>${node.actions.length ? `<div class="architecture-actions">${node.actions.map((item, index) => `<button type="button" data-architecture-action="${index}">${escapeText(item.label)}<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3 8H13M9 4L13 8L9 12"/></svg></button>`).join('')}</div>` : ''}${node.sections.length ? `<div class="architecture-raw">${node.sections.map((item, index) => { const raw = formatValue(item.value); const value = raw.length > 24000 ? raw.slice(0, 24000) + '\n…此处预览已截断' : raw; return `<details data-architecture-section="${index}"><summary>${escapeText(item.title)}</summary><pre>${escapeText(value)}</pre></details>`; }).join('')}</div>` : ''}`;
    if (this.detailHTML === html) return;
    const opened = new Set([...this.inspectorBody.querySelectorAll('details[open]')].map((item) => item.dataset.architectureSection));
    const currentAction = this.inspectorBody.contains(document.activeElement) ? document.activeElement.dataset.architectureAction : null;
    const top = this.inspector.scrollTop;
    this.inspectorBody.innerHTML = html;
    this.detailHTML = html;
    this.inspectorBody.querySelectorAll('details').forEach((item) => { item.open = opened.has(item.dataset.architectureSection); });
    this.inspector.scrollTop = top;
    if (currentAction != null) this.inspectorBody.querySelector(`[data-architecture-action="${currentAction}"]`)?.focus({ preventScroll: true });
  }

  destroy() {
    this.container.removeEventListener('click', this.handleClick);
    this.container.removeEventListener('keydown', this.handleKey);
    this.container.replaceChildren();
  }
}
