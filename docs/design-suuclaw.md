# SuuClaw interface direction

SuuClaw is a local agent workbench for understanding what a task is doing while
keeping the conversation in reach. Its expressive element is an original curled
S creature: three small claws catch an idea, an eye gives it attention, and the
warm dot suggests the idea it carries. The SVG remains legible at favicon size.
No visual assets or proprietary design tokens are taken from Waku.

## Plan and critique

The layout is left aligned and organized around one shared run, not equal-sized
metric cards. Navigation and past conversations sit left, the actual execution
graph occupies the center, and a resizable conversation dock stays on the right.

```text
┌────────────┬─────────────────────────────┬─────────────────────┐
│ SuuClaw    │ Workbench / Workflow        │ SuuClaw conversation│
│ New chat   ├─────────────────────────────┤                     │
│ Workflow   │ Task and real run status    │ User request        │
│ Tools      │                             │ Assistant responses │
│ Files      │ Actual execution graph      │ Tool activity       │
│ History    │                             │ Approval controls   │
│            │ Selected node inspection    │                     │
│ Sessions   ├─────────────────────────────┤                     │
│            │ Events / Plan / Verification│ Task composer       │
│ Theme      │                             │ Local model status  │
└────────────┴─────────────────────────────┴─────────────────────┘
```

Three styles change color, typography, shape, and density:

| Style | Palette | Typography and shape |
| --- | --- | --- |
| 潮汐 Tide | `#edf4f7` `#ffffff` `#173849` `#2b8394` `#f5ac7c` | Segoe UI / Microsoft YaHei; airy, medium-radius surfaces |
| 深航 Abyss | `#0c1d32` `#122943` `#cfe9f4` `#68d4dd` `#dfa781` | Bahnschrift / DengXian; compact, precise corners, technical canvas |
| 花间 Bloom | `#f4eff7` `#fffafb` `#48314d` `#936c9d` `#dc9a83` | Georgia / STSong headings, Segoe UI body; soft rounded shapes |

The first plan had several summary cards and a decorative demo run. Both were
removed: they would make an empty local tool look busy without helping someone
understand an actual task. The static graph is honestly labelled as structure;
only received execution events can mark a node active or completed. The brand
mark supplies character while the data surfaces stay quiet.

Keyboard focus, reduced motion, a mobile drawer, accessible node buttons, and
legible status labels are part of the foundation. Themes and conversation dock
width are persisted locally. Raw model and tool text is escaped before rendering;
file previews use text content, never execute returned HTML.

## Runtime sources

- `/api/bootstrap`: configured model, tool registry, actual graph topology.
- `/api/sessions`: saved conversations, messages and run metadata.
- `/api/runs/{id}`: historical event replay.
- `/api/runs/{id}/events?after=N`: sequenced SSE with reconnection replay.
- Session file endpoints: bounded previews and downloads of real files.

The interface has no invented token cost, completion percentage, activity or
example history. All sample prompts are explicitly invitations to send a task.


## Workflow diagram revision

The original diagram made every box and return edge equally prominent. That
forced the reader to decode a graph before they could understand the task.
Waku's useful lesson is information hierarchy: groups, short primary routes,
explicit loop scope, secondary descriptions, and runtime detail on demand.
The revised SuuClaw diagram is original SVG; no Waku design assets were copied.

The execution perspective is a readable 840 × 610 composition: Intent Router and direct
chat across the top, Supervisor → Context Monitor → Verifier in the work group,
ReAct specialists nested beneath their supervisor, and Context Compaction in a
separate optional branch. English AI terms identify the component; Chinese text
explains its purpose. Actual runtime state occupies its own third line.

The main-path view reduces unused retry/restore routes. Every actual edge remains
available in the complete-topology view and the connection table. Visited,
currently traversed, and selected-node routes are visible even in the main view.
Geometry is presentation only: nodes and connections continue to come from the
compiled graphs. Specialist returns reuse the delegation line in reverse and do
not invent a LangGraph edge or imply simultaneous specialist execution.

Filled arrowheads match route state. Active nodes remain lit while executing;
moving segments make direction clear. A bounded queue provides brief amber
recent-travel feedback for fast transitions without delaying real node completion.
Historical loading never queues this feedback. Reduced-motion removes travel
animation while preserving state colors, arrowheads and text labels.

The run summary and actual-path breadcrumb answer where execution is now and how
it arrived. Inspection separates recorded task/delegation input, actual output,
routing and tool calls. The interface explicitly distinguishes task input from a
full model context, which the event stream does not contain.

Supervisor resumes within one superstep do not increment its visit counter.
Delegated work marks it waiting; its elapsed duration includes that waiting time.
Search results close their matching web_search call; code results read the actual
output field. Failed verification and interrupted final states remain visibly
different from successful completion.


## System overview and execution are different scopes

Waku separates its harness diagram (`waku/ops/static/js/diagram.js`) from the
compiled execution graph (`graph.js`). Our first revision followed the latter,
so even its complete topology omitted the wider relationships that explain
where context comes from and what supports each turn.

The workbench now defaults to Architecture, with Workflow as an explicit
second perspective on the same selected run. The overview groups Gateway,
Context Assembly, Agent Runtime, Tool Runtime, Approval and Response around
the act/observe loop. Layered Memory supplies the rules, current task state and
history summary. Trace, Checkpoints and Workspace form a separate supporting
area. These groups encode responsibility; they are not additional LangGraph
execution nodes. Workflow still reads nodes and edges from the compiled backend.

The existing three theme palettes and fonts remain the token system. A directed
main route and a distinct tool-return route carry the visual emphasis; groups
and bilingual labels explain scope. The overview does not compress a second
copy of all workflow nodes into another unreadable graph. Selecting a module
reveals recorded data or a link to the detailed execution perspective.

Runtime fidelity:

- Real memory events have type `memory` and a `memory` payload containing
  `rules`, `working_memory` and `history_summary_store`. They are displayed as
  context assembly and can be filtered in activity. The previous UI only named
  `memory_snapshot`, leaving real events unexplained.
- Working memory is rebuilt for model prompts. The rules layer is fixed policy,
  not procedural skill retrieval. History includes compressed context and actual
  notepad/history file records, not a semantic or episodic retrieval database.
- Dashboard run events describe Browser-originated work only. CLI and Feishu
  exist as entrypoints, but this UI does not aggregate their current executions.
- Tool call/result pairs and approval events drive current activity. A historical
  or terminal run never animates an unfinished tool as a live operation.
- Dashboard writes events before sending them, so the observed event count is
  evidence of its trace. Core checkpoint/session save receipts are not streamed
  to this UI; their presence in the architecture is a capability, not a claim
  that a particular save succeeded.
- The configured model name is available; provider usage, costs and request
  latency are not. Context budget estimates must not be presented as billing.

Waku's retrieval gate, FTS5 semantic store, episodic/procedural memory,
consolidation and Evals/Release loop are not equivalent SuuClaw features. They
are intentionally absent from this runtime diagram. The existing per-task
Verifier remains available in Workflow and is not labelled as an offline eval
or release gate.

Reference source: https://github.com/ShenSeanChen/waku-agent/tree/main/waku/ops/static/js
