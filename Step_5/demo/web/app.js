"use strict";
// HardLLM demo page. All numbers come from /api/story (server.py reads them
// from the steps' output files); answers stream from /api/ask.

const $ = (s, el = document) => el.querySelector(s);
const el = (tag, cls, text) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
};
const fmt = (x, d = 2) => Number(x).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
const int = (x) => Number(x).toLocaleString("en-US");
const pct = (x, d = 1) => fmt(100 * x, d) + "%";
const wrap = (table) => {
  const w = el("div", "table-wrap");
  w.append(table);
  return w;
};
const short = (q, n = 90) => (q.length > n ? q.slice(0, n).replace(/\s+\S*$/, "") + "…" : q);
const ICON_PLUS = '<svg viewBox="0 0 32 32"><path d="M13 7h6v6h6v6h-6v6h-6v-6H7v-6h6z"/></svg>';
const ICON_UNDO = '<svg viewBox="0 0 24 24"><path d="M9 5 3 11l6 6v-4h6a4 4 0 0 1 0 8h-2v3h2a7 7 0 0 0 0-14H9z"/></svg>';
const VERSION_NAME = { q4_k_m: "4-bit", q8_0: "8-bit", f16: "16-bit" };

let S = null;          // the story
let busy = false;      // one answer at a time

// ---------- theme ----------
(function theme() {
  try {
    const saved = localStorage.getItem("hardllm-theme");
    if (saved) document.documentElement.dataset.theme = saved;
  } catch (e) { /* storage blocked: follow the OS */ }
  $("#theme").addEventListener("click", () => {
    const dark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === "dark"
      : matchMedia("(prefers-color-scheme: dark)").matches;
    const next = dark ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("hardllm-theme", next); } catch (e) { /* fine */ }
  });
})();

// ---------- tabs ----------
function showTab(name) {
  document.querySelectorAll(".tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === name)));
  document.querySelectorAll(".panel").forEach((p) => (p.hidden = p.id !== "tab-" + name));
  history.replaceState(null, "", "#" + name);
  if (name === "queue") refreshQueue();
}
document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));

// ---------- streaming ----------
async function ask(question, onPiece, onAlert = () => {}, signal = undefined, filter = true) {
  const res = await fetch("/api/ask", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question, filter }), signal,
  });
  if (!res.ok) throw new Error("server said " + res.status);
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "", done = null;
  for (;;) {
    const { value, done: end } = await reader.read();
    if (end) break;
    buf += dec.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf("\n\n")) >= 0) {
      const line = buf.slice(0, i).trim();
      buf = buf.slice(i + 2);
      if (!line.startsWith("data: ")) continue;
      const ev = JSON.parse(line.slice(6));
      if (ev.done) done = ev;
      else if (ev.alert) onAlert(ev.alert);
      else onPiece(ev.t);
    }
  }
  return done || {};
}

function metaLine(ev, ms) {
  const hidden = ev.dropped ? [["", ev.dropped + (ev.dropped === 1 ? " advice sentence" : " advice sentences") + " hidden: information only"]] : [];
  if (ev.canned) return [["", "Fixed reply: a greeting, so the model was not asked"]];
  if (ev.mock) return [["warn", "Preview mode: recorded answer, no live model"], ["", "matched demo question: “" + short(ev.matched, 60) + "”"], ...hidden];
  const v = VERSION_NAME[S.version] || S.version;
  return [["accent", v + " model · on this laptop"], ["", int(ev.tokens) + " tokens"], ["", fmt(ev.tps, 1) + " tokens/s"], ["", fmt(ms / 1000, 1) + " s"], ...hidden];
}

function renderMeta(box, ev, ms) {
  box.replaceChildren(...metaLine(ev, ms).map(([kind, text]) => el("span", kind ? "badge " + kind : "", text)));
}

// ---------- ask tab ----------
function addUser(text) {
  const m = el("div", "msg user");
  m.append(el("div", "bubble", text));
  $("#thread").append(m);
}

function addBot() {
  const m = el("div", "msg bot");
  const av = el("div", "avatar");
  av.innerHTML = ICON_PLUS;
  const b = el("div", "bubble");
  const text = el("span", "text");
  const caret = el("span", "caret");
  const meta = el("div", "meta");
  b.append(text, caret, meta);
  m.append(av, b);
  $("#thread").append(m);
  return { bubble: b, text, caret, meta };
}

async function submit(q) {
  q = q.trim();
  if (!q || busy) return;
  busy = true;
  $("#send").disabled = true;
  $("#empty")?.remove();
  addUser(q);
  const bot = addBot();
  bot.text.scrollIntoView({ block: "nearest" });
  const t0 = performance.now();
  try {
    const ev = await ask(q, (piece) => {
      bot.text.textContent += piece;
      if (innerHeight + scrollY >= document.body.scrollHeight - 160) scrollTo(0, document.body.scrollHeight);
    }, (alert) => {
      const a = el("div", "emergency", alert);
      a.setAttribute("role", "alert");
      bot.bubble.prepend(a);
    }, undefined, false);
    bot.text.textContent = bot.text.textContent.trim();
    if (MEDICINE.test(bot.text.textContent)) bot.bubble.insertBefore(el("div", "caution", MEDICINE_NOTE.replace("The AI note", "This answer")), bot.text);
    bot.bubble.insertBefore(el("p", "ai-disclaimer", AI_DISCLAIMER), bot.meta);
    renderMeta(bot.meta, ev, performance.now() - t0);
  } catch (e) {
    bot.text.textContent = "Could not reach the model (" + e.message + "). Is server.py still running?";
  } finally {
    bot.caret.remove();
    busy = false;
    $("#send").disabled = false;
    $("#input").focus();
  }
}

const input = $("#input");
input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 180) + "px";
});
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    $("#composer").requestSubmit();
  }
});
$("#composer").addEventListener("submit", (e) => {
  e.preventDefault();
  const q = input.value;
  input.value = "";
  input.style.height = "auto";
  submit(q);
});

function renderAsk() {
  const picks = [0, 3, 6, 9, 12, 15].filter((i) => i < S.demo.length);
  $("#chips").replaceChildren(...picks.map((i) => {
    const c = el("button", "chip", short(S.demo[i].question, 70));
    c.type = "button";
    c.title = S.demo[i].question;
    c.addEventListener("click", () => submit(S.demo[i].question));
    return c;
  }));

  const m = Object.fromEntries(S.models.map((x) => [x.id, x]));
  const card = $("#model-card");
  card.replaceChildren();
  card.append(el("h3", "", "Running now"));
  card.append(el("div", "big", "Llama-3.2-3B, HardLLM final model"));
  card.append(el("p", "", "Fine-tuned on public Q&A chosen by the private records (Step 2), then compressed for the device."));
  const kv = el("dl", "kv");
  const v = S.step5.find((x) => x.version === S.version);
  [["Version", VERSION_NAME[S.version] || S.version], ["File size", fmt(S.file_gb, 2) + " GB"],
   ["Runs on", "this laptop, offline"], ["Engine", "llama.cpp"], ["Answer perplexity", fmt(v.ppl, 2)]]
    .forEach(([k, val]) => kv.append(el("dt", "", k), el("dd", "", val)));
  card.append(kv);

  const vs = el("div", "vs");
  vs.append(el("h3", "", "Lower is better"));
  const rows = [["Base Llama", m.base.ppl, false], ["Step 3 data", m.hardllm.ppl, false], ["This model", v.ppl, true]];
  const max = Math.max(...rows.map((r) => r[1]));
  rows.forEach(([name, ppl, hi]) => {
    const r = el("div", "vs-row" + (hi ? " hi" : ""));
    const bar = el("div", "vs-bar");
    bar.style.width = (100 * ppl) / max + "%";
    r.append(el("span", "", name), el("span", "", fmt(ppl, 2)), bar);
    vs.append(r);
  });
  card.append(vs);
}

// ---------- intake tab ----------
const LEVEL_NAME = { emergency: "Emergency", urgent: "Urgent", review: "Needs review", routine: "Routine" };
const INTAKE_EXAMPLES = [
  "I am having sharp pain in my left chest and I am unable to breathe",
  "My 2 year old has had a fever of 104 for three days",
  "Burning sensation when I urinate since yesterday",
  "Itchy red rash on my arm for a week, no fever",
];
// reassurance that should not sit under an urgent result unchallenged
const REASSURING = /no need to (?:worry|panic)|nothing (?:serious|to worry)|don'?t worry|not (?:a )?serious|need not worry|probably (?:a |just )?(?:viral|nothing|normal|harmless|minor)|(?:does ?n[o']t|do ?n[o']t) (?:require|need) (?:any )?(?:treatment|antibiotics|medicines?)|will (?:settle|go away|resolve) (?:on its own|by itself)|self[- ]limiting/i;
// a dose, or an instruction to take a medicine: the model writes these with no examination
const MEDICINE = /\b\d+(?:\.\d+)?\s?(?:mg|mcg|µg|ml|g|iu|units?)\b|\b(?:take|taking|start|give|prescribe[ds]?|use)\s+(?:\w+\s+){0,4}?(?:tablets?|capsules?|syrup|injections?|antibiotics?|antihistamines?|antacids?|painkillers?|analgesics?|inhibitors?|relaxants?|supplements?|hormones?|estrogen|progesterone|steroids?|insulin|medicines?|medication|drops|cream|ointment)\b|\b(?:medicines?|drugs?|medication) like\b|\b[a-z]+(?:azole|prazole|cillin|mycin|cycline|floxacin|statin|olol|pril|sartan|profen|cetamol|oxetine|tidine|zepam|azepam)\b/i;
const AI_DISCLAIMER = "AI-generated answer from a research model. Do not take it as medical advice or act on it without a doctor: it can be wrong and it has not examined you.";
const MEDICINE_NOTE = "The AI note mentions medicines. Do not give any medicine or dose because of it: a doctor or pharmacist decides.";

function noteCautions(text, level) {
  const out = [];
  if (level === "urgent" && REASSURING.test(text)) out.push("The note sounds reassuring, but the rules found urgent signs. Follow the urgency above.");
  if (MEDICINE.test(text)) out.push(MEDICINE_NOTE);
  return out;
}
let noteAbort = null;
let intakeBusy = false;

async function postJSON(url, body) {
  const res = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || "server said " + res.status);
  return data;
}

function summaryList(a) {
  const dl = el("dl", "kv summary");
  const rows = [
    ["Symptoms", a.symptoms.length ? a.symptoms.join(", ") : "none recognised"],
    ["Duration", a.duration || "not stated"],
    ["Age", a.age || "not stated"],
    ["Sex", a.sex || "not stated"],
    ["Send to", a.department + (a.specialty ? " (then " + a.specialty + ")" : "")],
  ];
  if (a.temperature_c != null) rows.splice(3, 0, ["Temperature", fmt(a.temperature_c, 1) + " °C / " + fmt(a.temperature_c * 9 / 5 + 32, 1) + " °F"]);
  rows.forEach(([k, v]) => dl.append(el("dt", "", k), el("dd", "", v)));
  return dl;
}

function urgencyCard(a) {
  const c = el("div", "card urgency level-" + a.level);
  c.setAttribute("role", a.level === "emergency" ? "alert" : "status");
  c.append(el("div", "urgency-level", LEVEL_NAME[a.level]), el("p", "urgency-advice", a.advice));
  if (a.reasons.length) {
    const ul = el("ul", "reasons");
    a.reasons.forEach((r) => ul.append(el("li", "", r)));
    c.append(el("div", "reasons-head", a.level === "routine" ? "Checked" : "Why"), ul);
  }
  if (a.other_reasons.length) c.append(el("p", "also", "Also noted: " + a.other_reasons.join("; ")));
  return c;
}

const LAST_INTAKE = "hardllm-last-intake";  // the case shown on the Intake tab, so a refresh brings it back

function rememberIntake(id) {
  try {
    if (id == null) localStorage.removeItem(LAST_INTAKE);
    else localStorage.setItem(LAST_INTAKE, String(id));
  } catch (e) { /* storage blocked: the case is still in the queue */ }
}

// every assessment is a case in the hospital queue straight away (the server triages it)
async function assessIntake(text, patient) {
  text = text.trim();
  if (!text || intakeBusy) return;
  intakeBusy = true;  // only while the case is created; a new assessment may start while the AI note is written
  $("#assess").disabled = true;
  let c;
  try {
    c = await postJSON("/api/cases", { text, patient });
  } catch (e) {
    $("#intake-result").replaceChildren(el("div", "card caution", "Could not assess: " + e.message + ". Is server.py still running?"));
    return;
  } finally {
    intakeBusy = false;
    $("#assess").disabled = false;
  }
  rememberIntake(c.id);
  refreshQueue();
  showIntake(c);
}

async function showIntake(c) {
  if (noteAbort) noteAbort.abort();
  noteAbort = null;
  const a = c.assessment, text = c.text;
  const out = $("#intake-result");
  try {
    out.replaceChildren(urgencyCard(a));

    const sum = el("div", "card");
    sum.append(el("h3", "", "Summary for the doctor"), summaryList(a));
    const act = el("div", "form-actions");
    const sent = el("span", "sent", `In the hospital queue as case #${c.id}${c.patient ? " (" + c.patient + ")" : ""}. `);
    const open = el("button", "btn primary", "Open the queue");
    open.type = "button";
    open.addEventListener("click", () => showTab("queue"));
    const undo = el("button", "btn quiet", "Remove from queue");
    undo.type = "button";
    undo.addEventListener("click", async () => {
      undo.disabled = true;
      try {
        await postJSON("/api/cases/" + c.id, { remove: true });
        sent.textContent = `Case #${c.id} removed from the queue.`;
        open.remove();
        undo.remove();
        rememberIntake(null);
        refreshQueue();
      } catch (e) {
        sent.textContent = "Could not remove it: " + e.message;
        undo.disabled = false;
      }
    });
    act.append(open, undo, sent);
    sum.append(act, el("p", "fine", "Rules, not the AI, set the urgency and this summary. A clinician makes the final decision."));
    out.append(sum);

    const note = el("div", "card ai-note");
    const head = el("div", "ai-head");
    head.append(el("h3", "", "AI note"), el("span", "badge accent", "HardLLM model · not a diagnosis"));
    const body = el("div", "ai-body");
    note.append(head, body);
    out.append(note);
    if (a.level === "emergency") {
      body.append(el("p", "muted", "Skipped for emergencies: act on the urgency above first."));
    } else if (a.level === "review") {
      body.append(el("p", "muted", "Skipped: no symptoms were recognised, so a nurse should talk to the patient."));
    } else if (S.mock) {
      body.append(el("p", "muted", "Preview mode: the model is not loaded, so there is no AI note. Start server.py without --mock."));
    } else {
      const txt = el("span", "text");
      const caret = el("span", "caret");
      body.append(txt, caret);
      const ctl = (noteAbort = new AbortController());
      try {
        const done = await ask(text, (p) => (txt.textContent += p), () => {}, ctl.signal);
        txt.textContent = txt.textContent.trim();
        if (done.dropped) note.append(el("p", "fine", done.dropped + (done.dropped === 1 ? " sentence" : " sentences") + " of advice on what to do were hidden. The AI note only explains."));
        body.prepend(...noteCautions(txt.textContent, a.level).map((c) => el("p", "caution", c)));
        note.append(el("p", "fine", "Written by a 3-billion-parameter model trained on doctors' forum answers. It can be wrong, and it has not examined the patient."));
      } catch (e) {
        if (e.name !== "AbortError") txt.textContent = "Could not reach the model (" + e.message + ").";
      } finally {
        caret.remove();
      }
    }
  } catch (e) {
    out.replaceChildren(el("div", "card caution", "Could not show the case: " + e.message));
  }
}

// after a refresh: show the last case again if it is still in the queue
async function restoreIntake() {
  let id = null;
  try { id = Number(localStorage.getItem(LAST_INTAKE)); } catch (e) { return; }
  if (!id) return;
  try {
    const res = await fetch("/api/cases");
    const c = (await res.json()).cases.find((x) => x.id === id);
    if (!c) return rememberIntake(null);
    $("#intake-text").value = c.text;
    $("#patient").value = c.patient || "";
    showIntake(c);
  } catch (e) { /* the form still works */ }
}

function renderIntake() {
  $("#intake-chips").replaceChildren(...INTAKE_EXAMPLES.map((q) => {
    const c = el("button", "chip", q);
    c.type = "button";
    c.addEventListener("click", () => {
      $("#intake-text").value = q;
      assessIntake(q, $("#patient").value.trim());
    });
    return c;
  }));
}
$("#intake-form").addEventListener("submit", (e) => {
  e.preventDefault();
  assessIntake($("#intake-text").value, $("#patient").value.trim());
});
$("#intake-text").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) $("#intake-form").requestSubmit();
});

// ---------- hospital queue tab ----------
let queue = { cases: [], now: 0 };

function waited(c) {
  const min = Math.max(0, Math.round(((c.status === "seen" ? c.seen_at : queue.now) - c.created) / 60));
  const t = min < 60 ? min + " min" : Math.floor(min / 60) + " h " + (min % 60) + " min";
  return c.status === "seen" ? "seen after " + t : "waiting " + t;
}

async function refreshQueue() {
  try {
    const res = await fetch("/api/cases");
    if (!res.ok) throw new Error("server said " + res.status);
    queue = await res.json();
    renderQueue();
  } catch (e) {
    $("#queue-list").replaceChildren(el("div", "card caution", "Could not load the queue: " + e.message));
  }
}

async function updateCase(id, body) {
  try {
    await postJSON("/api/cases/" + id, body);
  } catch (e) {
    alert("Could not update case #" + id + ": " + e.message);
  }
  refreshQueue();
}

function renderQueue() {
  const waiting = queue.cases.filter((c) => c.status === "waiting");
  const counts = $("#queue-counts");
  counts.replaceChildren(...["emergency", "urgent", "review", "routine"].map((l) => {
    const n = waiting.filter((c) => c.assessment.level === l).length;
    const b = el("span", "count level-" + l);
    b.append(el("b", "", String(n)), document.createTextNode(" " + LEVEL_NAME[l].toLowerCase()));
    return b;
  }));
  const tc = $("#queue-count");
  tc.textContent = String(waiting.length);
  tc.hidden = !waiting.length;

  const sel = $("#dept-filter");
  const keep = sel.value;
  const depts = [...new Set(queue.cases.map((c) => c.assessment.department))].sort();
  sel.replaceChildren(el("option", "", "All departments"), ...depts.map((d) => el("option", "", d)));
  sel.options[0].value = "";
  sel.value = depts.includes(keep) ? keep : "";

  const shown = queue.cases.filter((c) => (sel.value === "" || c.assessment.department === sel.value)
    && ($("#show-seen").checked || c.status === "waiting"));
  const list = $("#queue-list");
  if (!shown.length) {
    list.replaceChildren(el("div", "card placeholder", queue.cases.length
      ? "No patients match this view."
      : "No patients yet. Assess one on the Intake tab and send it here."));
    return;
  }
  list.replaceChildren(...shown.map((c) => {
    const a = c.assessment;
    const card = el("article", "case level-" + a.level + (c.status === "seen" ? " seen" : ""));
    const top = el("header", "case-top");
    top.append(el("span", "badge lvl", LEVEL_NAME[a.level]), el("h3", "", c.patient || "Patient #" + c.id),
      el("span", "case-dept", a.department), el("span", "case-wait", waited(c)));
    const sym = el("p", "case-sym", a.symptoms.length ? a.symptoms.join(", ") : "no symptoms recognised");
    const why = a.level !== "routine" && a.reasons.length ? el("p", "case-why", "Why: " + a.reasons.join("; ")) : null;
    const quote = el("blockquote", "case-text", c.text);
    const acts = el("div", "case-acts");
    const toggle = el("button", "btn", c.status === "seen" ? "Back to waiting" : "Mark seen");
    toggle.type = "button";
    toggle.addEventListener("click", () => updateCase(c.id, { status: c.status === "seen" ? "waiting" : "seen" }));
    const rm = el("button", "btn quiet", "Remove");
    rm.type = "button";
    rm.addEventListener("click", () => {
      if (confirm("Remove case #" + c.id + " from the queue?")) updateCase(c.id, { remove: true });
    });
    acts.append(toggle, rm, el("span", "case-id", "#" + c.id + (a.duration ? " · " + a.duration : "") + (a.age ? " · " + a.age : "")));
    card.append(top, sym, ...(why ? [why] : []), quote, acts);
    return card;
  }));
}
$("#dept-filter").addEventListener("change", renderQueue);
$("#show-seen").addEventListener("change", renderQueue);
setInterval(() => {
  if (!document.hidden && !$("#tab-queue").hidden) refreshQueue();
}, 5000);

// ---------- compare tab ----------
const CARDS = [
  ["base", "Base Llama-3.2-3B", "no fine-tuning"],
  ["hardllm", "Trained on Step 3 synthetic data", "the paper's full pipeline"],
  ["selected", "Final: trained on Step 2 selection", "public data chosen privately"],
  ["private", "Trained on the private records", "ceiling, not private"],
];

function renderCompare(i) {
  const m = Object.fromEntries(S.models.map((x) => [x.id, x]));
  document.querySelectorAll("#q-list button").forEach((b, j) => b.setAttribute("aria-current", String(j === i)));
  const d = S.demo[i];
  $("#q-full").textContent = d.question;
  const cards = CARDS.map(([id, title, sub]) => {
    const c = el("article", "answer" + (id === "selected" ? " final" : ""));
    const h = el("header");
    h.append(el("h4", "", title), el("span", "ppl", "perplexity " + fmt(m[id].ppl, 2)));
    c.append(h, el("div", "sub", sub), el("div", "body", d.answers[id]));
    return c;
  });
  const live = el("article", "answer final");
  const h = el("header");
  h.append(el("h4", "", "Final model, live on this laptop"), el("span", "ppl", (VERSION_NAME[S.version] || "") + " · perplexity " + fmt(S.step5.find((x) => x.version === S.version).ppl, 2)));
  const body = el("div", "body");
  const meta = el("div", "meta");
  const btn = el("button", "btn", S.mock ? "Show recorded 4-bit answer" : "Ask the live model");
  btn.type = "button";
  btn.addEventListener("click", async () => {
    if (busy) return;
    busy = true;
    btn.disabled = true;
    body.textContent = "";
    const t0 = performance.now();
    try {
      const ev = await ask(d.question, (p) => (body.textContent += p));
      body.textContent = body.textContent.trim();
      renderMeta(meta, ev, performance.now() - t0);
    } catch (e) {
      body.textContent = "Could not reach the model (" + e.message + ").";
    } finally {
      busy = false;
      btn.disabled = false;
    }
  });
  live.append(h, el("div", "sub", "same question, answered now"), body, meta, btn);
  $("#answers").replaceChildren(...cards, live);
}

function renderCompareList() {
  $("#q-list").replaceChildren(...S.demo.map((d, i) => {
    const li = el("li");
    const b = el("button");
    b.type = "button";
    b.append(el("span", "", short(d.question, 110)));
    b.addEventListener("click", () => renderCompare(i));
    li.append(b);
    return li;
  }));
  renderCompare(0);
}

// ---------- pipeline tab ----------
function stat(value, label) {
  const s = el("div", "stat");
  s.append(el("b", "", value), el("span", "", label));
  return s;
}

function renderPipeline() {
  const { step1: a, step2: b, step3: c } = S;
  const m = Object.fromEntries(S.models.map((x) => [x.id, x]));
  const q4 = S.step5.find((x) => x.version === "q4_k_m");
  const full = S.step5.find((x) => x.version === "merged");
  const gap = (Math.log(m.random.ppl) - Math.log(m.selected.ppl)) / (Math.log(m.random.ppl) - Math.log(m.private.ppl));
  const flow = (items) => {
    const f = el("div", "flow");
    items.forEach((it, i) => {
      if (i) f.append(el("span", "arrow", "→"));
      f.append(el("span", "box " + (it[1] || ""), it[0]));
    });
    return f;
  };
  const steps = [
    {
      title: "Index the public data",
      text: `${int(a.pool_rows)} public texts from 11 datasets (reviews, news, Q&A, medical, legal, finance) are grouped into ${int(a.clusters)} topics. A retrieval model (mT5-base) learns to name the topic of any text.`,
      stats: [[int(a.pool_rows), "public texts"], [int(a.clusters), "topics"], [pct(a.hits1), "topic right first try"], [pct(a.hits10), "right topic in its top 10"]],
    },
    {
      title: "Choose public data with the private records, privately",
      text: `Each of the hospital's ${int(b.records)} private records votes for its ${b.votes_per_record} closest topics. Random noise is added to the vote counts (differential privacy, ε = ${b.eps}, δ = 1/${int(Math.round(1 / b.delta))}) so no single patient can be detected, and topics below ${b.threshold_sigmas}σ are dropped. Only public rows from the winning topics are used. No patient text leaves the hospital.`,
      stats: [[`ε = ${b.eps}`, "privacy budget"], [int(b.clusters_selected), "topics chosen"], [int(b.rows_selected), "public rows selected"], [pct(b.medical_frac_selected), `medical, vs ${pct(b.pool_medical_frac)} of the pool`]],
      flow: [["60,000 private records", "private"], ["noisy topic votes"], [int(b.rows_selected) + " public rows", "public"]],
    },
    {
      title: "Synthetic data",
      rolled: true,
      text: `GPT-2, trained on the selected Q&A, wrote ${int(c.generated)} synthetic Q&A pairs (${int(c.kept)} kept after cleaning). Llama trained on them came out worse than the untouched base model: ${fmt(m.hardllm.ppl)} vs ${fmt(m.base.ppl)}. The paper's sampling (top-k ${c.top_k}, top-p ${c.top_p}) cuts rare words: ${int(c.distinct_words.hardllm)} distinct words per 100k against ${int(c.distinct_words.private)} in real answers. Sampling without the cut restored the vocabulary and improved it to ${fmt(m["hardllm-fullsample"].ppl)}, still worse than base. A 124M-parameter writer cannot teach a 3B model. So the pipeline rolls back to the Step 2 output.`,
      stats: [[int(c.generated), "synthetic pairs written"], [fmt(m.hardllm.ppl), "perplexity, paper's sampling"], [fmt(m["hardllm-fullsample"].ppl), "perplexity, full sampling"], [fmt(m.base.ppl), "base model, for reference"]],
    },
    {
      title: "Fine-tune Llama-3.2-3B",
      text: `LoRA fine-tuning with identical settings for every training set (16,000 Q&A rows, 1,000 steps), so the data is the only difference. Scored on 1,000 real doctor answers none of the models saw. Training on the privately chosen public data closes ${pct(gap, 0)} of the gap between random public data and the private records themselves.`,
      stats: [[fmt(m.selected.ppl), "final model"], [fmt(m.random.ppl), "random public data"], [fmt(m.private.ppl), "private records (ceiling)"], [fmt(m.base.ppl), "base model"]],
    },
    {
      title: "Compress for the device",
      text: `The final model is merged and converted to GGUF, the format phone and laptop apps run, then quantized to 4 bits with llama.cpp. ${fmt(full.file_gb / q4.file_gb, 1)}× smaller for ${pct(q4.ppl / full.ppl - 1)} higher perplexity. This page is running that file.`,
      stats: [[fmt(q4.file_gb) + " GB", `4-bit file (was ${fmt(full.file_gb, 1)} GB)`], [fmt(q4.ppl), "4-bit perplexity"], [pct(q4.weight_change), "weights changed"], ["offline", "no network needed"]],
    },
  ];
  $("#steps").replaceChildren(...steps.map((s, i) => {
    const li = el("li", "step" + (s.rolled ? " rolled" : ""));
    li.append(el("div", "step-n", String(i + 1)));
    const card = el("div", "card");
    const h = el("h3", "", s.title);
    if (s.rolled) {
      const badge = el("span", "badge warn");
      badge.innerHTML = ICON_UNDO;
      badge.append(" Assessed, rolled back");
      h.append(badge);
    }
    const st = el("div", "stats");
    s.stats.forEach(([v, l]) => st.append(stat(v, l)));
    card.append(h, el("p", "", s.text), st);
    if (s.flow) card.append(flow(s.flow));
    li.append(card);
    return li;
  }));
}

// ---------- results tab ----------
const FINAL = new Set(["selected", "q4_k_m"]);
// the Results chart shows only these; the other arms stay in the data for the other tabs
const SHOWN = new Set(["base", "selected", "q4_k_m", "private"]);

function renderChart() {
  const rows = S.models.filter((x) => SHOWN.has(x.id)).sort((x, y) => x.ppl - y.ppl);
  const base = S.models.find((x) => x.id === "base").ppl;
  const xmax = Math.ceil(Math.max(...rows.map((r) => r.ppl)) / 5) * 5;
  const pos = (v) => (100 * v) / xmax + "%";
  const chart = $("#chart");
  chart.replaceChildren();
  const plot = el("div", "c-plot");
  const grid = el("div", "c-grid");
  for (let t = 0; t <= xmax; t += 5) {
    const g = el("i");
    g.style.left = pos(t);
    grid.append(g);
  }
  const bl = el("i", "base");
  bl.style.left = pos(base);
  bl.dataset.label = "base model " + fmt(base);
  grid.append(bl);
  plot.append(grid);
  const tip = $("#tooltip");
  rows.forEach((r) => {
    const row = el("div", "c-row" + (FINAL.has(r.id) ? " hi" : ""));
    row.tabIndex = 0;
    const track = el("div", "c-track");
    const bar = el("div", "c-bar");
    bar.style.width = pos(r.ppl);
    track.append(bar);
    row.append(el("div", "c-label", r.name), track, el("div", "c-value", fmt(r.ppl)));
    const show = () => {
      tip.replaceChildren(el("b", "", r.name), document.createTextNode(r.trained_on), el("br"),
        document.createTextNode(`perplexity ${fmt(r.ppl)} · ${r.ppl < base ? fmt(base - r.ppl) + " better" : fmt(r.ppl - base) + " worse"} than base`));
      tip.hidden = false;
      const cr = $(".chart-card").getBoundingClientRect();
      const br = bar.getBoundingClientRect();
      tip.style.left = Math.min(br.right - cr.left + 10, cr.width - 270) + "px";
      tip.style.top = br.top - cr.top - 8 + "px";
    };
    row.addEventListener("mouseenter", show);
    row.addEventListener("focus", show);
    row.addEventListener("mouseleave", () => (tip.hidden = true));
    row.addEventListener("blur", () => (tip.hidden = true));
    plot.append(row);
  });
  const axis = el("div", "c-axis");
  for (let t = 0; t <= xmax; t += 5) {
    const s = el("span", "", String(t));
    s.style.left = pos(t);
    axis.append(s);
  }
  chart.append(plot, axis, el("p", "chart-note", "Blue: the final model, full size and the 4-bit version this demo runs. Dashed line: base Llama, no fine-tuning."));

  const t = el("table");
  t.innerHTML = '<thead><tr><th>Model</th><th>Trained on</th><th class="n">Answer perplexity</th></tr></thead>';
  const tb = el("tbody");
  rows.forEach((r) => {
    const tr = el("tr", FINAL.has(r.id) ? "hi" : "");
    tr.append(el("td", "", r.name), el("td", "", r.trained_on), el("td", "n", fmt(r.ppl)));
    tb.append(tr);
  });
  t.append(tb);
  $("#chart-table").replaceChildren(wrap(t));
}

document.querySelectorAll(".seg button").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll(".seg button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
  $("#chart").hidden = b.dataset.view !== "chart";
  $("#chart-table").hidden = b.dataset.view !== "table";
}));

function renderCompress() {
  const names = { merged: "Merged, PyTorch (bf16)", f16: "GGUF 16-bit", q8_0: "GGUF 8-bit", q4_k_m: "GGUF 4-bit (this demo)" };
  const t = el("table");
  t.innerHTML = '<thead><tr><th>Version</th><th class="n">File size</th><th class="n">Answer perplexity</th><th class="n">llama.cpp perplexity</th><th class="n">Weights changed</th></tr></thead>';
  const tb = el("tbody");
  S.step5.forEach((r) => {
    const tr = el("tr", r.version === S.version ? "hi" : "");
    tr.append(el("td", "", names[r.version]), el("td", "n", fmt(r.file_gb) + " GB"), el("td", "n", fmt(r.ppl, 3)),
      el("td", "n", r.llamacpp_ppl == null ? "–" : fmt(r.llamacpp_ppl, 2)), el("td", "n", pct(r.weight_change, 2)));
    tb.append(tr);
  });
  t.append(tb);
  const note = el("p", "chart-note", "llama.cpp perplexity scores the whole text, questions included, in the runtime this demo uses: compare its rows with each other, not with answer perplexity. The 16-bit file scoring exactly what the merged model scores shows the measurement is exact.");
  $("#compress").replaceChildren(wrap(t), note);
}

function renderSources() {
  const names = { step1: "Step 1", step2: "Step 2", step3: "Step 3", step4: "Step 4 and 5 models", step5: "Step 5", demo: "Demo questions" };
  const ul = el("ul");
  Object.entries(S.sources).forEach(([k, files]) => {
    const li = el("li");
    li.append(el("span", "", names[k] + ": "));
    files.forEach((f, i) => {
      if (i) li.append(", ");
      li.append(el("code", "", f));
    });
    ul.append(li);
  });
  $("#sources").replaceChildren(ul);
}

// ---------- start ----------
fetch("/api/story").then((r) => r.json()).then((story) => {
  S = story;
  const st = $("#status");
  st.classList.add(S.mock ? "mock" : "live");
  $("#status-text").textContent = S.mock ? "Preview: recorded answers" : `Live · ${VERSION_NAME[S.version] || S.version} · offline`;
  renderAsk();
  renderIntake();
  restoreIntake();
  refreshQueue();
  // Compare and How it works tabs are off for now; their render functions are kept above
  renderChart();
  renderCompress();
  renderSources();
  const tab = location.hash.slice(1);
  showTab(document.getElementById("tab-" + tab) ? tab : "ask");
}).catch((e) => {
  $("#status-text").textContent = "server not reachable";
  console.error(e);
});
