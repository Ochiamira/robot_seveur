// ═══════════════════════════════════════════════════════════════════════
// Config
// ═══════════════════════════════════════════════════════════════════════
const API_URL = location.origin;
const WS_SCHEME = location.protocol === "https:" ? "wss" : "ws";

function loadAccessToken() {
  const hash = new URLSearchParams(location.hash.replace(/^#/, ""));
  const fromUrl = hash.get("token");
  if (fromUrl) {
    sessionStorage.setItem("nexorToken", fromUrl);
    history.replaceState(null, "", location.pathname + location.search);
  }
  return fromUrl || sessionStorage.getItem("nexorToken") || "";
}

const ACCESS_TOKEN = loadAccessToken();
const WS_URL = `${WS_SCHEME}://${location.host}/ws${ACCESS_TOKEN ? `?token=${encodeURIComponent(ACCESS_TOKEN)}` : ""}`;

function authHeaders(extra = {}) {
  return ACCESS_TOKEN
    ? { ...extra, Authorization: `Bearer ${ACCESS_TOKEN}` }
    : extra;
}

async function apiFetch(path, options = {}) {
  const response = await fetch(`${API_URL}${path}`, {
    ...options,
    headers: authHeaders(options.headers || {}),
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    const error = new Error(body.detail || `Erreur HTTP ${response.status}`);
    error.status = response.status;
    throw error;
  }
  return response;
}

function escapeHtml(value) {
  const div = document.createElement("div");
  div.textContent = String(value ?? "");
  return div.innerHTML;
}

const STATUS_LABELS = {
  ok: { label: "Disponible", cls: "ok" },
  en_route: { label: "En service", cls: "ok" },
  attente: { label: "En attente", cls: "warn" },
  bloque: { label: "Bloqué", cls: "alert" },
  aide_demandee: { label: "Aide demandée", cls: "alert" },
  hors_ligne: { label: "Hors ligne", cls: "off" },
};

// Positions fixes des tables sur le plan (normalisées 0–1).
// À adapter aux coordonnées réelles de la salle de NEXOR.
const TABLES = [
  { name: "T1", x: 0.15, y: 0.20 }, { name: "T2", x: 0.50, y: 0.15 }, { name: "T3", x: 0.85, y: 0.20 },
  { name: "T4", x: 0.15, y: 0.55 }, { name: "T5", x: 0.50, y: 0.55 }, { name: "T6", x: 0.85, y: 0.55 },
  { name: "T7", x: 0.15, y: 0.88 }, { name: "T8", x: 0.50, y: 0.88 }, { name: "T9", x: 0.85, y: 0.88 },
];

// Workflow des commandes — doit rester cohérent avec db.STATUS_FLOW côté serveur
const STATUS_FLOW = {
  confirmee: ["en_preparation", "annulee"],
  en_preparation: ["prete", "annulee"],
  prete: ["servie"],
  servie: [],   // -> payee uniquement via la modale d'encaissement
  payee: [],
  annulee: [],
};

const STATUS_LABELS_ORDER = {
  confirmee: "Confirmée",
  en_preparation: "En préparation",
  prete: "Prête",
  servie: "Servie",
  payee: "Payée",
  annulee: "Annulée",
};

const NEXT_ACTION_LABEL = {
  en_preparation: "Lancer préparation",
  prete: "Marquer prête",
  servie: "Marquer servie",
};

const KANBAN_COLUMNS = [
  { status: "confirmee", label: "À préparer", hint: "Nouvelles commandes" },
  { status: "en_preparation", label: "En préparation", hint: "En cuisine" },
  { status: "prete", label: "Prêtes", hint: "À récupérer" },
  { status: "servie", label: "Servies", hint: "En attente de paiement" },
];

// ═══════════════════════════════════════════════════════════════════════
// État local
// ═══════════════════════════════════════════════════════════════════════
let orders = {};       // order_id -> order (vue "en direct")
let historyOrders = [];
let currentTab = "live";
let alerts = [];
let robotPos = { x: 0.5, y: 0.5, theta: 0 };
let robotTrail = [];
let tables = {};       // table_id ("T3") -> {status, message, waiting_s, ts} — vision_bridge.py

// ═══════════════════════════════════════════════════════════════════════
// WebSocket
// ═══════════════════════════════════════════════════════════════════════
function connect() {
  const ws = new WebSocket(WS_URL);

  ws.onopen = () => setConnState(true);
  ws.onclose = (event) => {
    setConnState(false, event.code === 4401 ? "Accès refusé — ajoutez #token=… à l’URL" : null);
    setTimeout(connect, event.code === 4401 ? 10000 : 2000);
  };
  ws.onerror = () => ws.close();

  ws.onmessage = (evt) => {
    try {
      handleMessage(JSON.parse(evt.data));
    } catch (error) {
      console.warn("Message WebSocket invalide", error);
    }
  };
}

function setConnState(live, detail = null) {
  const dot = document.getElementById("connDot");
  const label = document.getElementById("connLabel");
  dot.className = "conn-dot " + (live ? "live" : "down");
  label.textContent = detail || (live ? "Connecté" : "Reconnexion…");
}

function handleMessage(msg) {
  switch (msg.type) {
    case "snapshot":
      orders = {}; msg.orders.forEach(o => orders[o.order_id] = o);
      alerts = msg.alerts || [];
      robotPos = msg.robot_position || robotPos;
      tables = msg.tables || {};
      updateRobotStatus(msg.robot_status);
      renderOrders(); renderAlerts();
      break;

    case "order":
      if (["payee", "annulee"].includes(msg.order.status)) {
        delete orders[msg.order.order_id];   // sort de la vue "en direct", reste dans l'historique
      } else {
        orders[msg.order.order_id] = msg.order;
      }
      if (currentTab === "live") renderOrders();
      break;

    case "robot_status":
      updateRobotStatus(msg.robot_status);
      break;

    case "alert":
      alerts.push(msg.alert);
      renderAlerts();
      break;

    case "alert_cleared":
      alerts = alerts.filter(a => a.id !== msg.alert_id);
      renderAlerts();
      break;

    case "robot_position":
      robotPos = msg.robot_position;
      break;

    case "table_status":
      tables[msg.table] = msg.info;
      break;
  }
}

// ═══════════════════════════════════════════════════════════════════════
// Rendu — Commandes
// ═══════════════════════════════════════════════════════════════════════
function switchTab(tab) {
  currentTab = tab;
  document.getElementById("tabLive").classList.toggle("active", tab === "live");
  document.getElementById("tabHistory").classList.toggle("active", tab === "history");
  if (tab === "history") loadHistory();
  else renderOrders();
}

async function loadHistory() {
  try {
    const res = await apiFetch("/api/orders/history?limit=50");
    historyOrders = await res.json();
    renderOrders();
  } catch (error) {
    setConnState(false, error.message);
  }
}

function renderOrders() {
  const list = document.getElementById("ordersList");
  const count = document.getElementById("orderCount");

  const items = currentTab === "live"
    ? Object.values(orders).sort((a, b) => b.created_ts - a.created_ts)
    : historyOrders;

  count.textContent = currentTab === "live"
    ? items.length
    : items.length;

  if (!items.length) {
    list.innerHTML = `<p class="empty-state">${currentTab === "live"
      ? "Aucune commande active. Les commandes confirmées par NEXOR apparaîtront ici."
      : "Aucun historique pour l'instant."
      }</p>`;
    return;
  }

  if (currentTab === "live") {
    list.innerHTML = `
      <div class="kanban-board">
        ${KANBAN_COLUMNS.map(column => {
          const columnOrders = items.filter(order => order.status === column.status);
          return `
            <section class="kanban-column" data-column-status="${column.status}">
              <header class="kanban-head">
                <div>
                  <h3>${escapeHtml(column.label)}</h3>
                  <span>${escapeHtml(column.hint)}</span>
                </div>
                <strong>${columnOrders.length}</strong>
              </header>
              <div class="kanban-cards">
                ${columnOrders.length
                  ? columnOrders.map(renderOrderCard).join("")
                  : `<div class="kanban-empty"><span>✓</span>Aucune commande</div>`}
              </div>
            </section>`;
        }).join("")}
      </div>`;
    return;
  }

  list.innerHTML = `<div class="history-list">${items.map(renderOrderCard).join("")}</div>`;
}

function renderOrderCard(o) {
    const nextStatuses = STATUS_FLOW[o.status] || [];
    const nextAdvance = nextStatuses.find(s => s !== "annulee");
    const canCancel = nextStatuses.includes("annulee");
    const canPay = o.status === "servie";
    const orderId = escapeHtml(o.order_id);
    const safeStatus = escapeHtml(o.status);

    return `
    <div class="order-card" data-status="${safeStatus}" data-order-id="${orderId}">
      <div class="order-top">
        <div class="order-table-wrap">
          <span class="order-table-label">Table</span>
          <span class="order-table">${escapeHtml(o.table_name || '—')}</span>
        </div>
        <span class="order-time"><i></i>${timeAgo(o.created_ts)}</span>
      </div>
      <ul class="order-items">${(o.items || []).map(item => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
      <div class="order-bottom">
        <div class="order-summary">
          <span class="order-total">${o.total != null ? escapeHtml(o.total.toFixed(3) + " " + o.devise) : ""}</span>
          <span class="status-pill" data-status="${safeStatus}">${escapeHtml(STATUS_LABELS_ORDER[o.status] || o.status)}</span>
        </div>
        ${currentTab === "live" ? `
        <div class="order-actions">
          ${canCancel ? `<button class="btn-cancel" data-action="advance" data-status-target="annulee">Annuler</button>` : ""}
          ${canPay ? `<button class="btn-advance" data-action="pay" data-total="${Number(o.total)}">Encaisser</button>` : ""}
          ${nextAdvance ? `<button class="btn-advance" data-action="advance" data-status-target="${nextAdvance}">${escapeHtml(NEXT_ACTION_LABEL[nextAdvance])}</button>` : ""}
        </div>` : `
        ${o.status === 'payee' ? `<span class="order-total" style="text-align:right">${o.payment_method === 'especes' ? 'Espèces' : 'Carte'}</span>` : ""}
        `}
      </div>
    </div>
  `;
}

document.getElementById("ordersList").addEventListener("click", (event) => {
  const card = event.target.closest(".order-card");
  if (!card) return;
  const button = event.target.closest("button[data-action]");
  if (!button) {
    openTimeline(card.dataset.orderId);
    return;
  }
  event.stopPropagation();
  if (button.dataset.action === "pay") {
    openPayModal(card.dataset.orderId, Number(button.dataset.total));
  } else if (button.dataset.action === "advance") {
    advanceOrder(card.dataset.orderId, button.dataset.statusTarget, button);
  }
});

async function advanceOrder(orderId, toStatus, button = null) {
  if (button) button.disabled = true;
  try {
    await apiFetch(`/api/orders/${encodeURIComponent(orderId)}/advance`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ to_status: toStatus }),
    });
    return true;
  } catch (error) {
    setConnState(false, error.message);
    if (button) button.disabled = false;
    return false;
  }
}

// ── Timeline d'une commande ───────────────────────────────────────────
async function openTimeline(orderId) {
  let payload;
  try {
    const res = await apiFetch(`/api/orders/${encodeURIComponent(orderId)}/timeline`);
    payload = await res.json();
  } catch (error) {
    setConnState(false, error.message);
    return;
  }
  const { order, timeline } = payload;

  document.getElementById("timelineTitle").textContent =
    order.table_name ? `Table ${order.table_name}` : "Commande";

  document.getElementById("timelineOrderSummary").innerHTML = `
    <span class="timeline-order-id">Commande #${escapeHtml(order.order_id)}</span>
    <ul>${(order.items || []).map(item => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
    <strong>${order.total != null ? `${Number(order.total).toFixed(3)} ${escapeHtml(order.devise || "TND")}` : ""}</strong>`;

  document.getElementById("timelineSteps").innerHTML = timeline.map(t => `
    <div class="timeline-step">
      <span class="timeline-dot"></span>
      <div>
        <div class="timeline-label">${escapeHtml(STATUS_LABELS_ORDER[t.status] || t.status)}</div>
        <div class="timeline-time">${new Date(t.ts * 1000).toLocaleTimeString('fr-FR')}</div>
      </div>
    </div>
  `).join("");

  const nextStatuses = STATUS_FLOW[order.status] || [];
  const nextAdvance = nextStatuses.find(status => status !== "annulee");
  const canCancel = nextStatuses.includes("annulee");
  const canPay = order.status === "servie";
  document.getElementById("timelineActions").innerHTML = `
    ${canCancel ? `<button class="btn-cancel" data-timeline-action="advance" data-order-id="${escapeHtml(order.order_id)}" data-status-target="annulee">Annuler</button>` : ""}
    ${canPay ? `<button class="btn-advance" data-timeline-action="pay" data-order-id="${escapeHtml(order.order_id)}" data-total="${Number(order.total)}">Encaisser</button>` : ""}
    ${nextAdvance ? `<button class="btn-advance" data-timeline-action="advance" data-order-id="${escapeHtml(order.order_id)}" data-status-target="${nextAdvance}">${escapeHtml(NEXT_ACTION_LABEL[nextAdvance])}</button>` : ""}`;

  document.getElementById("timelineOverlay").classList.add("open");
}

function closeTimeline(evt) {
  if (evt && evt.target.id !== "timelineOverlay") return;
  document.getElementById("timelineOverlay").classList.remove("open");
}

// ── Modale de paiement ─────────────────────────────────────────────────
let payState = { orderId: null, total: 0, method: "especes" };

function openPayModal(orderId, total) {
  payState = { orderId, total, method: "especes" };

  document.getElementById("payTitle").textContent = "Encaisser";
  document.getElementById("payTotal").textContent = `${total.toFixed(3)} TND`;
  document.getElementById("payAmount").value = "";
  document.getElementById("payChange").textContent = "";
  document.getElementById("payError").textContent = "";
  selectPayMethod("especes");

  document.getElementById("payOverlay").classList.add("open");
}

function closePayModal(evt) {
  if (evt && evt.target.id !== "payOverlay") return;
  document.getElementById("payOverlay").classList.remove("open");
}

function selectPayMethod(method) {
  payState.method = method;
  document.getElementById("payMethodEspeces").classList.toggle("active", method === "especes");
  document.getElementById("payMethodCarte").classList.toggle("active", method === "carte");
  document.getElementById("payCashFields").style.display = method === "especes" ? "block" : "none";
  document.getElementById("payError").textContent = "";
}

document.addEventListener("input", (e) => {
  if (e.target.id !== "payAmount") return;
  const amount = parseFloat(e.target.value);
  const changeEl = document.getElementById("payChange");
  if (isNaN(amount)) { changeEl.textContent = ""; return; }
  const diff = amount - payState.total;
  if (diff < 0) {
    changeEl.textContent = `Manque ${Math.abs(diff).toFixed(3)} TND`;
    changeEl.classList.add("insufficient");
  } else {
    changeEl.textContent = `Monnaie à rendre : ${diff.toFixed(3)} TND`;
    changeEl.classList.remove("insufficient");
  }
});

async function confirmPayment() {
  const errorEl = document.getElementById("payError");
  const confirmButton = document.getElementById("payConfirmBtn");
  const body = { method: payState.method };

  if (payState.method === "especes") {
    const amount = parseFloat(document.getElementById("payAmount").value);
    if (isNaN(amount)) { errorEl.textContent = "Saisis le montant reçu."; return; }
    body.amount_paid = amount;
  }

  confirmButton.disabled = true;
  try {
    await apiFetch(`/api/orders/${encodeURIComponent(payState.orderId)}/pay`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    closePayModal();
  } catch (error) {
    errorEl.textContent = error.message || "Erreur lors de l'encaissement.";
  } finally {
    confirmButton.disabled = false;
  }
}

// ═══════════════════════════════════════════════════════════════════════
// Rendu — Statut robot + alertes
// ═══════════════════════════════════════════════════════════════════════
function updateRobotStatus(status) {
  if (!status) return;
  const info = STATUS_LABELS[status.status] || STATUS_LABELS.hors_ligne;
  document.getElementById("statusDot").className = "status-dot " + info.cls;
  document.getElementById("statusLabel").textContent = info.label;
  document.getElementById("statusDetail").textContent =
    status.message || (status.table ? `Table ${status.table}` : "—");
}

function renderAlerts() {
  const list = document.getElementById("alertsList");
  const count = document.getElementById("alertCount");
  count.textContent = alerts.length;

  if (!alerts.length) {
    list.innerHTML = `<p class="empty-state">Aucune alerte active.</p>`;
    return;
  }

  list.innerHTML = alerts.slice().reverse().map(a => {
    const isPayment = a.type === "payment_intent";
    const label = isPayment
      ? `💳 Le client veut payer ${a.payment_method === "especes" ? "en espèces" : "par carte"} — ${(a.total ?? 0).toFixed(3)} ${a.devise || ""}`
      : (a.message || (a.status === "bloque" ? "Robot bloqué" : "Aide demandée"));
    const btnLabel = isPayment ? "Vu, j'y vais" : "Acquitter";
    return `
    <div class="alert-card ${isPayment ? "alert-payment" : ""}">
      <div>
        <div class="alert-msg">${escapeHtml(label)}</div>
        <div class="alert-meta">${escapeHtml(a.table ? "Table " + a.table + " · " : "")}${timeAgo(a.ts)}${a.order_id ? ` · ${escapeHtml(a.order_id)}` : ""}</div>
      </div>
      <button class="btn-ack" data-alert-id="${escapeHtml(a.id)}">${btnLabel}</button>
    </div>
  `;
  }).join("");
}

async function ackAlert(alertId) {
  try {
    await apiFetch("/api/staff_action", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action: "acquitter_alerte", order_id: alertId }),
    });
  } catch (error) {
    setConnState(false, error.message);
  }
}

document.getElementById("alertsList").addEventListener("click", (event) => {
  const button = event.target.closest("button[data-alert-id]");
  if (!button || button.disabled) return;
  button.disabled = true;
  ackAlert(button.dataset.alertId).finally(() => { button.disabled = false; });
});

// ═══════════════════════════════════════════════════════════════════════
// Rendu — Plan de salle + position robot
// ═══════════════════════════════════════════════════════════════════════
const canvas = document.getElementById("floorCanvas");
const ctx = canvas.getContext("2d");

function drawFloor() {
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  // Tables — couleur selon le statut poussé par vision_bridge.py
  const TABLE_COLORS = {
    libre: "#3B3830",  // neutre, comme avant si aucune info reçue
    occupee: "#3A5F8A",  // bleu — client assis, pas encore de robot
    attente_robot: "#D9A441",  // ambre (--warn) — robot en route
    attente_longue: "#C1502E",  // rouge (--alert) — ça traîne, alerte déjà levée
    en_service: "#4C9A6A",  // vert (--ok) — robot a pris en charge
  };
  TABLES.forEach(t => {
    const px = t.x * W, py = t.y * H;
    const info = tables[t.name];
    ctx.beginPath();
    ctx.arc(px, py, 20, 0, Math.PI * 2);
    ctx.fillStyle = TABLE_COLORS[info?.status] || TABLE_COLORS.libre;
    ctx.fill();
    ctx.fillStyle = "#EDEAE1";
    ctx.font = "11px ui-monospace, monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(t.name, px, py);
  });

  // Traînée du robot
  robotTrail.push({ x: robotPos.x, y: robotPos.y });
  if (robotTrail.length > 25) robotTrail.shift();
  robotTrail.forEach((p, i) => {
    const alpha = i / robotTrail.length * 0.35;
    ctx.beginPath();
    ctx.arc(p.x * W, p.y * H, 4, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(217, 164, 65, ${alpha})`;
    ctx.fill();
  });

  // Robot (point pulsant + direction)
  const rx = robotPos.x * W, ry = robotPos.y * H;
  const pulse = 8 + Math.sin(Date.now() / 250) * 2;

  ctx.beginPath();
  ctx.arc(rx, ry, pulse + 6, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(217, 164, 65, 0.18)";
  ctx.fill();

  ctx.beginPath();
  ctx.arc(rx, ry, 8, 0, Math.PI * 2);
  ctx.fillStyle = "#D9A441";
  ctx.fill();

  const dirX = rx + Math.cos(robotPos.theta || 0) * 16;
  const dirY = ry + Math.sin(robotPos.theta || 0) * 16;
  ctx.beginPath();
  ctx.moveTo(rx, ry);
  ctx.lineTo(dirX, dirY);
  ctx.strokeStyle = "#D9A441";
  ctx.lineWidth = 2;
  ctx.stroke();

  requestAnimationFrame(drawFloor);
}

// ═══════════════════════════════════════════════════════════════════════
// Utilitaires
// ═══════════════════════════════════════════════════════════════════════
function timeAgo(ts) {
  if (!ts) return "";
  const seconds = Math.max(0, Math.floor(Date.now() / 1000 - ts));
  if (seconds < 60) return `il y a ${seconds}s`;
  if (seconds < 3600) return `il y a ${Math.floor(seconds / 60)}min`;
  if (seconds < 86400) return `il y a ${Math.floor(seconds / 3600)}h`;
  return `il y a ${Math.floor(seconds / 86400)}j`;
}

function updateServiceClock() {
  const clock = document.getElementById("serviceClock");
  if (clock) {
    clock.textContent = new Date().toLocaleTimeString("fr-FR", {
      hour: "2-digit",
      minute: "2-digit",
    });
  }
}

// ═══════════════════════════════════════════════════════════════════════
// Init
// ═══════════════════════════════════════════════════════════════════════
document.getElementById("tabLive").addEventListener("click", () => switchTab("live"));
document.getElementById("tabHistory").addEventListener("click", () => switchTab("history"));
document.getElementById("timelineCloseBtn").addEventListener("click", () => closeTimeline());
document.getElementById("timelineOverlay").addEventListener("click", closeTimeline);
document.getElementById("timelineActions").addEventListener("click", async (event) => {
  const button = event.target.closest("button[data-timeline-action]");
  if (!button || button.disabled) return;
  if (button.dataset.timelineAction === "pay") {
    closeTimeline();
    openPayModal(button.dataset.orderId, Number(button.dataset.total));
    return;
  }
  const updated = await advanceOrder(
    button.dataset.orderId,
    button.dataset.statusTarget,
    button,
  );
  if (updated) closeTimeline();
});
document.getElementById("payCloseBtn").addEventListener("click", () => closePayModal());
document.getElementById("payOverlay").addEventListener("click", closePayModal);
document.getElementById("payMethodEspeces").addEventListener("click", () => selectPayMethod("especes"));
document.getElementById("payMethodCarte").addEventListener("click", () => selectPayMethod("carte"));
document.getElementById("payConfirmBtn").addEventListener("click", confirmPayment);

async function refreshState() {
  try {
    const response = await apiFetch("/api/state");
    handleMessage(await response.json());
  } catch (error) {
    setConnState(false, error.message);
  }
}

connect();
drawFloor();
updateServiceClock();
setInterval(updateServiceClock, 30000);
setInterval(renderOrders, 15000); // rafraîchit les "il y a Xmin" même sans nouvel événement
setInterval(refreshState, 5000);   // cohérence même si plusieurs workers/processus publient
