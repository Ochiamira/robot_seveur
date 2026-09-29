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
const WS_URL = `${WS_SCHEME}://${location.host}/ws/client${ACCESS_TOKEN ? `?token=${encodeURIComponent(ACCESS_TOKEN)}` : ""}`;

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
    throw new Error(body.detail || `Erreur HTTP ${response.status}`);
  }
  return response;
}

const STATE_LABELS = {
  fr: {
    idle: "Bonjour ! Dites-moi ce que vous souhaitez commander.",
    listening: "Je vous écoute…",
    processing: "Je réfléchis…",
    speaking: "…",
  },
  en: {
    idle: "Hello! Tell me what you'd like to order.",
    listening: "I'm listening…",
    processing: "Thinking…",
    speaking: "…",
  },
  ar: {
    idle: "مرحبًا! أخبرني بما ترغب في طلبه.",
    listening: "أنا أستمع…",
    processing: "أفكر…",
    speaking: "…",
  },
};

const UI_TEXT = {
  fr: {
    assistantSubtitle: "Votre assistant à table",
    stageKicker: "Je suis là pour vous aider",
    cartTitle: "Votre commande",
    cartEmpty: "Aucun article pour l'instant",
    menuOpen: "Voir le menu",
    qrLabel: "Scanner pour le menu sur votre téléphone",
    callStaff: "Appeler le staff",
    menuKicker: "Fait avec soin",
    menuTitle: "Notre menu",
    sendOrder: "Envoyer la commande",
    item: "article",
    items: "articles",
    allCategories: "Tout",
    confirmTitle: "Commande envoyée !",
    preparation: "Le staff commence sa préparation.",
    tableLabel: "Table",
    orderLabel: "Commande",
    close: "Fermer",
    accessDenied: "Accès refusé — ajoutez #token=… à l’URL",
    menuUnavailable: "Menu indisponible.",
    tableMissing: "Table non configurée : utilisez /client/?table=T4",
    orderSendError: "Commande non envoyée.",
    staffOnWay: "Un membre du staff arrive à votre table 🙌",
    staffCallError: "Impossible de contacter le staff.",
  },
  en: {
    assistantSubtitle: "Your table assistant",
    stageKicker: "I'm here to help",
    cartTitle: "Your order",
    cartEmpty: "No items yet",
    menuOpen: "View menu",
    qrLabel: "Scan to view the menu on your phone",
    callStaff: "Call staff",
    menuKicker: "Made with care",
    menuTitle: "Our menu",
    sendOrder: "Send order",
    item: "item",
    items: "items",
    allCategories: "All",
    confirmTitle: "Order sent!",
    preparation: "The staff is starting its preparation.",
    tableLabel: "Table",
    orderLabel: "Order",
    close: "Close",
    accessDenied: "Access denied — add #token=… to the URL",
    menuUnavailable: "The menu is unavailable.",
    tableMissing: "No table configured: use /client/?table=T4",
    orderSendError: "The order could not be sent.",
    staffOnWay: "A staff member is on the way to your table 🙌",
    staffCallError: "Unable to contact the staff.",
  },
  ar: {
    assistantSubtitle: "مساعدك على الطاولة",
    stageKicker: "أنا هنا لمساعدتك",
    cartTitle: "طلبك",
    cartEmpty: "لا توجد عناصر بعد",
    menuOpen: "عرض القائمة",
    qrLabel: "امسح الرمز لعرض القائمة على هاتفك",
    callStaff: "طلب المساعدة",
    menuKicker: "أُعدّ بعناية",
    menuTitle: "قائمتنا",
    sendOrder: "إرسال الطلب",
    item: "عنصر",
    items: "عناصر",
    allCategories: "الكل",
    confirmTitle: "تم إرسال الطلب!",
    preparation: "بدأ فريق العمل في تحضير طلبك.",
    tableLabel: "الطاولة",
    orderLabel: "الطلب",
    close: "إغلاق",
    accessDenied: "تم رفض الوصول — أضف #token=… إلى الرابط",
    menuUnavailable: "القائمة غير متاحة.",
    tableMissing: "لم يتم تحديد الطاولة: استخدم /client/?table=T4",
    orderSendError: "تعذر إرسال الطلب.",
    staffOnWay: "أحد أفراد الطاقم في طريقه إلى طاولتك 🙌",
    staffCallError: "تعذر التواصل مع فريق العمل.",
  },
};

const STATE_ICON = {
  idle: "😊", listening: "🎙️", processing: "⚙️", speaking: "🔊",
};

// Langue choisie : ?lang=... est prioritaire, puis la préférence de la session.
const SUPPORTED_LANGUAGES = new Set(["fr", "en", "ar"]);
const requestedLang = new URLSearchParams(location.search).get("lang");
const storedLang = sessionStorage.getItem("nexorUiLang");
let uiLang = SUPPORTED_LANGUAGES.has(requestedLang)
  ? requestedLang
  : (SUPPORTED_LANGUAGES.has(storedLang) ? storedLang : "fr");
let lastDialogState = { state: "idle", text: "", lang: "fr" };
let lastDraftOrder = { items: [], total: null };
let lastConfirmedOrder = null;

function tr(key) {
  return (UI_TEXT[uiLang] && UI_TEXT[uiLang][key]) || UI_TEXT.fr[key] || key;
}

function applyStaticTranslations() {
  const textById = {
    brandSub: "assistantSubtitle",
    stageKicker: "stageKicker",
    cartTitle: "cartTitle",
    menuOpenLabel: "menuOpen",
    qrLabel: "qrLabel",
    callStaffLabel: "callStaff",
    menuKicker: "menuKicker",
    menuTitle: "menuTitle",
    confirmTitle: "confirmTitle",
  };
  Object.entries(textById).forEach(([id, key]) => {
    const element = document.getElementById(id);
    if (element) element.textContent = tr(key);
  });

  document.getElementById("menuSendBtn").textContent = tr("sendOrder");
  document.getElementById("menuCloseBtn").setAttribute("aria-label", tr("close"));
  document.documentElement.lang = uiLang;
  document.documentElement.dir = uiLang === "ar" ? "rtl" : "ltr";
}

// ═══════════════════════════════════════════════════════════════════════
// WebSocket
// ═══════════════════════════════════════════════════════════════════════
function connect() {
  const ws = new WebSocket(WS_URL);
  ws.onclose = (event) => {
    if (event.code === 4401) {
      showToast(tr("accessDenied"), false);
    }
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

function handleMessage(msg) {
  switch (msg.type) {
    case "snapshot":
      applyDialogState(msg.dialog_state);
      applyDraftOrder(msg.draft_order);
      break;
    case "dialog_state":
      applyDialogState(msg.dialog_state);
      break;
    case "draft_order":
      applyDraftOrder(msg.draft_order);
      break;
    case "order_confirmed":
      applyDraftOrder({ items: [], total: null });
      flashConfirmation(msg.order);
      break;
  }
}

// ═══════════════════════════════════════════════════════════════════════
// Orbe vocal + transcription
// ═══════════════════════════════════════════════════════════════════════
const orb = document.getElementById("orb");
const orbIcon = document.getElementById("orbIcon");
const stateLabel = document.getElementById("stateLabel");
const transcriptBox = document.getElementById("transcriptBox");

function applyDialogState(ds) {
  if (!ds) return;
  lastDialogState = ds;
  const state = ds.state || "idle";

  orb.className = "orb " + state;

  const labels = STATE_LABELS[uiLang] || STATE_LABELS.fr;
  stateLabel.textContent = state === "speaking" && ds.text
    ? ds.text
    : (labels[state] || "");

  // Transcription : on n'affiche le texte reconnu que pendant listening/processing
  if ((state === "listening" || state === "processing") && ds.text) {
    transcriptBox.innerHTML = `<p class="transcript-text">« ${escapeHtml(ds.text)} »</p>`;
  } else if (state === "idle") {
    transcriptBox.innerHTML = `<p class="transcript-placeholder">…</p>`;
  }

  renderEqBars(state === "speaking");
}

function selectLang(lang) {
  if (!SUPPORTED_LANGUAGES.has(lang)) return;
  uiLang = lang;
  sessionStorage.setItem("nexorUiLang", lang);
  document.querySelectorAll(".lang-pill").forEach(el => {
    el.classList.toggle("active", el.dataset.lang === lang);
  });

  applyStaticTranslations();
  applyDialogState(lastDialogState);   // recalcule la phrase d'accueil dans la nouvelle langue
  applyDraftOrder(lastDraftOrder);
  renderConfirmationSummary(lastConfirmedOrder);
  if (menuLoaded) renderMenu();        // si le menu est déjà chargé, on le retraduit sans re-fetch
}

function renderEqBars(active) {
  const g = document.getElementById("orbWave");
  if (!active) { g.innerHTML = ""; delete g.dataset.built; return; }
  if (g.dataset.built === "1") return;   // déjà construites, l'animation CSS tourne seule
  g.dataset.built = "1";

  const bars = 5;
  const spacing = 14;
  const startX = 100 - ((bars - 1) * spacing) / 2;
  let html = "";
  for (let i = 0; i < bars; i++) {
    const x = startX + i * spacing;
    html += `<rect class="eq-bar" x="${x - 3}" y="88" width="6" height="24" rx="3"
              style="animation-delay:${i * 0.09}s"></rect>`;
  }
  g.innerHTML = html;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

// ═══════════════════════════════════════════════════════════════════════
// Panier live
// ═══════════════════════════════════════════════════════════════════════
function applyDraftOrder(draft) {
  lastDraftOrder = draft || { items: [], total: null };
  const list = document.getElementById("cartList");
  const totalEl = document.getElementById("cartTotal");
  const items = (draft && draft.items) || [];
  const countEl = document.getElementById("cartItemCount");
  const panel = document.getElementById("cartPanel");
  const itemCount = items.reduce((sum, item) => {
    const match = String(item).match(/^\s*(\d+)\s*[xX×]/);
    return sum + (match ? Number(match[1]) : 1);
  }, 0);
  countEl.textContent = itemCount;
  panel.classList.toggle("has-items", items.length > 0);

  if (!items.length) {
    list.innerHTML = `<li class="cart-empty" id="cartEmpty">${escapeHtml(tr("cartEmpty"))}</li>`;
    totalEl.textContent = "";
    return;
  }

  list.innerHTML = items.map(it => `<li class="cart-item">${escapeHtml(it)}</li>`).join("");
  totalEl.textContent = draft.total != null ? `${draft.total.toFixed(3)} TND` : "";
}

function flashConfirmation(order = null) {
  const flash = document.getElementById("confirmFlash");
  lastConfirmedOrder = order;
  renderConfirmationSummary(order);
  flash.classList.remove("show");
  void flash.offsetWidth;   // relance l'animation CSS
  flash.classList.add("show");
}

function renderConfirmationSummary(order = null) {
  const summary = document.getElementById("confirmSummary");
  if (order) {
    const table = order.table_name ? `${tr("tableLabel")} ${order.table_name}` : tr("orderLabel");
    const total = order.total != null ? `${Number(order.total).toFixed(3)} ${order.devise || "TND"}` : "";
    summary.textContent = `${table}${total ? ` · ${total}` : ""} · ${tr("preparation")}`;
  } else {
    summary.textContent = tr("preparation");
  }
}

// ═══════════════════════════════════════════════════════════════════════
// Menu + commande tactile (fallback pour un client qui ne peut/veut pas parler)
// ═══════════════════════════════════════════════════════════════════════
let menuLoaded = false;
let menuData = null;
let tactileCart = {};   // itemKey -> { qty, nom, prix }
let menuItemsByKey = {};
let activeMenuCategory = "all";

// Table servie par cet écran — passe-la dans l'URL, ex: /client/?table=T4
const TABLE_ID = (new URLSearchParams(location.search).get("table") || "").trim();

async function toggleMenu() {
  const overlay = document.getElementById("menuOverlay");
  if (overlay.classList.contains("open")) {
    overlay.classList.remove("open");
    return;
  }
  try {
    if (!menuLoaded) await loadMenu();
    renderMenu();
    overlay.classList.add("open");
  } catch (error) {
    showToast(uiLang === "fr" && error.message ? error.message : tr("menuUnavailable"), false);
  }
}

async function loadMenu() {
  const res = await apiFetch("/api/menu");
  menuData = await res.json();
  menuLoaded = true;
}

// Gère les deux formats : nouveau ({fr,en,ar}) et ancien (texte simple),
// pour ne jamais afficher un menu vide si menu.json n'a pas encore été mis à jour.
function t(field) {
  if (!field) return "";
  if (typeof field === "string") return field;
  return field[uiLang] || field.fr || "";
}

function renderMenu() {
  if (!menuData) return;
  menuItemsByKey = {};

  document.getElementById("menuTitle").textContent = tr("menuTitle");
  document.getElementById("menuSendBtn").textContent = tr("sendOrder");

  const categories = menuData.categories.map((cat, ci) => ({ cat, ci }));
  document.getElementById("menuCategories").innerHTML = `
    <button class="menu-category-pill ${activeMenuCategory === "all" ? "active" : ""}" data-category-index="all">${escapeHtml(tr("allCategories"))}</button>
    ${categories.map(({ cat, ci }) => `
      <button class="menu-category-pill ${activeMenuCategory === String(ci) ? "active" : ""}" data-category-index="${ci}">
        <span>${escapeHtml(cat.icon || "🍽️")}</span>${escapeHtml(t(cat.name))}
      </button>`).join("")}`;

  const visibleCategories = activeMenuCategory === "all"
    ? categories
    : categories.filter(({ ci }) => String(ci) === activeMenuCategory);

  const body = document.getElementById("menuBody");
  body.innerHTML = visibleCategories.map(({ cat, ci }) => `
    <div class="menu-cat">
      <h3><span>${escapeHtml(cat.icon || "🍽️")}</span>${escapeHtml(t(cat.name))}</h3>
      ${cat.items.map((it, ii) => {
    const key = `${ci}_${ii}`;
    const qty = (tactileCart[key] && tactileCart[key].qty) || 0;
    const name = t(it.nom);
    const price = Number(it.prix);
    menuItemsByKey[key] = { nom: name, prix: price };
    if (tactileCart[key]) {
      tactileCart[key].nom = name;
      tactileCart[key].prix = price;
    }
    return `
        <div class="menu-row">
          <div class="menu-row-thumb">
            ${it.image
        ? `<img class="menu-thumb-img" src="/images/${encodeURIComponent(it.image)}" alt="">`
        : `<span class="menu-thumb-fallback">🍽️</span>`}
          </div>
          <div class="menu-row-mid">
            <div class="menu-row-name">${escapeHtml(name)}</div>
            ${t(it.desc) ? `<div class="menu-row-desc">${escapeHtml(t(it.desc))}</div>` : ""}
          </div>
          <div class="menu-row-right">
            <div class="menu-row-price">${price.toFixed(2)} TND</div>
            <div class="stepper">
              <button class="stepper-btn minus" ${qty === 0 ? "disabled" : ""}
                      data-item-key="${key}" data-delta="-1">−</button>
              <span class="stepper-qty">${qty}</span>
              <button class="stepper-btn plus"
                      data-item-key="${key}" data-delta="1">+</button>
            </div>
          </div>
        </div>`;
  }).join("")}
    </div>
  `).join("");

  body.querySelectorAll("img.menu-thumb-img").forEach(img => {
    img.addEventListener("error", () => {
      const fallback = document.createElement("span");
      fallback.className = "menu-thumb-fallback";
      fallback.textContent = "🍽️";
      img.replaceWith(fallback);
    }, { once: true });
  });

  updateMenuFooter();
}

function changeQty(key, delta) {
  const item = menuItemsByKey[key];
  if (!item) return;
  const { nom, prix } = item;
  const current = tactileCart[key] || { qty: 0, nom, prix };
  current.qty = Math.max(0, current.qty + delta);
  current.nom = nom;
  current.prix = prix;
  if (current.qty === 0) delete tactileCart[key];
  else tactileCart[key] = current;

  renderMenu();   // re-rendu simple : le menu reste léger (quelques dizaines de lignes)
}

function updateMenuFooter() {
  const entries = Object.values(tactileCart);
  const count = entries.reduce((sum, e) => sum + e.qty, 0);
  const total = entries.reduce((sum, e) => sum + e.qty * e.prix, 0);

  document.getElementById("menuCartCount").textContent =
    `${count} ${count === 1 ? tr("item") : tr("items")}`;
  document.getElementById("menuCartTotal").textContent = `${total.toFixed(3)} TND`;
  document.getElementById("menuSendBtn").disabled = count === 0 || !TABLE_ID;
}

async function sendTactileOrder() {
  const entries = Object.values(tactileCart);
  if (!entries.length) return;
  if (!TABLE_ID) {
    showToast(tr("tableMissing"), false);
    return;
  }

  const items = entries.map(e => `${e.qty}× ${e.nom}`);
  const total = entries.reduce((sum, e) => sum + e.qty * e.prix, 0);

  const btn = document.getElementById("menuSendBtn");
  btn.disabled = true;
  btn.textContent = "…";

  try {
    await apiFetch("/api/events/order", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        event: "confirmed",
        table: TABLE_ID,
        items,
        total: Math.round(total * 1000) / 1000,
        lang: uiLang,
      }),
    });

    tactileCart = {};
    document.getElementById("menuOverlay").classList.remove("open");
    // flashConfirmation() se déclenche automatiquement via le WebSocket (order_confirmed)
  } catch (e) {
    showToast(uiLang === "fr" && e.message ? e.message : tr("orderSendError"), false);
    btn.textContent = tr("sendOrder");
    btn.disabled = false;
  }
}

// ═══════════════════════════════════════════════════════════════════════
// Appel staff (réutilise le bridge robot_status déjà branché à l'app staff)
// ═══════════════════════════════════════════════════════════════════════
async function callStaff() {
  if (!TABLE_ID) {
    showToast(tr("tableMissing"), false);
    return;
  }
  const button = document.getElementById("callStaffBtn");
  button.disabled = true;
  try {
    await apiFetch("/api/events/robot_status", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        status: "aide_demandee",
        message: "Client demande assistance depuis l'écran",
        table: TABLE_ID,
      }),
    });
    showToast(tr("staffOnWay"), true);
  } catch (error) {
    showToast(uiLang === "fr" && error.message ? error.message : tr("staffCallError"), false);
  } finally {
    button.disabled = false;
  }
}

let toastTimer = null;
function showToast(message, success = true) {
  const toast = document.getElementById("staffToast");
  toast.textContent = message;
  toast.dataset.kind = success ? "success" : "error";
  toast.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("show"), 3500);
}

// ═══════════════════════════════════════════════════════════════════════
// Init
// ═══════════════════════════════════════════════════════════════════════
document.getElementById("langRow").addEventListener("click", (event) => {
  const button = event.target.closest("[data-lang]");
  if (button) selectLang(button.dataset.lang);
});
document.getElementById("menuOpenBtn").addEventListener("click", toggleMenu);
document.getElementById("menuCloseBtn").addEventListener("click", toggleMenu);
document.getElementById("menuSendBtn").addEventListener("click", sendTactileOrder);
document.getElementById("callStaffBtn").addEventListener("click", callStaff);
document.getElementById("menuBody").addEventListener("click", (event) => {
  const button = event.target.closest("button[data-item-key]");
  if (!button) return;
  changeQty(button.dataset.itemKey, Number(button.dataset.delta));
});
document.getElementById("menuCategories").addEventListener("click", (event) => {
  const button = event.target.closest("button[data-category-index]");
  if (!button) return;
  activeMenuCategory = button.dataset.categoryIndex;
  renderMenu();
});

async function refreshClientState() {
  try {
    const response = await apiFetch("/api/client_state");
    handleMessage(await response.json());
  } catch (error) {
    console.warn("Synchronisation écran client impossible", error);
  }
}

selectLang(uiLang);
connect();
setInterval(refreshClientState, 5000);

if (!TABLE_ID) {
  showToast(tr("tableMissing"), false);
}

apiFetch("/api/menu_qr_url")
  .then(r => r.json())
  .then(d => { document.getElementById("qrUrl").textContent = d.url; })
  .catch(() => { });
