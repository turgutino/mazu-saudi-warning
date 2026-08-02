/* MAZU-FENGYUN floating AI chat widget — embeds the live agent chat via a
   Cloudflare quick-tunnel HTTPS URL fronting the HTTP-only chat backend.
   NOTE: quick-tunnel URLs are ephemeral — update CHAT_BASE if the tunnel restarts. */
(function () {
  var CHAT_BASE = "https://calculate-appropriations-simulation-knows.trycloudflare.com";
  var ACCESS_CODE = "fw1as9Ng0KOZ";
  var CHAT_URL = CHAT_BASE + "/?code=" + encodeURIComponent(ACCESS_CODE);

  var style = document.createElement("style");
  style.textContent = [
    "#mazu-chat-fab{position:fixed;right:22px;bottom:22px;width:58px;height:58px;border-radius:50%;",
    "background:linear-gradient(135deg,#2BC8E2,#1a8fae);border:none;cursor:pointer;z-index:9998;",
    "box-shadow:0 8px 24px rgba(0,0,0,.4);display:flex;align-items:center;justify-content:center;",
    "font-size:26px;transition:transform .15s;color:#08131f}",
    "#mazu-chat-fab:hover{transform:scale(1.07)}",
    "#mazu-chat-overlay{position:fixed;inset:0;background:rgba(6,12,20,.55);backdrop-filter:blur(2px);",
    "z-index:9999;display:none;align-items:center;justify-content:center;padding:20px}",
    "#mazu-chat-overlay.open{display:flex}",
    "#mazu-chat-panel{position:relative;width:100%;max-width:900px;height:min(760px,92vh);",
    "background:#0E1B2A;border:1px solid #243B54;border-radius:16px;overflow:hidden;",
    "box-shadow:0 20px 60px rgba(0,0,0,.5)}",
    "#mazu-chat-frame{width:100%;height:100%;border:none;display:block}",
    "#mazu-chat-close{position:absolute;top:10px;right:12px;z-index:2;width:32px;height:32px;",
    "border-radius:50%;border:1px solid #243B54;background:#16283C;color:#E6EEF6;font-size:18px;",
    "cursor:pointer;line-height:1}",
    "#mazu-chat-close:hover{border-color:#2BC8E2;color:#2BC8E2}",
    "@media (max-width:480px){#mazu-chat-panel{height:100vh;border-radius:0}#mazu-chat-overlay{padding:0}}"
  ].join("");
  document.head.appendChild(style);

  var fab = document.createElement("button");
  fab.id = "mazu-chat-fab";
  fab.title = "Ask the MAZU AI agent";
  fab.setAttribute("aria-label", "Open MAZU AI chat");
  fab.textContent = "💬"; // 💬
  document.body.appendChild(fab);

  var overlay = document.createElement("div");
  overlay.id = "mazu-chat-overlay";
  overlay.innerHTML =
    '<div id="mazu-chat-panel">' +
    '<button id="mazu-chat-close" aria-label="Close chat">&times;</button>' +
    '<iframe id="mazu-chat-frame" title="MAZU AI Chat" src="about:blank" loading="lazy"></iframe>' +
    "</div>";
  document.body.appendChild(overlay);

  var frame = document.getElementById("mazu-chat-frame");
  var close = document.getElementById("mazu-chat-close");
  var loaded = false;

  function open() {
    if (!loaded) {
      frame.src = CHAT_URL;
      loaded = true;
    }
    overlay.classList.add("open");
  }
  function shut() {
    overlay.classList.remove("open");
  }

  fab.addEventListener("click", open);
  close.addEventListener("click", shut);
  overlay.addEventListener("click", function (e) {
    if (e.target === overlay) shut();
  });
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape" && overlay.classList.contains("open")) shut();
  });
})();
