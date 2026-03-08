// Use same origin when page is served from backend (http://localhost:8000)
const API_URL = window.location.origin;

const form = document.getElementById("form");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send");
const messagesEl = document.getElementById("messages");

// Conversation memory: last 10 messages (5 user + 5 bot) so the bot can recall what was discussed
let chatHistory = [];

/** Escape HTML to prevent XSS, then render **bold**, [text](url), and clean LaTeX-style math for bot messages. */
function renderBotMessage(text) {
  if (!text || typeof text !== "string") return "";
  // Convert LaTeX-style fractions and inline math to plain text so \frac{2}{5} and \( \) don't show raw
  let out = String(text)
    .replace(/\\frac\{([^{}]*)\}\{([^{}]*)\}/g, "$1/$2")
    .replace(/\\\(([^]*?)\\\)/g, "$1")
    .replace(/\\\[([^]*?)\\\]/g, "$1");
  const escape = (s) =>
    String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  out = escape(out);
  // **bold** -> <strong>bold</strong>
  out = out.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  // [link text](url) -> clickable link (open in new tab)
  out = out.replace(/\[([^\]]*)\]\((https?:\/\/[^)\s]+)\)/g, (_, label, url) => {
    const safeUrl = escape(url);
    const safeLabel = escape(label) || safeUrl;
    return '<a href="' + safeUrl + '" target="_blank" rel="noopener noreferrer" class="chat-link">' + safeLabel + "</a>";
  });
  // line breaks
  out = out.replace(/\n/g, "<br>");
  return out;
}

function addMessage(text, isUser, suggestions, lastQuestion) {
  const wrapper = document.createElement("div");
  wrapper.className = "message-wrapper " + (isUser ? "user" : "bot");
  const div = document.createElement("div");
  div.className = "message " + (isUser ? "user" : "bot");
  if (isUser) {
    div.textContent = text;
  } else {
    div.innerHTML = renderBotMessage(text);
  }
  wrapper.appendChild(div);
  if (!isUser && suggestions && suggestions.length > 0) {
    const chipsRow = document.createElement("div");
    chipsRow.className = "suggestion-chips";
    suggestions.forEach((q) => {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "suggestion-chip";
      chip.textContent = q;
      chip.addEventListener("click", () => sendMessage(q));
      chipsRow.appendChild(chip);
    });
    wrapper.appendChild(chipsRow);
  }
  // Copy + "Was this helpful?" only for bot replies from chat (not welcome)
  if (!isUser && lastQuestion !== undefined && lastQuestion !== null) {
    const actions = document.createElement("div");
    actions.className = "message-actions";
    const copyBtn = document.createElement("button");
    copyBtn.type = "button";
    copyBtn.className = "action-btn copy-btn";
    copyBtn.setAttribute("aria-label", "Copy answer");
    copyBtn.innerHTML = "Copy answer";
    copyBtn.addEventListener("click", () => {
      navigator.clipboard.writeText(typeof text === "string" ? text : "").then(() => {
        copyBtn.textContent = "Copied!";
        setTimeout(() => { copyBtn.textContent = "Copy answer"; }, 2000);
      });
    });
    actions.appendChild(copyBtn);
    const helpfulLabel = document.createElement("span");
    helpfulLabel.className = "helpful-label";
    helpfulLabel.textContent = "Was this helpful?";
    actions.appendChild(helpfulLabel);
    const thumbUp = document.createElement("button");
    thumbUp.type = "button";
    thumbUp.className = "action-btn thumb";
    thumbUp.setAttribute("aria-label", "Yes, helpful");
    thumbUp.textContent = "👍";
    const thumbDown = document.createElement("button");
    thumbDown.type = "button";
    thumbDown.className = "action-btn thumb";
    thumbDown.setAttribute("aria-label", "No, not helpful");
    thumbDown.textContent = "👎";
    function sendFeedback(helpful) {
      thumbUp.disabled = true;
      thumbDown.disabled = true;
      thumbUp.classList.toggle("active", helpful === true);
      thumbDown.classList.toggle("active", helpful === false);
      fetch(API_URL + "/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: lastQuestion, answer: (text || "").slice(0, 500), helpful }),
      }).catch(() => {});
    }
    thumbUp.addEventListener("click", () => sendFeedback(true));
    thumbDown.addEventListener("click", () => sendFeedback(false));
    actions.appendChild(thumbUp);
    actions.appendChild(thumbDown);
    wrapper.appendChild(actions);
  }
  messagesEl.appendChild(wrapper);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function showLoader() {
  const div = document.createElement("div");
  div.className = "message bot loader-wrap";
  div.id = "typing-loader";
  div.innerHTML = '<span class="dot"></span><span class="dot"></span><span class="dot"></span>';
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return div;
}

function hideLoader() {
  const el = document.getElementById("typing-loader");
  if (el) el.remove();
}

async function sendMessage(questionFromChip) {
  const q = (typeof questionFromChip === "string" && questionFromChip.trim()) || input.value.trim();
  if (!q) return;
  input.value = "";

  addMessage(q, true);
  sendBtn.disabled = true;
  showLoader();

  try {
    const res = await fetch(API_URL + "/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: q, history: chatHistory }),
    });
    const data = await res.json();
    hideLoader();
    const answer = data.answer || "No answer returned.";
    const suggestions = data.suggestions || [];
    addMessage(answer, false, suggestions, q);
    chatHistory.push({ role: "user", content: q });
    chatHistory.push({ role: "assistant", content: answer });
    if (chatHistory.length > 10) chatHistory = chatHistory.slice(-10);
  } catch (err) {
    hideLoader();
    addMessage("Error: " + (err.message || "Could not reach the server. Is the backend running on port 8000?"), false, [], q);
  } finally {
    sendBtn.disabled = false;
  }
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (!question) return;
  await sendMessage(question);
});

// Dark/Light theme toggle (persisted in localStorage)
(function initTheme() {
  const KEY = "faq-chat-theme";
  const toggle = document.getElementById("theme-toggle");
  const isDark = () => !document.body.classList.contains("theme-light");
  function applyTheme(dark) {
    document.body.classList.toggle("theme-light", !dark);
    if (toggle) toggle.textContent = dark ? "🌙" : "☀️";
  }
  const saved = localStorage.getItem(KEY);
  applyTheme(saved !== "light");
  if (toggle) {
    toggle.addEventListener("click", () => {
      const dark = !isDark();
      localStorage.setItem(KEY, dark ? "dark" : "light");
      applyTheme(dark);
    });
  }
})();

// Friendly welcome message when chat opens
(function showWelcome() {
  const welcome = "Hi there! 👋 I'm your South Carolina Real Estate & Licensing assistant. Ask me anything about licensing requirements, exams, CE hours, fees, applications, or general real estate topics. How can I help you today?";
  const wrapper = document.createElement("div");
  wrapper.className = "message-wrapper bot";
  const div = document.createElement("div");
  div.className = "message bot";
  div.innerHTML = renderBotMessage(welcome);
  wrapper.appendChild(div);
  const chips = document.createElement("div");
  chips.className = "suggestion-chips";
  ["How do I get licensed in SC?", "What are the CE requirements?", "How to apply online?"].forEach((q) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "suggestion-chip";
    chip.textContent = q;
    chip.addEventListener("click", () => sendMessage(q));
    chips.appendChild(chip);
  });
  wrapper.appendChild(chips);
  messagesEl.appendChild(wrapper);
})();
