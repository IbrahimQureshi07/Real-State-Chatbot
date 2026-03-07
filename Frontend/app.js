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

function addMessage(text, isUser, suggestions) {
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
    addMessage(answer, false, suggestions);
    chatHistory.push({ role: "user", content: q });
    chatHistory.push({ role: "assistant", content: answer });
    if (chatHistory.length > 10) chatHistory = chatHistory.slice(-10);
  } catch (err) {
    hideLoader();
    addMessage("Error: " + (err.message || "Could not reach the server. Is the backend running on port 8000?"), false);
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
