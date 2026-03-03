// Use same origin when page is served from backend (http://localhost:8000)
const API_URL = window.location.origin;

const form = document.getElementById("form");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send");
const messagesEl = document.getElementById("messages");

function addMessage(text, isUser) {
  const div = document.createElement("div");
  div.className = "message " + (isUser ? "user" : "bot");
  div.textContent = text;
  messagesEl.appendChild(div);
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

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (!question) return;

  addMessage(question, true);
  input.value = "";
  sendBtn.disabled = true;
  showLoader();

  try {
    const res = await fetch(API_URL + "/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: question }),
    });
    const data = await res.json();
    hideLoader();
    addMessage(data.answer || "No answer returned.", false);
  } catch (err) {
    hideLoader();
    addMessage("Error: " + (err.message || "Could not reach the server. Is the backend running on port 8000?"), false);
  } finally {
    sendBtn.disabled = false;
  }
});
