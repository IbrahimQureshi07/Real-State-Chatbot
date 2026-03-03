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

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (!question) return;

  addMessage(question, true);
  input.value = "";
  sendBtn.disabled = true;

  try {
    const res = await fetch(API_URL + "/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: question }),
    });
    const data = await res.json();
    addMessage(data.answer || "No answer returned.", false);
  } catch (err) {
    addMessage("Error: " + (err.message || "Could not reach the server. Is the backend running on port 8000?"), false);
  } finally {
    sendBtn.disabled = false;
  }
});
