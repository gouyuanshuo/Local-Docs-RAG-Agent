const askButton = document.getElementById("ask-button");
const ingestButton = document.getElementById("ingest-button");
const evalButton = document.getElementById("eval-button");
const questionInput = document.getElementById("question-input");
const runtimeSelect = document.getElementById("runtime-select");
const askOutput = document.getElementById("ask-output");
const actionOutput = document.getElementById("action-output");
const healthPill = document.getElementById("health-pill");
const infoPill = document.getElementById("info-pill");

async function requestJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed: ${response.status}`);
  }
  return response.json();
}

function pretty(data) {
  return JSON.stringify(data, null, 2);
}

function selectedRuntime() {
  return runtimeSelect.value || null;
}

async function bootstrap() {
  try {
    const [health, info] = await Promise.all([
      requestJson("/api/health"),
      requestJson("/api/info"),
    ]);
    healthPill.textContent = `Backend: ${health.status}`;
    infoPill.textContent = `Runtime ${info.runtime} · Backend ${info.vector_backend}`;
  } catch (error) {
    healthPill.textContent = "Backend unavailable";
    infoPill.textContent = String(error.message || error);
  }
}

askButton.addEventListener("click", async () => {
  askOutput.textContent = "Thinking...";
  try {
    const payload = await requestJson("/api/ask", {
      method: "POST",
      body: JSON.stringify({
        question: questionInput.value.trim(),
        runtime: selectedRuntime(),
      }),
    });
    askOutput.textContent = pretty(payload);
  } catch (error) {
    askOutput.textContent = `Error:\n${String(error.message || error)}`;
  }
});

ingestButton.addEventListener("click", async () => {
  actionOutput.textContent = "Running ingest...";
  try {
    const payload = await requestJson("/api/ingest", { method: "POST" });
    actionOutput.textContent = pretty(payload);
    bootstrap();
  } catch (error) {
    actionOutput.textContent = `Error:\n${String(error.message || error)}`;
  }
});

evalButton.addEventListener("click", async () => {
  actionOutput.textContent = "Running eval...";
  try {
    const payload = await requestJson("/api/eval", {
      method: "POST",
      body: JSON.stringify({ runtime: selectedRuntime() }),
    });
    actionOutput.textContent = pretty(payload);
  } catch (error) {
    actionOutput.textContent = `Error:\n${String(error.message || error)}`;
  }
});

bootstrap();
