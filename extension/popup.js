const $ = (selector) => document.querySelector(selector);
const labels = ["negative", "neutral", "positive"];
const colors = { negative: "#ee5a52", neutral: "#b0a832", positive: "#44b879" };

function userFacingError(error) {
  const message = String(error?.message || error);
  if (message.includes("Failed to fetch")) return "Cannot reach the API. Keep the local API terminal running, then try again.";
  if (message.includes("String should have at most")) return "A comment was too long. Reload the extension and try again; long comments are now safely shortened.";
  return message.length > 180 ? "The API rejected the request. Reload the extension and try again." : message;
}

async function activeYouTubeTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab?.url?.includes("youtube.com/")) throw new Error("Open a YouTube video first.");
  return tab;
}

function render(data) {
  $("#dashboard").hidden = false;
  $("#total").textContent = data.total.toLocaleString();
  $("#model").textContent = `Inference engine · ${data.model}`;
  const dominant = labels.reduce((best, label) => data.distribution[label] > data.distribution[best] ? label : best, labels[0]);
  $("#dominant").textContent = `${dominant[0].toUpperCase() + dominant.slice(1)} leads`;
  const score = Math.round((data.mean_probabilities.positive - data.mean_probabilities.negative) * 100);
  $("#score").textContent = `${score > 0 ? "+" : ""}${score}`;
  $("#score-caption").textContent = Math.abs(score) < 4 ? "balanced index" : "sentiment index";
  $("#score-orb").style.borderColor = score > 15 ? colors.positive : score < -15 ? colors.negative : colors.neutral;
  $("#bars").innerHTML = labels.map((label) => {
    const count = data.distribution[label], percent = data.total ? (count / data.total) * 100 : 0;
    return `<div class="bar ${label}"><div class="bar-top"><span><i></i>${label}</span><span>${percent.toFixed(0)}%</span></div><b>${count.toLocaleString()}</b><div class="bar-meter"><span style="width:${percent}%"></span></div></div>`;
  }).join("");
  const contexts = data.neutral_contexts ?? {};
  const neutralTotal = Object.values(contexts).reduce((sum, count) => sum + count, 0);
  $("#neutral-insight").hidden = neutralTotal === 0;
  $("#neutral-contexts").innerHTML = [["question", "Questions"], ["spam_or_bot_like", "Spam / short"], ["informational", "Informational"]]
    .map(([key, title]) => `<span>${title}: ${contexts[key] ?? 0}</span>`).join("");
  $("#probability").innerHTML = labels.map((label) => `<span class="${label}" style="width:${data.mean_probabilities[label] * 100}%" title="${label}: ${(data.mean_probabilities[label] * 100).toFixed(1)}%"></span>`).join("");
  $(".probability-labels").innerHTML = labels.map((label) => `<span>${label[0].toUpperCase() + label.slice(1)} ${(data.mean_probabilities[label] * 100).toFixed(1)}%</span>`).join("");
  const canvas = $("#spectrum"), ctx = canvas.getContext("2d"), { width, height } = canvas;
  ctx.clearRect(0, 0, width, height);
  const gradient = ctx.createLinearGradient(8, 0, width - 8, 0); gradient.addColorStop(0, "#ff6572"); gradient.addColorStop(.5, "#d9b952"); gradient.addColorStop(1, "#56cf9a");
  ctx.strokeStyle = gradient; ctx.globalAlpha = .48; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(8, height / 2); ctx.lineTo(width - 8, height / 2); ctx.stroke();
  // Binning turns vertical position into meaningful density, rather than random placement.
  const bins = Array.from({ length: 32 }, () => []);
  data.results.forEach((item) => {
    const score = item.probabilities.positive - item.probabilities.negative;
    bins[Math.max(0, Math.min(31, Math.floor(((score + 1) / 2) * 32)))].push(item);
  });
  const largestBin = Math.max(...bins.map((bin) => bin.length), 1);
  bins.forEach((bin, binIndex) => bin.forEach((item, stackIndex) => {
    const score = item.probabilities.positive - item.probabilities.negative;
    const x = 8 + ((score + 1) / 2) * (width - 16);
    const y = height - 10 - (stackIndex / largestBin) * (height - 22);
    ctx.fillStyle = colors[item.label]; ctx.globalAlpha = .78; ctx.beginPath(); ctx.arc(x, y, 4.2, 0, Math.PI * 2); ctx.fill();
  }));
  ctx.globalAlpha = 1;
}

async function initialise() {
  const { apiEndpoint = "http://127.0.0.1:8000" } = await chrome.storage.sync.get("apiEndpoint");
  $("#endpoint").value = apiEndpoint;
  $("#endpoint").addEventListener("change", () => chrome.storage.sync.set({ apiEndpoint: $("#endpoint").value.trim() }));
  $("#analyse").addEventListener("click", async () => {
    const button = $("#analyse"), status = $("#status");
    button.disabled = true; status.textContent = "Collecting comments and running inference…";
    try {
      await chrome.storage.sync.set({ apiEndpoint: $("#endpoint").value.trim() });
      const tab = await activeYouTubeTab();
      const response = await chrome.runtime.sendMessage({ type: "ANALYSE", tabId: tab.id, limit: 1000 });
      if (!response.ok) throw new Error(response.error);
      render(response.data); status.textContent = `Completed ${response.collected.toLocaleString()} comments.`;
    } catch (error) { status.textContent = `Error: ${userFacingError(error)}`; }
    finally { button.disabled = false; }
  });
}
initialise();
