const DEFAULT_ENDPOINT = "http://127.0.0.1:8000";
const CHUNK_SIZE = 200;

async function endpoint() {
  const saved = await chrome.storage.sync.get({ apiEndpoint: DEFAULT_ENDPOINT });
  return saved.apiEndpoint.replace(/\/$/, "");
}

async function requestAnalysis(comments) {
  const api = await endpoint();
  const safeComments = comments
    .map(({ id, text }) => ({ id: String(id).slice(0, 256), text: String(text).trim().slice(0, 2000) }))
    .filter((comment) => comment.id && comment.text);
  if (!safeComments.length) throw new Error("No usable comment text was found on this page.");
  const batches = [];
  for (let start = 0; start < safeComments.length; start += CHUNK_SIZE) batches.push(safeComments.slice(start, start + CHUNK_SIZE));
  const responses = [];
  for (const batch of batches) {
    const response = await fetch(`${api}/v1/sentiment/analyze`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ comments: batch })
    });
    if (!response.ok) {
      const body = await response.json().catch(() => null);
      const detail = Array.isArray(body?.detail) ? body.detail[0]?.msg : body?.detail;
      throw new Error(detail ? `API rejected this batch: ${detail}` : `The API returned HTTP ${response.status}.`);
    }
    responses.push(await response.json());
  }
  const results = responses.flatMap((item) => item.results);
  const labels = ["negative", "neutral", "positive"];
  const distribution = Object.fromEntries(labels.map((label) => [label, results.filter((item) => item.label === label).length]));
  const mean_probabilities = Object.fromEntries(labels.map((label) => [label, results.reduce((sum, item) => sum + item.probabilities[label], 0) / results.length]));
  const neutral_contexts = Object.fromEntries(["question", "spam_or_bot_like", "informational"].map((context) => [context, results.filter((item) => item.neutral_context === context).length]));
  return { model: responses[0].model, total: results.length, distribution, mean_probabilities, neutral_contexts, results };
}

async function collectFromTab(tabId, limit) {
  try {
    return await chrome.tabs.sendMessage(tabId, { type: "COLLECT_COMMENTS", limit });
  } catch (error) {
    // Covers a page that was already open when the extension was installed.
    if (!String(error.message).includes("Receiving end does not exist")) throw error;
    await chrome.scripting.executeScript({ target: { tabId }, files: ["content.js"] });
    return chrome.tabs.sendMessage(tabId, { type: "COLLECT_COMMENTS", limit });
  }
}

chrome.runtime.onMessage.addListener((message, sender, respond) => {
  if (message.type !== "ANALYSE") return;
  (async () => {
    const tabId = sender.tab?.id ?? message.tabId;
    if (!tabId) throw new Error("No active YouTube tab found.");
    const collected = await collectFromTab(tabId, message.limit ?? 1000);
    if (!collected.comments.length) throw new Error("No comments found. Open a YouTube video and scroll to its comments first.");
    respond({ ok: true, data: await requestAnalysis(collected.comments), collected: collected.comments.length });
  })().catch((error) => respond({ ok: false, error: error.message }));
  return true;
});
