const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const MAX_COMMENT_LENGTH = 2000;

function extractComments() {
  const seen = new Map();
  document.querySelectorAll("ytd-comment-thread-renderer").forEach((thread, index) => {
    const node = thread.querySelector("#content-text");
    // Keep the API contract intact even when YouTube renders an unusually long comment.
    const text = node?.innerText?.trim().slice(0, MAX_COMMENT_LENGTH);
    if (!text) return;
    const id = thread.getAttribute("comment-id") || thread.id || `comment-${index}-${text.slice(0, 24)}`;
    seen.set(id, { id, text });
  });
  return [...seen.values()];
}

async function collectComments(limit) {
  let comments = extractComments();
  let stalled = 0;
  // Scroll only the page, in bounded cycles, and let YouTube's lazy loader populate comment renderers.
  while (comments.length < limit && stalled < 6) {
    const before = comments.length;
    window.scrollTo({ top: document.documentElement.scrollHeight, behavior: "smooth" });
    await wait(1400);
    comments = extractComments();
    stalled = comments.length === before ? stalled + 1 : 0;
  }
  return { comments: comments.slice(0, limit) };
}

chrome.runtime.onMessage.addListener((message, _sender, respond) => {
  if (message.type !== "COLLECT_COMMENTS") return;
  collectComments(Math.min(Number(message.limit) || 1000, 1000)).then(respond).catch((error) => respond({ comments: [], error: error.message }));
  return true;
});
