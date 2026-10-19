(() => {
  const $ = (s) => document.querySelector(s);
  const tabs = document.querySelectorAll(".tab-btn");
  const panelKnowledge = $("#panel-knowledge");
  const panelChat = $("#panel-chat");
  const dropzone = $("#dropzone");
  const fileInput = $("#file-input");
  const fileList = $("#file-list");
  const websiteUrl = $("#website-url");
  const ingestBtn = $("#ingest-btn");
  const ingestStatus = $("#ingest-status");
  const progressWrap = $("#learn-progress");
  const progressLabel = $("#learn-progress-label");
  const progressPct = $("#learn-progress-pct");
  const progressBar = $("#learn-progress-bar");
  const progressTrack = progressWrap?.querySelector(".learn-progress-track");
  const crawlLog = $("#crawl-log");
  const crawlLogList = $("#crawl-log-list");
  const crawlLogSummary = $("#crawl-log-summary");
  const chatLog = $("#chat-log");
  const chatForm = $("#chat-form");
  const chatInput = $("#chat-input");
  const langSelect = $("#lang-select");
  const modelSelect = $("#model-select");
  const knowledgeList = $("#knowledge-list");
  const knowledgeEmpty = $("#knowledge-empty");

  const STATUS_FA = {
    visiting: "در حال خواندن",
    learned: "آموخته شد",
    skipped: "رد شد",
    failed: "ناموفق",
  };

  const sessionId =
    localStorage.getItem("javad_session") ||
    (() => {
      const id = `نشست_${Math.random().toString(36).slice(2, 9)}`;
      localStorage.setItem("javad_session", id);
      return id;
    })();

  if (localStorage.getItem("javad_lang")) {
    langSelect.value = localStorage.getItem("javad_lang");
  }
  if (localStorage.getItem("javad_model")) {
    // populated after config load
  }

  langSelect.addEventListener("change", () => {
    localStorage.setItem("javad_lang", langSelect.value);
  });
  modelSelect.addEventListener("change", () => {
    localStorage.setItem("javad_model", modelSelect.value);
  });

  let selectedFiles = [];
  /** @type {Map<string, {url:string,status:string,depth:number,title:string,domain:string}>} */
  let crawlRows = new Map();

  tabs.forEach((btn) => {
    btn.addEventListener("click", () => {
      const tab = btn.dataset.tab;
      tabs.forEach((b) => {
        b.classList.toggle("active", b === btn);
        b.setAttribute("aria-selected", b === btn ? "true" : "false");
      });
      panelKnowledge.classList.toggle("hidden", tab !== "knowledge");
      panelChat.classList.toggle("hidden", tab !== "chat");
      if (tab === "chat") chatInput.focus();
      if (tab === "knowledge") refreshKnowledge();
    });
  });

  function renderFiles() {
    fileList.innerHTML = selectedFiles.length
      ? selectedFiles.map((f) => `<span class="block">${escapeHtml(f.name)}</span>`).join("")
      : "";
  }

  fileInput.addEventListener("change", () => {
    selectedFiles = Array.from(fileInput.files || []);
    renderFiles();
  });

  ["dragenter", "dragover"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    });
  });
  ["dragleave", "drop"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
    });
  });
  dropzone.addEventListener("drop", (e) => {
    if (e.dataTransfer?.files?.length) {
      selectedFiles = Array.from(e.dataTransfer.files);
      fileInput.files = e.dataTransfer.files;
      renderFiles();
    }
  });

  function setProgress(percent, message, state) {
    const pct = Math.max(0, Math.min(100, Math.round(Number(percent) || 0)));
    progressWrap.hidden = false;
    progressWrap.classList.toggle("is-done", state === "done");
    progressWrap.classList.toggle("is-error", state === "error");
    progressBar.style.width = `${pct}%`;
    progressPct.textContent = `${pct}٪`;
    if (message) progressLabel.textContent = message;
    if (progressTrack) progressTrack.setAttribute("aria-valuenow", String(pct));
  }

  function hideProgressSoon(delay = 3200) {
    clearTimeout(hideProgressSoon._t);
    hideProgressSoon._t = setTimeout(() => {
      progressWrap.hidden = true;
      progressWrap.classList.remove("is-done", "is-error");
      progressBar.style.width = "0%";
    }, delay);
  }

  function resetCrawlLog() {
    crawlRows = new Map();
    crawlLogList.innerHTML = "";
    crawlLogSummary.textContent = "";
    crawlLog.hidden = true;
  }

  function upsertCrawlLink(link) {
    if (!link || !link.url) return;
    const prev = crawlRows.get(link.url);
    const rank = { visiting: 1, skipped: 2, failed: 2, learned: 3 };
    if (prev && (rank[prev.status] || 0) > (rank[link.status] || 0)) {
      link = { ...link, status: prev.status, title: link.title || prev.title };
    }
    crawlRows.set(link.url, {
      url: link.url,
      status: link.status || "visiting",
      depth: Number(link.depth) || 0,
      title: link.title || (prev && prev.title) || "",
      domain: link.domain || (prev && prev.domain) || "",
    });
    renderCrawlLog();
  }

  function renderCrawlLog() {
    if (!crawlRows.size) {
      crawlLog.hidden = true;
      return;
    }
    crawlLog.hidden = false;
    const rows = Array.from(crawlRows.values());
    const learned = rows.filter((r) => r.status === "learned").length;
    const failed = rows.filter((r) => r.status === "failed").length;
    const skipped = rows.filter((r) => r.status === "skipped").length;
    const domain = rows.find((r) => r.domain)?.domain || "";
    crawlLogSummary.textContent = domain
      ? `دامنه: ${domain} · بازدید: ${rows.length} · آموخته: ${learned}` +
        (skipped ? ` · ردشده: ${skipped}` : "") +
        (failed ? ` · ناموفق: ${failed}` : "")
      : `بازدید: ${rows.length} · آموخته: ${learned}`;

    crawlLogList.innerHTML = rows
      .map((r) => {
        const status = STATUS_FA[r.status] || r.status;
        const titleBit = r.title ? escapeHtml(r.title) : "بدون عنوان";
        return `<li class="crawl-log-item">
          <span class="crawl-log-status is-${escapeHtml(r.status)}">${escapeHtml(status)}</span>
          <span class="crawl-log-url" dir="ltr">${escapeHtml(r.url)}</span>
          <span class="crawl-log-meta">عمق ${r.depth} · ${titleBit}</span>
        </li>`;
      })
      .join("");
    crawlLogList.scrollTop = crawlLogList.scrollHeight;
  }

  async function readNdjsonStream(response, onEvent) {
    if (!response.body || !response.body.getReader) {
      const text = await response.text();
      for (const line of text.split("\n")) {
        if (!line.trim()) continue;
        onEvent(JSON.parse(line));
      }
      return;
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let nl;
      while ((nl = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, nl).trim();
        buffer = buffer.slice(nl + 1);
        if (!line) continue;
        onEvent(JSON.parse(line));
      }
    }
    const tail = buffer.trim();
    if (tail) onEvent(JSON.parse(tail));
  }

  ingestBtn.addEventListener("click", async () => {
    const url = websiteUrl.value.trim();
    if (!selectedFiles.length && !url) {
      ingestStatus.textContent = "لطفاً فایل یا آدرس وب‌سایت وارد کنید.";
      return;
    }
    const form = new FormData();
    selectedFiles.forEach((f) => form.append("files", f));
    if (url) form.append("url", url);
    form.append("max_depth", String($("#crawl-depth")?.value || 2));
    form.append("max_pages", String($("#crawl-pages")?.value || 20));

    ingestBtn.disabled = true;
    clearTimeout(hideProgressSoon._t);
    resetCrawlLog();
    setProgress(3, url ? "در حال پیمایش صفحات داخلی وب‌سایت…" : "در حال یادگیری…");
    ingestStatus.textContent = "";

    let finalMessage = "";
    let failed = false;

    try {
      const res = await fetch("/api/rag/upload/stream", { method: "POST", body: form });
      if (!res.ok) {
        let detail = "خطا در یادگیری";
        try {
          const errBody = await res.json();
          detail = _detail(errBody) || detail;
        } catch {
          /* ignore */
        }
        throw new Error(detail);
      }

      await readNdjsonStream(res, (evt) => {
        if (!evt || typeof evt !== "object") return;
        if (evt.type === "progress") {
          setProgress(evt.percent ?? 0, evt.message || "در حال یادگیری…");
          ingestStatus.textContent = evt.message || "";
          if (evt.link) upsertCrawlLink(evt.link);
          if (Array.isArray(evt.crawl_log)) {
            evt.crawl_log.forEach((row) => upsertCrawlLink(row));
          }
        } else if (evt.type === "error") {
          failed = true;
          finalMessage = evt.message || "خطا در یادگیری";
          setProgress(evt.percent ?? 100, finalMessage, "error");
        } else if (evt.type === "done") {
          finalMessage = evt.message || "یادگیری کامل شد.";
          setProgress(100, finalMessage, "done");
          const results = Array.isArray(evt.results) ? evt.results : [];
          results.forEach((r) => {
            if (Array.isArray(r.crawl_log)) r.crawl_log.forEach((row) => upsertCrawlLink(row));
          });
        } else if (evt.type === "item" && evt.result) {
          const r = evt.result;
          if (Array.isArray(r.crawl_log)) r.crawl_log.forEach((row) => upsertCrawlLink(row));
        }
      });

      if (failed) throw new Error(finalMessage || "خطا در یادگیری");
      if (!finalMessage) finalMessage = "یادگیری کامل شد.";

      ingestStatus.textContent = finalMessage;
      showToast(finalMessage);
      selectedFiles = [];
      fileInput.value = "";
      websiteUrl.value = "";
      renderFiles();
      hideProgressSoon();
      refreshHealth();
      refreshKnowledge();
    } catch (err) {
      const msg = err.message || "خطا در یادگیری";
      ingestStatus.textContent = msg;
      setProgress(100, msg, "error");
      showToast(msg, true);
      hideProgressSoon(4500);
    } finally {
      ingestBtn.disabled = false;
    }
  });

  $("#clear-memory-btn").addEventListener("click", async () => {
    if (!confirm("آیا مطمئن هستید؟ تمام آموخته‌ها و گفتگوها پاک می‌شود.")) return;
    const btn = $("#clear-memory-btn");
    btn.disabled = true;
    try {
      const res = await fetch("/api/memory/clear", { method: "POST" });
      const data = await res.json();
      if (!res.ok) throw new Error(_detail(data) || "خطا");
      const msg = data.message || "حافظه پاک شد.";
      ingestStatus.textContent = msg;
      showToast(msg);
      chatLog.innerHTML = "";
      appendBubble("javad", "جواد", "حافظه‌ام پاک شد. از نو آماده‌ام.");
      refreshHealth();
      refreshKnowledge();
    } catch (err) {
      showToast(err.message || "خطا در پاکسازی", true);
    } finally {
      btn.disabled = false;
    }
  });

  $("#refresh-knowledge-btn")?.addEventListener("click", () => refreshKnowledge());

  async function refreshKnowledge() {
    try {
      const res = await fetch("/api/knowledge");
      const data = await res.json();
      const sources = data.sources || [];
      if (!sources.length) {
        knowledgeList.innerHTML = "";
        knowledgeEmpty.hidden = false;
        return;
      }
      knowledgeEmpty.hidden = true;
      knowledgeList.innerHTML = sources
        .map((s) => {
          const date = s.updated_at
            ? new Date(s.updated_at * 1000).toLocaleString("fa-IR")
            : "";
          return `<li class="knowledge-item flex items-start justify-between gap-3 border border-ink-100 px-3 py-2">
            <div class="min-w-0">
              <p class="font-medium text-ink-700 truncate">${escapeHtml(s.name)}</p>
              <p class="text-xs text-ink-400 mt-0.5">${escapeHtml(s.kind)} · ${s.chunk_count || 0} قطعه${date ? " · " + date : ""}</p>
            </div>
            <button type="button" class="shrink-0 text-xs text-red-800 border border-red-200 px-2 py-1 hover:bg-red-50" data-delete-source="${escapeHtml(s.id)}">حذف</button>
          </li>`;
        })
        .join("");
      knowledgeList.querySelectorAll("[data-delete-source]").forEach((btn) => {
        btn.addEventListener("click", async () => {
          const id = btn.getAttribute("data-delete-source");
          if (!id || !confirm("این منبع حذف شود؟")) return;
          const res = await fetch(`/api/rag/source/${id}`, { method: "DELETE" });
          const data = await res.json();
          if (!res.ok) {
            showToast(_detail(data) || "حذف ناموفق", true);
            return;
          }
          showToast(data.message || "حذف شد");
          refreshKnowledge();
          refreshHealth();
        });
      });
    } catch {
      knowledgeEmpty.hidden = false;
      knowledgeEmpty.textContent = "بارگذاری فهرست دانش ممکن نشد.";
    }
  }

  chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const prompt = chatInput.value.trim();
    if (!prompt) return;
    appendBubble("user", "شما", prompt);
    chatInput.value = "";
    chatInput.disabled = true;
    const bubble = appendBubble("javad", "جواد", "");
    const p = bubble.querySelector("p");
    p.textContent = "";
    let citations = [];
    let sources = [];
    try {
      const res = await fetch("/api/ollama/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          session_id: sessionId,
          language: langSelect.value || "fa",
          model: modelSelect.value || null,
        }),
      });
      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}));
        throw new Error(_detail(errBody) || "خطا در گفتگو");
      }
      let full = "";
      await readNdjsonStream(res, (evt) => {
        if (!evt || typeof evt !== "object") return;
        if (evt.type === "meta") {
          citations = evt.citations || [];
          sources = evt.sources || [];
        } else if (evt.type === "token") {
          if (evt.replace) {
            full = evt.text || "";
          } else {
            full += evt.text || "";
          }
          p.textContent = full;
          chatLog.scrollTop = chatLog.scrollHeight;
        } else if (evt.type === "done") {
          full = evt.answer || full;
          p.textContent = full;
          citations = evt.citations || citations;
          sources = evt.sources || sources;
          attachCitations(bubble, citations, sources);
        }
      });
      if (!p.textContent) p.textContent = "پاسخی دریافت نشد.";
    } catch (err) {
      p.textContent = err.message || "خطایی رخ داد.";
    } finally {
      chatInput.disabled = false;
      chatInput.focus();
    }
  });

  function attachCitations(bubble, citations, sources) {
    const existing = bubble.querySelector(".chat-sources");
    if (existing) existing.remove();
    const items = Array.isArray(citations) && citations.length
      ? citations
      : (sources || []).map((s) => ({ source: s, excerpt: "" }));
    if (!items.length) return;
    const wrap = document.createElement("div");
    wrap.className = "chat-sources";
    wrap.innerHTML =
      "<span class=\"chat-sources-label\">منابع:</span> " +
      items
        .map((c, i) => {
          const title = escapeHtml(c.source || "منبع");
          const excerpt = c.excerpt ? ` — ${escapeHtml(c.excerpt)}` : "";
          return `<button type="button" class="chat-cite" title="${excerpt}" data-cite="${i}">[${i + 1}] ${title}</button>`;
        })
        .join(" ");
    bubble.appendChild(wrap);
    wrap.querySelectorAll(".chat-cite").forEach((btn) => {
      btn.addEventListener("click", () => {
        const idx = Number(btn.getAttribute("data-cite"));
        const c = items[idx];
        if (c?.excerpt) showToast(`${c.source}: ${c.excerpt}`);
        else if (c?.source) showToast(c.source);
      });
    });
  }

  async function refreshHealth() {
    const badge = $("#ollama-badge");
    try {
      const res = await fetch("/api/health");
      const data = await res.json();
      const bits = [];
      if (data.ollama) bits.push("اولاما آماده");
      else bits.push("اولاما خاموش");
      bits.push(`دانش: ${data.chunks || 0} قطعه`);
      if (data.vector_store) bits.push(`بردار: ${data.vector_store}`);
      if (data.embed_ok) bits.push("امبدینگ آماده");
      badge.textContent = bits.join(" · ");
      if (Array.isArray(data.models) && data.models.length) {
        const current = localStorage.getItem("javad_model") || "";
        const opts = ['<option value="">پیش‌فرض سامانه</option>'].concat(
          data.models.map(
            (m) =>
              `<option value="${escapeHtml(m)}"${m === current || m === data.chat_model ? " selected" : ""}>${escapeHtml(m)}</option>`
          )
        );
        modelSelect.innerHTML = opts.join("");
        if (current) modelSelect.value = current;
      }
    } catch {
      badge.textContent = "وضعیت سامانه مشخص نیست.";
    }
  }

  function appendBubble(role, label, text) {
    const el = document.createElement("div");
    el.className = `chat-bubble ${role}`;
    el.innerHTML = `<span class="chat-label">${escapeHtml(label)}</span><p></p>`;
    el.querySelector("p").textContent = text;
    chatLog.appendChild(el);
    chatLog.scrollTop = chatLog.scrollHeight;
    return el;
  }

  function showToast(message, isError = false) {
    const toast = $("#toast");
    toast.hidden = false;
    toast.textContent = message;
    toast.classList.toggle("toast-error", !!isError);
    toast.classList.add("toast-show");
    clearTimeout(showToast._t);
    showToast._t = setTimeout(() => {
      toast.classList.remove("toast-show");
      setTimeout(() => {
        toast.hidden = true;
      }, 280);
    }, 4200);
  }

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function _detail(data) {
    const d = data && data.detail;
    if (!d) return "";
    if (typeof d === "string") return d;
    if (Array.isArray(d)) return d.map((x) => x.msg || "").join(" · ");
    return String(d);
  }

  refreshHealth();
  refreshKnowledge();
})();
