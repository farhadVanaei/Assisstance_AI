(() => {
  const $ = (s) => document.querySelector(s);
  const viewButtons = document.querySelectorAll(".view-switch-btn");
  const viewChat = $("#view-chat");
  const viewLearning = $("#view-learning");
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
  const sessionListEl = $("#session-list");
  const newChatBtn = $("#new-chat-btn");
  const chatTitle = $("#chat-title");
  const micBtn = $("#mic-btn");
  const voiceStatus = $("#voice-status");

  const STATUS_FA = {
    visiting: "در حال خواندن",
    learned: "آموخته شد",
    skipped: "رد شد",
    unchanged: "بدون تغییر",
    failed: "ناموفق",
  };

  function newSessionId() {
    return `نشست_${Math.random().toString(36).slice(2, 9)}`;
  }

  let currentSessionId =
    localStorage.getItem("javad_session") ||
    (() => {
      const id = newSessionId();
      localStorage.setItem("javad_session", id);
      return id;
    })();

  if (localStorage.getItem("javad_lang")) {
    langSelect.value = localStorage.getItem("javad_lang");
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

  function showView(view) {
    viewButtons.forEach((b) => {
      const active = b.dataset.view === view;
      b.classList.toggle("is-active", active);
      b.setAttribute("aria-selected", active ? "true" : "false");
    });
    viewChat.hidden = view !== "chat";
    viewLearning.hidden = view !== "learning";
    if (view === "chat") chatInput.focus();
    if (view === "learning") refreshKnowledge();
  }
  viewButtons.forEach((btn) => {
    btn.addEventListener("click", () => showView(btn.dataset.view));
  });

  // ---------- Chat sessions (new / switch / rename) ----------

  async function refreshSessions() {
    try {
      const res = await fetch("/api/sessions");
      const data = await res.json();
      renderSessionList(data.sessions || []);
    } catch {
      /* non-fatal */
    }
  }

  function renderSessionList(sessions) {
    if (!sessions.length) {
      sessionListEl.innerHTML = "";
      return;
    }
    sessionListEl.innerHTML = sessions
      .map((s) => {
        const active = s.id === currentSessionId;
        return `<li class="session-item${active ? " is-active" : ""}" data-session-id="${escapeHtml(s.id)}">
          <div class="session-item-main">
            <p class="session-item-title">${escapeHtml(s.title || "گفتگوی جدید")}</p>
            ${s.preview ? `<p class="session-item-preview">${escapeHtml(s.preview)}</p>` : ""}
          </div>
          <button type="button" class="session-item-btn" data-rename-session="${escapeHtml(s.id)}" title="تغییر نام" aria-label="تغییر نام">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="m16.5 3.5 4 4L7 21l-4.5 1 1-4.5L16.5 3.5Z" stroke-linecap="round" stroke-linejoin="round"/></svg>
          </button>
        </li>`;
      })
      .join("");
    sessionListEl.querySelectorAll("[data-session-id]").forEach((li) => {
      li.addEventListener("click", (e) => {
        if (e.target.closest("[data-rename-session]")) return;
        switchSession(li.getAttribute("data-session-id"));
      });
    });
    sessionListEl.querySelectorAll("[data-rename-session]").forEach((btn) => {
      btn.addEventListener("click", async (e) => {
        e.stopPropagation();
        const id = btn.getAttribute("data-rename-session");
        const current = sessions.find((s) => s.id === id);
        const title = prompt("نام جدید گفتگو:", current?.title || "");
        if (!title || !title.trim()) return;
        try {
          const res = await fetch(`/api/sessions/${encodeURIComponent(id)}/rename`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ title: title.trim() }),
          });
          if (!res.ok) throw new Error("تغییر نام ناموفق بود.");
          if (id === currentSessionId) chatTitle.textContent = title.trim();
          refreshSessions();
        } catch (err) {
          showToast(err.message || "تغییر نام ناموفق بود.", true);
        }
      });
    });
  }

  async function switchSession(id) {
    if (!id || id === currentSessionId) return;
    currentSessionId = id;
    localStorage.setItem("javad_session", id);
    await loadSessionMessages(id);
    refreshSessions();
  }

  async function loadSessionMessages(id) {
    chatLog.innerHTML = "";
    try {
      const res = await fetch(`/api/sessions/${encodeURIComponent(id)}/messages`);
      const data = await res.json();
      const messages = Array.isArray(data.messages) ? data.messages : [];
      if (!messages.length) {
        appendBubble("javad", "جواد", "سلام! من جواد هستم. اسناد را بارگذاری کنید یا مستقیم با من حرف بزنید.");
        return;
      }
      messages.forEach((m) => {
        const role = m.role === "assistant" ? "javad" : "user";
        appendBubble(role, role === "javad" ? "جواد" : "شما", m.content || "");
      });
    } catch {
      appendBubble("javad", "جواد", "بارگذاری این گفتگو ممکن نشد.");
    }
  }

  newChatBtn?.addEventListener("click", () => {
    currentSessionId = newSessionId();
    localStorage.setItem("javad_session", currentSessionId);
    chatLog.innerHTML = "";
    appendBubble("javad", "جواد", "سلام! من جواد هستم. اسناد را بارگذاری کنید یا مستقیم با من حرف بزنید.");
    chatTitle.textContent = "گفتگوی زنده با جواد";
    renderSessionList(
      Array.from(sessionListEl.querySelectorAll("[data-session-id]")).map((li) => ({
        id: li.getAttribute("data-session-id"),
        title: li.querySelector(".session-item-title")?.textContent || "",
        preview: li.querySelector(".session-item-preview")?.textContent || "",
      }))
    );
    refreshSessions();
    chatInput.focus();
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
    progressWrap.classList.toggle("is-warning", state === "warning");
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
      progressWrap.classList.remove("is-done", "is-warning", "is-error");
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
    const rank = { visiting: 1, skipped: 2, unchanged: 2, failed: 2, learned: 3 };
    if (prev && (rank[prev.status] || 0) > (rank[link.status] || 0)) {
      link = {
        ...link,
        status: prev.status,
        title: link.title || prev.title,
        reason: link.reason || prev.reason,
      };
    }
    crawlRows.set(link.url, {
      url: link.url,
      status: link.status || "visiting",
      depth: Number(link.depth) || 0,
      title: link.title || (prev && prev.title) || "",
      domain: link.domain || (prev && prev.domain) || "",
      reason: link.reason || (prev && prev.reason) || "",
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
    const unchanged = rows.filter((r) => r.status === "unchanged").length;
    const domain = rows.find((r) => r.domain)?.domain || "";
    crawlLogSummary.textContent = domain
      ? `دامنه: ${domain} · بازدید: ${rows.length} · آموخته: ${learned}` +
        (unchanged ? ` · بدون‌تغییر: ${unchanged}` : "") +
        (skipped ? ` · ردشده: ${skipped}` : "") +
        (failed ? ` · ناموفق: ${failed}` : "")
      : `بازدید: ${rows.length} · آموخته: ${learned}`;

    crawlLogList.innerHTML = rows
      .map((r) => {
        const status = STATUS_FA[r.status] || r.status;
        const titleBit = r.title ? escapeHtml(r.title) : "بدون عنوان";
        const reasonBit = r.reason
          ? `<span class="crawl-log-reason">${escapeHtml(r.reason)}</span>`
          : "";
        return `<li class="crawl-log-item">
          <span class="crawl-log-status is-${escapeHtml(r.status)}">${escapeHtml(status)}</span>
          <span class="crawl-log-url" dir="ltr">${escapeHtml(r.url)}</span>
          <span class="crawl-log-meta">عمق ${r.depth} · ${titleBit}</span>
          ${reasonBit}
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
    /** @type {{source:string, chunks_added:number, message:string}[]} */
    const itemResults = [];

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
          const results = Array.isArray(evt.results) ? evt.results : [];
          results.forEach((r) => {
            if (Array.isArray(r.crawl_log)) r.crawl_log.forEach((row) => upsertCrawlLink(row));
          });
        } else if (evt.type === "item" && evt.result) {
          const r = evt.result;
          itemResults.push(r);
          if (Array.isArray(r.crawl_log)) r.crawl_log.forEach((row) => upsertCrawlLink(row));
        }
      });

      if (failed) throw new Error(finalMessage || "خطا در یادگیری");
      if (!finalMessage) finalMessage = "یادگیری کامل شد.";

      // Files that produced zero new chunks are duplicates already in the
      // knowledge base — surface that distinctly instead of letting it blend
      // into a generic "learning complete" success message.
      const duplicates = itemResults.filter((r) => (r.chunks_added || 0) === 0);
      const learned = itemResults.filter((r) => (r.chunks_added || 0) > 0);
      const allDuplicates = itemResults.length > 0 && duplicates.length === itemResults.length;

      ingestStatus.textContent = finalMessage;

      if (duplicates.length) {
        const names = duplicates.map((r) => `«${r.source}»`).join("، ");
        const warnMsg = allDuplicates
          ? `قبلاً آموخته شده بود، چیز جدیدی اضافه نشد: ${names}`
          : `این مورد تکراری بود و اضافه نشد: ${names}`;
        setProgress(100, finalMessage, allDuplicates ? "warning" : "done");
        showToast(warnMsg, false, true);
        if (learned.length) {
          // Also show the normal success toast for whatever genuinely got added.
          setTimeout(() => showToast(finalMessage), 300);
        }
      } else {
        setProgress(100, finalMessage, "done");
        showToast(finalMessage);
      }

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
          const url = typeof s.url === "string" ? s.url.trim() : "";
          const safeUrl = /^https?:\/\//i.test(url) ? escapeHtml(url) : "";
          const linkBit = safeUrl
            ? `<p class="text-xs mt-0.5"><a class="chat-cite" href="${safeUrl}" target="_blank" rel="noopener noreferrer" dir="ltr">${safeUrl}</a></p>`
            : "";
          const images = Array.isArray(s.images) ? s.images.filter(Boolean) : [];
          const thumbsBit = images.length
            ? `<div class="knowledge-thumbs">` +
              images
                .slice(0, 6)
                .map(
                  (src) =>
                    `<a href="${escapeHtml(src)}" target="_blank" rel="noopener noreferrer"><img src="${escapeHtml(src)}" loading="lazy" alt="تصویر از ${escapeHtml(s.name)}" /></a>`
                )
                .join("") +
              (images.length > 6 ? `<span class="knowledge-thumbs-more">+${images.length - 6}</span>` : "") +
              `</div>`
            : "";
          return `<li class="knowledge-item">
            <div class="min-w-0">
              <p class="text-sm font-medium text-ink-700 truncate">${escapeHtml(s.name)}</p>
              <p class="text-xs text-ink-400 mt-0.5">${escapeHtml(s.kind)} · ${s.chunk_count || 0} قطعه${date ? " · " + date : ""}${images.length ? ` · ${images.length} تصویر` : ""}</p>
              ${linkBit}
              ${thumbsBit}
            </div>
            <button type="button" class="source-delete-btn" data-delete-source="${escapeHtml(s.id)}">حذف</button>
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

  function autoGrow() {
    chatInput.style.height = "auto";
    chatInput.style.height = `${Math.min(chatInput.scrollHeight, 144)}px`;
  }
  chatInput.addEventListener("input", autoGrow);
  chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      chatForm.requestSubmit();
    }
  });

  chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const prompt = chatInput.value.trim();
    if (!prompt) return;
    appendBubble("user", "شما", prompt);
    chatInput.value = "";
    autoGrow();
    chatInput.disabled = true;
    const bubble = appendBubble("javad", "جواد", "", { typing: true });
    const p = bubble.querySelector("p");
    let citations = [];
    let sources = [];
    let gotToken = false;
    try {
      const res = await fetch("/api/ollama/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          session_id: currentSessionId,
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
          if (!gotToken) {
            gotToken = true;
            p.innerHTML = "";
          }
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
      refreshSessions();
    } catch (err) {
      p.textContent = err.message || "خطایی رخ داد.";
    } finally {
      chatInput.disabled = false;
      chatInput.focus();
    }
  });

  // ---------- Voice (mic -> STT -> chat -> TTS, over /ws/voip-stream) ----------

  let voiceSocket = null;
  let voiceAudioCtx = null;
  let voiceStream = null;
  let voiceProcessor = null;
  let voiceRecording = false;
  let voiceJavadBubble = null;

  function floatTo16BitPCM(float32) {
    const out = new Int16Array(float32.length);
    for (let i = 0; i < float32.length; i++) {
      const s = Math.max(-1, Math.min(1, float32[i]));
      out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    return out;
  }

  function downsampleTo16k(float32, inputRate) {
    if (inputRate === 16000) return float32;
    const ratio = inputRate / 16000;
    const outLength = Math.floor(float32.length / ratio);
    const out = new Float32Array(outLength);
    for (let i = 0; i < outLength; i++) {
      out[i] = float32[Math.floor(i * ratio)];
    }
    return out;
  }

  function setVoiceStatus(text) {
    if (!voiceStatus) return;
    if (!text) {
      voiceStatus.hidden = true;
      voiceStatus.textContent = "";
      return;
    }
    voiceStatus.hidden = false;
    voiceStatus.textContent = text;
  }

  async function startVoice() {
    try {
      voiceStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch {
      showToast("دسترسی به میکروفون ممکن نشد.", true);
      return;
    }
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const params = new URLSearchParams({
      session_id: currentSessionId,
      language: langSelect.value || "fa",
    });
    voiceSocket = new WebSocket(`${proto}//${location.host}/ws/voip-stream?${params}`);
    voiceSocket.binaryType = "arraybuffer";

    voiceSocket.onopen = () => {
      voiceAudioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const source = voiceAudioCtx.createMediaStreamSource(voiceStream);
      voiceProcessor = voiceAudioCtx.createScriptProcessor(4096, 1, 1);
      const inputRate = voiceAudioCtx.sampleRate;
      voiceProcessor.onaudioprocess = (e) => {
        if (!voiceSocket || voiceSocket.readyState !== WebSocket.OPEN) return;
        const input = e.inputBuffer.getChannelData(0);
        const down = downsampleTo16k(input, inputRate);
        const pcm16 = floatTo16BitPCM(down);
        voiceSocket.send(pcm16.buffer);
      };
      const silentGain = voiceAudioCtx.createGain();
      silentGain.gain.value = 0;
      source.connect(voiceProcessor);
      voiceProcessor.connect(silentGain);
      silentGain.connect(voiceAudioCtx.destination);
      voiceRecording = true;
      micBtn.classList.add("is-recording");
      setVoiceStatus("در حال گوش دادن… صحبت کنید");
    };

    voiceSocket.onmessage = (ev) => {
      if (typeof ev.data !== "string") {
        const blob = new Blob([ev.data], { type: "audio/wav" });
        const audio = new Audio(URL.createObjectURL(blob));
        audio.play().catch(() => {});
        return;
      }
      let evt;
      try {
        evt = JSON.parse(ev.data);
      } catch {
        return;
      }
      if (evt.type === "ready") {
        if (evt.voice && evt.voice.stt === false) {
          setVoiceStatus("موتور تشخیص گفتار در دسترس نیست.");
        }
      } else if (evt.type === "stt_start") {
        setVoiceStatus("در حال تشخیص گفتار…");
      } else if (evt.type === "transcript") {
        setVoiceStatus("در حال فکر کردن…");
        appendBubble("user", "شما", evt.text || "");
      } else if (evt.type === "answer") {
        voiceJavadBubble = appendBubble("javad", "جواد", evt.text || "");
        attachCitations(voiceJavadBubble, evt.citations || [], evt.sources || []);
        setVoiceStatus("در حال گوش دادن… صحبت کنید");
        refreshSessions();
      } else if (evt.type === "info") {
        setVoiceStatus(evt.message || "");
      } else if (evt.type === "error") {
        showToast(evt.message || "خطا در ارتباط صوتی", true);
        stopVoice();
      }
    };

    voiceSocket.onclose = () => {
      if (voiceRecording) stopVoice();
    };
    voiceSocket.onerror = () => {
      showToast("ارتباط صوتی قطع شد.", true);
      stopVoice();
    };
  }

  function stopVoice() {
    voiceRecording = false;
    micBtn.classList.remove("is-recording");
    setVoiceStatus("");
    try {
      voiceProcessor?.disconnect();
    } catch {
      /* ignore */
    }
    try {
      voiceAudioCtx?.close();
    } catch {
      /* ignore */
    }
    voiceStream?.getTracks().forEach((t) => t.stop());
    if (voiceSocket && voiceSocket.readyState === WebSocket.OPEN) {
      voiceSocket.close();
    }
    voiceSocket = null;
    voiceAudioCtx = null;
    voiceStream = null;
    voiceProcessor = null;
  }

  micBtn?.addEventListener("click", () => {
    if (voiceRecording) {
      stopVoice();
    } else {
      startVoice();
    }
  });

  function attachCitations(bubble, citations, sources) {
    const existing = bubble.querySelector(".chat-sources");
    if (existing) existing.remove();
    const items = Array.isArray(citations) && citations.length
      ? citations
      : (sources || []).map((s) =>
          typeof s === "string" ? { source: s, excerpt: "", url: "" } : s
        );
    if (!items.length) return;
    const wrap = document.createElement("div");
    wrap.className = "chat-sources";
    wrap.innerHTML =
      "<span class=\"chat-sources-label\">منابع:</span> " +
      items
        .map((c, i) => {
          const title = escapeHtml(c.source || "منبع");
          const excerpt = c.excerpt ? escapeHtml(c.excerpt) : "";
          const url = typeof c.url === "string" ? c.url.trim() : "";
          const safeUrl = /^https?:\/\//i.test(url) ? escapeHtml(url) : "";
          if (safeUrl) {
            return `<a class="chat-cite" href="${safeUrl}" target="_blank" rel="noopener noreferrer" title="${excerpt || safeUrl}">[${i + 1}] ${title}</a>`;
          }
          return `<button type="button" class="chat-cite" title="${excerpt}" data-cite="${i}">[${i + 1}] ${title}</button>`;
        })
        .join(" ");
    const allImages = [];
    items.forEach((c) => {
      (Array.isArray(c.images) ? c.images : []).forEach((src) => {
        if (src && !allImages.includes(src)) allImages.push(src);
      });
    });
    if (allImages.length) {
      const imgWrap = document.createElement("div");
      imgWrap.className = "chat-source-images";
      imgWrap.innerHTML = allImages
        .slice(0, 6)
        .map(
          (src) =>
            `<a href="${escapeHtml(src)}" target="_blank" rel="noopener noreferrer"><img src="${escapeHtml(src)}" loading="lazy" alt="تصویر از منبع" /></a>`
        )
        .join("");
      wrap.appendChild(imgWrap);
    }
    bubble.querySelector(".chat-bubble-body").appendChild(wrap);
    wrap.querySelectorAll("button.chat-cite").forEach((btn) => {
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
    const dots = [$("#ollama-dot")].filter(Boolean);
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
      dots.forEach((d) => {
        d.classList.toggle("is-ok", !!data.ollama);
        d.classList.toggle("is-down", !data.ollama);
      });
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
      dots.forEach((d) => d.classList.remove("is-ok", "is-down"));
    }
  }

  function appendBubble(role, label, text, { typing = false } = {}) {
    const el = document.createElement("div");
    el.className = `chat-bubble ${role}`;
    const avatar = role === "javad" ? "ج" : "شما";
    const body = typing
      ? '<span class="typing-dots"><span></span><span></span><span></span></span>'
      : "";
    el.innerHTML = `<span class="chat-avatar">${escapeHtml(avatar)}</span><div class="chat-bubble-body"><span class="chat-label">${escapeHtml(label)}</span><p>${body}</p></div>`;
    if (!typing) el.querySelector("p").textContent = text;
    chatLog.appendChild(el);
    chatLog.scrollTop = chatLog.scrollHeight;
    return el;
  }

  function showToast(message, isError = false, isWarning = false) {
    const toast = $("#toast");
    toast.hidden = false;
    toast.textContent = message;
    toast.classList.toggle("toast-error", !!isError);
    toast.classList.toggle("toast-warning", !!isWarning && !isError);
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
  refreshSessions();
})();
