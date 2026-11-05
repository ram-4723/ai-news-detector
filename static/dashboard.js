(function () {
  const MAX_IMAGE_BYTES = 5 * 1024 * 1024;
  const savedStorageKey = 'fakeNewsDetector.savedResults.v1';
  let uploadedImage = null;
  let activeMode = 'text';
  let latestResult = null;
  let latestQuery = '';
  let recentAnalyses = [];
  let toastTimer;

  const newsInput = document.getElementById('newsText');
  const imageNewsInput = document.getElementById('newsTextImage');
  const urlInput = document.getElementById('urlText');
  const urlKeywords = document.getElementById('urlKeywords');
  const fileInput = document.getElementById('fileInput');
  const errorMessage = document.getElementById('errorMessage');
  const analyzeButton = document.getElementById('analyzeBtn');
  const resultsSection = document.getElementById('result-card');
  const toast = document.getElementById('toast');

  function showToast(message) {
    toast.textContent = message;
    toast.classList.add('show');
    window.clearTimeout(toastTimer);
    toastTimer = window.setTimeout(function () { toast.classList.remove('show'); }, 2400);
  }

  document.querySelectorAll('.mode-tab').forEach(function (tab) {
    tab.addEventListener('click', function () {
      activeMode = tab.dataset.mode;
      document.querySelectorAll('.mode-tab').forEach(function (item) {
        const selected = item === tab;
        item.classList.toggle('active', selected);
        item.setAttribute('aria-selected', String(selected));
      });
      document.querySelectorAll('.mode-panel').forEach(function (panel) {
        panel.hidden = panel.id !== activeMode + 'Panel';
      });
      errorMessage.style.display = 'none';
    });
  });

  function updateCharCount(value) {
    document.getElementById('charCount').textContent = value.length.toLocaleString() + ' / 20,000';
  }
  newsInput.addEventListener('input', function () { updateCharCount(newsInput.value); });
  imageNewsInput.addEventListener('input', function () { updateCharCount(imageNewsInput.value); });
  document.getElementById('chooseImage').addEventListener('click', function () { fileInput.click(); });
  document.getElementById('removeImage').addEventListener('click', removeImage);
  analyzeButton.addEventListener('click', analyze);
  document.getElementById('notificationButton').addEventListener('click', function () { showToast('You are all caught up.'); });
  document.getElementById('clearForm').addEventListener('click', function () {
    newsInput.value = ''; imageNewsInput.value = ''; urlInput.value = ''; urlKeywords.value = '';
    removeImage(); updateCharCount(''); errorMessage.style.display = 'none';
  });

  document.getElementById('themeToggle').addEventListener('click', function () {
    document.body.classList.toggle('theme-dark');
    const dark = document.body.classList.contains('theme-dark');
    document.getElementById('themeToggle').setAttribute('aria-pressed', String(dark));
  });

  fileInput.addEventListener('change', function (event) {
    const file = event.target.files && event.target.files[0];
    if (!file) return;
    if (file.size > MAX_IMAGE_BYTES) {
      showError('Choose an image smaller than 5 MB.');
      fileInput.value = '';
      return;
    }
    if (['image/jpeg', 'image/png', 'image/webp'].indexOf(file.type) === -1) {
      showError('Choose a JPG, PNG, or WebP image.');
      fileInput.value = '';
      return;
    }
    const reader = new FileReader();
    reader.onload = function () {
      uploadedImage = reader.result;
      document.getElementById('previewImg').src = uploadedImage;
      document.getElementById('previewName').textContent = file.name;
      document.getElementById('preview').style.display = 'flex';
      errorMessage.style.display = 'none';
    };
    reader.onerror = function () { showError('The selected image could not be read.'); };
    reader.readAsDataURL(file);
  });

  function removeImage() {
    uploadedImage = null;
    fileInput.value = '';
    document.getElementById('previewImg').removeAttribute('src');
    document.getElementById('previewName').textContent = '';
    document.getElementById('preview').style.display = 'none';
  }

  function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
  }

  function getQueryForCurrentMode() {
    if (activeMode === 'url') {
      const rawUrl = urlInput.value.trim();
      if (!rawUrl) throw new Error('Paste a news article URL first.');
      let parsed;
      try { parsed = new URL(rawUrl); }
      catch (_) { throw new Error('Enter a complete URL, such as https://example.com/news/story.'); }
      if (['http:', 'https:'].indexOf(parsed.protocol) === -1) throw new Error('Use an http or https URL.');
      let path = parsed.pathname;
      try { path = decodeURIComponent(path); } catch (_) {}
      const hostTerms = parsed.hostname.replace(/^www\./i, '').replace(/\.[a-z]{2,}$/i, '').replace(/[.-]/g, ' ');
      const pathTerms = path.split('/').filter(Boolean).join(' ').replace(/[-_]/g, ' ').replace(/\.[a-z0-9]{2,5}$/i, ' ');
      return (hostTerms + ' ' + pathTerms + ' ' + urlKeywords.value.trim()).replace(/\s+/g, ' ').trim();
    }
    if (activeMode === 'image') return imageNewsInput.value.trim();
    return newsInput.value.trim();
  }

  function setLoading(isLoading) {
    analyzeButton.disabled = isLoading;
    resultsSection.setAttribute('aria-busy', String(isLoading));
    const status = document.getElementById('resultStatus');
    status.textContent = isLoading ? 'Analyzing…' : (latestResult ? 'Complete' : 'Ready');
    status.classList.toggle('complete', !isLoading && Boolean(latestResult));
    document.getElementById('spinner').style.display = isLoading ? 'block' : 'none';
    document.getElementById('searchIcon').style.display = isLoading ? 'none' : 'block';
    document.getElementById('buttonLabel').textContent = isLoading ? 'Analyzing News…' : 'Analyze News';
  }

  async function analyze() {
    errorMessage.style.display = 'none';
    let news;
    try { news = getQueryForCurrentMode(); }
    catch (error) { showError(error.message); return; }
    if (!news && !uploadedImage) {
      showError(activeMode === 'image' ? 'Add a screenshot or enter a claim to analyze.' : 'Enter a claim or headline first.');
      if (activeMode === 'url') urlInput.focus();
      else if (activeMode === 'image') imageNewsInput.focus();
      else newsInput.focus();
      return;
    }
    latestQuery = news || 'Text extracted from uploaded screenshot';
    setLoading(true);
    try {
      const response = await fetch('/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ news: news, image: uploadedImage })
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'The news search could not be completed.');
      renderResults(data);
      recentAnalyses.unshift({ query: latestQuery, mode: activeMode, url: urlInput.value.trim(), keywords: urlKeywords.value.trim(), createdAt: Date.now(), total: Number(data.total_results) || 0 });
      recentAnalyses = recentAnalyses.slice(0, 5);
      renderRecentAnalyses();
      resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    } catch (error) {
      showError(error.message || 'Unable to connect to the analysis service.');
    } finally {
      setLoading(false);
    }
  }

  function makeArticle(article) {
    const card = document.createElement('article');
    card.className = 'article';
    const source = document.createElement('p');
    source.className = 'article-source';
    source.textContent = article.source || 'News source';
    const title = document.createElement('a');
    title.className = 'article-title';
    title.textContent = article.title || 'Open related article';
    try {
      const url = new URL(article.url);
      if (url.protocol === 'https:' || url.protocol === 'http:') {
        title.href = url.href;
        title.target = '_blank';
        title.rel = 'noopener noreferrer';
      }
    } catch (_) { title.removeAttribute('href'); }
    const description = document.createElement('p');
    description.className = 'article-description';
    description.textContent = article.description || 'NewsAPI did not provide a description for this article.';
    card.append(source, title, description);
    return card;
  }

  function renderResults(data) {
    latestResult = data;
    const articles = Array.isArray(data.articles) ? data.articles : [];
    const total = Number(data.total_results) || 0;
    document.getElementById('verdictText').textContent = 'Inconclusive';
    document.getElementById('resultStatus').textContent = 'Complete';
    document.getElementById('resultStatus').classList.add('complete');
    document.getElementById('articleCount').textContent = total.toLocaleString() + ' matching article' + (total === 1 ? '' : 's');
    document.getElementById('evidenceSummary').textContent = data.evidence_summary || 'Review the article links below; related coverage is not proof.';
    document.getElementById('showingCount').textContent = articles.length ? 'Showing ' + articles.length : '';

    const estimate = document.getElementById('classifierEstimate');
    const confidenceBox = document.getElementById('confidenceBox');
    if (data.classifier_estimate) {
      estimate.replaceChildren();
      const label = document.createElement('strong');
      label.textContent = 'Separate ML estimate (not a fact-check): ';
      estimate.append(label, document.createTextNode(data.classifier_estimate + ' · ' + data.classifier_probability + '% model probability'));
      estimate.style.display = 'block';
      document.getElementById('confidenceLabel').textContent = 'Separate model estimate · ' + data.classifier_estimate;
      const probability = Math.max(0, Math.min(100, Number(data.classifier_probability) || 0));
      document.getElementById('confidenceValue').textContent = probability.toFixed(1) + '%';
      const confidenceFill = document.getElementById('confidenceFill');
      confidenceFill.classList.toggle('is-fake', data.classifier_estimate === 'Likely Fake');
      confidenceFill.classList.toggle('is-real', data.classifier_estimate === 'Likely Real');
      confidenceFill.style.width = probability + '%';
      confidenceBox.style.display = 'block';
    } else {
      estimate.style.display = 'none';
      confidenceBox.style.display = 'none';
    }

    const ocrDetails = document.getElementById('ocrDetails');
    if (data.ocr_text && data.ocr_text.trim()) {
      document.getElementById('ocrText').textContent = data.ocr_text;
      ocrDetails.style.display = 'block';
    } else {
      ocrDetails.style.display = 'none';
    }

    const list = document.getElementById('articleList');
    list.replaceChildren();
    if (articles.length) articles.forEach(function (article) { list.appendChild(makeArticle(article)); });
    else {
      const empty = document.createElement('p');
      empty.className = 'empty-results';
      empty.textContent = 'NewsAPI found no matching articles. That does not mean the claim is false—try a shorter or more specific wording.';
      list.appendChild(empty);
    }
    document.getElementById('resultEmpty').style.display = 'none';
    document.getElementById('resultContent').style.display = 'block';
  }

  function createRecentRow(item, index) {
    const row = document.createElement('div');
    row.className = 'recent-item';
    const badge = document.createElement('span');
    badge.className = 'recent-badge';
    badge.innerHTML = '<svg class="icon" viewBox="0 0 24 24"><circle cx="10.5" cy="10.5" r="7.2"/><path d="m16 16 4.5 4.5"/></svg>';
    const copy = document.createElement('span');
    copy.className = 'recent-copy';
    const title = document.createElement('strong');
    title.textContent = item.query;
    title.title = item.query;
    const meta = document.createElement('span');
    const state = document.createElement('b');
    state.className = 'recent-state';
    state.textContent = 'Inconclusive';
    meta.append(state, document.createTextNode(' · ' + Number(item.total || 0).toLocaleString() + ' related · ' + formatAge(item.createdAt)));
    copy.append(title, meta);
    const open = document.createElement('button');
    open.type = 'button';
    open.className = 'recent-open';
    open.setAttribute('aria-label', 'Reopen analysis ' + (index + 1));
    open.innerHTML = '<svg class="icon" viewBox="0 0 24 24"><path d="m9 18 6-6-6-6"/></svg>';
    open.addEventListener('click', function () {
      document.querySelector('.mode-tab[data-mode="' + item.mode + '"]').click();
      if (item.mode === 'url') { urlInput.value = item.url || ''; urlKeywords.value = item.keywords || ''; }
      else if (item.mode === 'image') imageNewsInput.value = item.query;
      else newsInput.value = item.query;
      document.getElementById('analyze').scrollIntoView({ behavior: 'smooth' });
    });
    row.append(badge, copy, open);
    return row;
  }

  function renderRecentAnalyses() {
    const list = document.getElementById('recentAnalysisList');
    list.replaceChildren();
    if (!recentAnalyses.length) {
      const empty = document.createElement('p');
      empty.className = 'recent-empty';
      empty.textContent = 'Completed analyses appear here during this session.';
      list.appendChild(empty);
      return;
    }
    recentAnalyses.forEach(function (item, index) { list.appendChild(createRecentRow(item, index)); });
  }

  function formatAge(timestamp) {
    const minutes = Math.max(0, Math.floor((Date.now() - timestamp) / 60000));
    if (minutes < 1) return 'just now';
    if (minutes < 60) return minutes + ' min ago';
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return hours + ' hr ago';
    return Math.floor(hours / 24) + ' d ago';
  }

  function readSavedResults() {
    try {
      const saved = JSON.parse(localStorage.getItem(savedStorageKey) || '[]');
      return Array.isArray(saved) ? saved.slice(0, 10) : [];
    } catch (_) { return []; }
  }

  function renderSavedResults() {
    const list = document.getElementById('savedList');
    list.replaceChildren();
    const saved = readSavedResults();
    if (!saved.length) {
      const empty = document.createElement('p');
      empty.className = 'recent-empty';
      empty.textContent = 'Save a result to keep it in this browser.';
      list.appendChild(empty);
      return;
    }
    saved.forEach(function (item, index) {
      const row = document.createElement('div');
      row.className = 'recent-item';
      const badge = document.createElement('span');
      badge.className = 'recent-badge';
      badge.textContent = '✓';
      const copy = document.createElement('span');
      copy.className = 'recent-copy';
      const title = document.createElement('strong');
      title.textContent = item.query;
      const meta = document.createElement('span');
      meta.textContent = 'Inconclusive · ' + new Date(item.createdAt).toLocaleString();
      copy.append(title, meta);
      const remove = document.createElement('button');
      remove.type = 'button';
      remove.className = 'recent-open';
      remove.setAttribute('aria-label', 'Remove saved result');
      remove.textContent = '×';
      remove.addEventListener('click', function () {
        const next = readSavedResults();
        next.splice(index, 1);
        localStorage.setItem(savedStorageKey, JSON.stringify(next));
        renderSavedResults();
      });
      row.append(badge, copy, remove);
      list.appendChild(row);
    });
  }

  document.getElementById('saveResult').addEventListener('click', function () {
    if (!latestResult) return;
    try {
      const saved = readSavedResults();
      saved.unshift({ query: latestQuery, createdAt: Date.now(), evidence: latestResult.evidence_summary, articles: latestResult.articles });
      localStorage.setItem(savedStorageKey, JSON.stringify(saved.slice(0, 10)));
      renderSavedResults();
      showToast('Analysis saved in this browser.');
    } catch (_) { showToast('Could not save this result in browser storage.'); }
  });

  document.getElementById('downloadReport').addEventListener('click', function () {
    if (!latestResult) return;
    const report = { title: 'News evidence review', query: latestQuery, generatedAt: new Date().toISOString(), ...latestResult };
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'news-evidence-review.json';
    link.click();
    URL.revokeObjectURL(link.href);
  });

  document.getElementById('shareResult').addEventListener('click', async function () {
    if (!latestResult) return;
    const summary = 'News evidence review: ' + latestQuery + '\nStatus: Inconclusive\n' + (latestResult.evidence_summary || '');
    try {
      await navigator.clipboard.writeText(summary);
      showToast('Review summary copied to clipboard.');
    } catch (_) { showToast('Clipboard access is unavailable in this browser.'); }
  });

  document.getElementById('analyzeAnother').addEventListener('click', function () {
    latestResult = null;
    document.getElementById('resultContent').style.display = 'none';
    document.getElementById('resultEmpty').style.display = 'flex';
    document.getElementById('resultStatus').textContent = 'Ready';
    document.getElementById('resultStatus').classList.remove('complete');
    newsInput.value = '';
    imageNewsInput.value = '';
    urlInput.value = '';
    urlKeywords.value = '';
    removeImage();
    updateCharCount('');
    document.querySelector('.mode-tab[data-mode="text"]').click();
    newsInput.focus();
    document.getElementById('analyze').scrollIntoView({ behavior: 'smooth' });
  });

  document.querySelectorAll('.top-nav a,.side-link').forEach(function (link) {
    link.addEventListener('click', function () {
      if (link.closest('.top-nav')) {
        document.querySelectorAll('.top-nav a').forEach(function (item) { item.classList.toggle('active', item === link); });
      }
      if (link.closest('.side-nav')) {
        document.querySelectorAll('.side-link').forEach(function (item) { item.classList.toggle('active', item === link); });
      }
    });
  });

  renderSavedResults();
})();
