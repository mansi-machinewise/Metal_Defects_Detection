/* =========================================================
   dashboard.js — Result Dashboard Logic
   ========================================================= */

document.addEventListener('DOMContentLoaded', async () => {

  const originalImage   = document.getElementById('originalImage');
  const detectionImage  = document.getElementById('detectionImage');
  const resultSubtitle  = document.getElementById('resultSubtitle');
  const classCard       = document.getElementById('classificationCard');
  const classVal        = document.getElementById('classificationValue');
  const confVal         = document.getElementById('confidenceValue');
  const defectTypeVal   = document.getElementById('defectTypeValue');
  const uploadAnotherBtn= document.getElementById('uploadAnotherBtn');
  const detectionCard   = document.getElementById('detectionCard');

  const lightboxOverlay = document.getElementById('lightboxOverlay');
  const lightboxClose   = document.getElementById('lightboxClose');
  const lightboxImage   = document.getElementById('lightboxImage');
  const bboxSvg         = document.getElementById('bboxSvg');
  const lbDefectList    = document.getElementById('lightboxDefectList');

  let allDefects = [];

  const PALETTE = [
    '#FF4B4B', '#00C8FF', '#FFD166', '#06D6A0',
    '#FF8C42', '#C77DFF', '#F72585', '#4CC9F0'
  ];
  const colourMap = {};
  let colourIdx = 0;
  function getColour(name) {
    if (!colourMap[name]) colourMap[name] = PALETTE[colourIdx++ % PALETTE.length];
    return colourMap[name];
  }

  // =========================================================
  // LIGHTBOX
  // =========================================================
  function openLightbox() {
    // Use the ORIGINAL clean image so YOLO's baked-in labels are hidden.
    // Our SVG boxes are drawn on top instead — much cleaner.
    const src = originalImage.src;
    if (!src || src === window.location.href) return;
    lightboxImage.src = src;

    const doRender = () => { drawBoundingBoxes(); renderSidebar(); };
    lightboxImage.onload = doRender;
    if (lightboxImage.complete && lightboxImage.naturalWidth > 0) doRender();

    lightboxOverlay.classList.add('lb-open');
    document.body.style.overflow = 'hidden';
  }

  function closeLightbox() {
    lightboxOverlay.classList.remove('lb-open');
    document.body.style.overflow = '';
    bboxSvg.innerHTML = '';
    lbDefectList.innerHTML = '';
  }

  detectionCard.addEventListener('click', openLightbox);
  lightboxClose.addEventListener('click', (e) => { e.stopPropagation(); closeLightbox(); });
  lightboxOverlay.addEventListener('click', (e) => { if (e.target === lightboxOverlay) closeLightbox(); });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeLightbox(); });
  window.addEventListener('resize', () => {
    if (lightboxOverlay.classList.contains('lb-open')) drawBoundingBoxes();
  });

  // =========================================================
  // SVG bounding boxes
  // =========================================================
  function drawBoundingBoxes() {
    bboxSvg.innerHTML = '';
    if (!allDefects.length) return;

    const img  = lightboxImage;
    const natW = img.naturalWidth;
    const natH = img.naturalHeight;
    if (!natW || !natH) return;

    // With object-fit:contain the image is letterboxed inside its element.
    // We need the actual rendered pixel rect of the image content, not the element rect.
    const wrapRect  = document.getElementById('lightboxImageWrap').getBoundingClientRect();
    const elemW     = wrapRect.width;
    const elemH     = wrapRect.height;

    const scale     = Math.min(elemW / natW, elemH / natH);
    const renderedW = natW * scale;
    const renderedH = natH * scale;
    const offX      = (elemW - renderedW) / 2;
    const offY      = (elemH - renderedH) / 2;

    const scaleX = renderedW / natW;
    const scaleY = renderedH / natH;

    bboxSvg.style.left   = offX + 'px';
    bboxSvg.style.top    = offY + 'px';
    bboxSvg.style.width  = renderedW + 'px';
    bboxSvg.style.height = renderedH + 'px';
    bboxSvg.setAttribute('viewBox', `0 0 ${renderedW} ${renderedH}`);

    allDefects.forEach((d, i) => {
      const bbox = d.bbox;
      if (!bbox || bbox.length < 4) return;

      const [x1, y1, x2, y2] = bbox;
      const rx = x1 * scaleX;
      const ry = y1 * scaleY;
      const rw = (x2 - x1) * scaleX;
      const rh = (y2 - y1) * scaleY;
      const colour = getColour(d.class_name);
      const pct    = Math.round(normConf(d.confidence) * 100);

      const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      g.setAttribute('data-idx', i);
      g.style.cursor = 'crosshair';
      g.style.transition = 'opacity 0.18s ease';

      // Box — always visible, outline only, no fill by default
      const box = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      box.setAttribute('x', rx); box.setAttribute('y', ry);
      box.setAttribute('width', rw); box.setAttribute('height', rh);
      box.setAttribute('fill', 'none');           // no fill by default
      box.setAttribute('stroke', colour);
      box.setAttribute('stroke-width', '2');
      box.setAttribute('rx', '4');
      box.classList.add('bbox-rect');

      // Label background — always visible
      const label  = `${d.class_name.replace(/_/g,' ')} ${pct}%`;
      const labelW = label.length * 7.5 + 12;
      const labelH = 22;
      const labelY = ry > labelH + 6 ? ry - labelH - 4 : ry + rh + 4;

      const bgRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      bgRect.setAttribute('x', rx); bgRect.setAttribute('y', labelY);
      bgRect.setAttribute('width', labelW); bgRect.setAttribute('height', labelH);
      bgRect.setAttribute('fill', colour); bgRect.setAttribute('rx', '4');

      const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      text.setAttribute('x', rx + 6); text.setAttribute('y', labelY + 15);
      text.setAttribute('fill', '#fff');
      text.setAttribute('font-size', '12');
      text.setAttribute('font-family', 'Poppins, sans-serif');
      text.setAttribute('font-weight', '700');
      text.setAttribute('pointer-events', 'none');
      text.textContent = label;

      g.appendChild(box); g.appendChild(bgRect); g.appendChild(text);

      g.addEventListener('mouseenter', () => { highlightSidebar(i, true, i); });
      g.addEventListener('mouseleave', () => { highlightSidebar(i, false, i); });

      bboxSvg.appendChild(g);
    });
  }

  function highlightBoxes(indices, on) {
    bboxSvg.querySelectorAll('g[data-idx]').forEach(g => {
      const idx    = parseInt(g.dataset.idx);
      const rect   = g.querySelector('.bbox-rect');
      if (!rect) return;
      const colour = allDefects[idx] ? getColour(allDefects[idx].class_name) : '#fff';
      if (on && indices.includes(idx)) {
        // Light fill — 20% opacity so defect stays clearly visible underneath
        rect.setAttribute('fill', colour + '0D');
        rect.setAttribute('stroke-width', '2.5');
        g.style.filter = `drop-shadow(0 0 4px ${colour}88)`;
      } else {
        // Back to outline only
        rect.setAttribute('fill', 'none');
        rect.setAttribute('stroke-width', '2');
        g.style.filter = 'none';
      }
    });
  }

  // highlightSidebar: when hovering a sidebar entry → highlight all boxes of that class
  // when hovering an SVG box directly → highlight only that specific box (singleIdx)
  function highlightSidebar(idx, on, singleIdx = null) {
    lbDefectList.querySelectorAll('.de-entry').forEach(el => {
      const indices = el.dataset.indices.split(',').map(Number);
      if (indices.includes(idx)) {
        el.classList.toggle('active', on);
        const toHighlight = singleIdx !== null ? [singleIdx] : indices;
        highlightBoxes(toHighlight, on);
      }
    });
  }

  // =========================================================
  // Sidebar
  // =========================================================
  function renderSidebar() {
    lbDefectList.innerHTML = '';
    if (!allDefects.length) {
      lbDefectList.innerHTML = '<p style="color:#8FA6BA;font-size:.8rem">No defects detected.</p>';
      return;
    }

    const groups = {};
    allDefects.forEach((d, i) => {
      const k = d.class_name;
      if (!groups[k]) groups[k] = { name: k, indices: [], maxConf: 0 };
      groups[k].indices.push(i);
      const c = normConf(d.confidence);
      if (c > groups[k].maxConf) groups[k].maxConf = c;
    });

    Object.values(groups).forEach(grp => {
      const colour = getColour(grp.name);
      const pct    = Math.round(grp.maxConf * 100);

      const entry  = document.createElement('div');
      entry.className = 'de-entry';
      entry.dataset.indices = grp.indices.join(',');
      entry.innerHTML = `
        <div class="de-dot" style="background:${colour}"></div>
        <div class="de-body">
          <div class="de-name">${grp.name.replace(/_/g,' ')}</div>
          <div class="de-conf">${pct}% confidence · ${grp.indices.length} region${grp.indices.length > 1 ? 's' : ''}</div>
          <div class="de-bar"><div class="de-bar-fill" style="width:${pct}%;background:${colour}"></div></div>
        </div>`;

      entry.addEventListener('mouseenter', () => {
        entry.classList.add('active');
        highlightBoxes(grp.indices, true);
      });
      entry.addEventListener('mouseleave', () => {
        entry.classList.remove('active');
        highlightBoxes(grp.indices, false);
      });

      lbDefectList.appendChild(entry);
    });
  }

  // =========================================================
  // Normalise confidence: handles 0-1 floats AND 0-100 values
  // =========================================================
  function normConf(val) {
    const n = parseFloat(val) || 0;
    return n > 1 ? n / 100 : n;
  }

  // =========================================================
  // Upload another
  // =========================================================
  uploadAnotherBtn.addEventListener('click', () => {
    sessionStorage.removeItem('uploadedImageData');
    sessionStorage.removeItem('uploadedImageName');
    window.location.href = 'upload.html';
  });

  // =========================================================
  // Load image from sessionStorage + call API
  // =========================================================
  const imageData = sessionStorage.getItem('uploadedImageData');
  const imageName = sessionStorage.getItem('uploadedImageName') || 'upload';

  if (!imageData) { window.location.href = 'upload.html'; return; }

  originalImage.src  = imageData;
  detectionImage.src = imageData;
  resultSubtitle.textContent = 'Running defect analysis — please wait …';

  try {
    const result = await runInspection(imageData, imageName);
    renderResult(result);
  } catch (err) {
    console.error('Inspection API error:', err);
    renderError(err.message);
  }

  // =========================================================
  // API
  // =========================================================
  async function runInspection(dataUrl, filename) {
    const blob = dataUrlToBlob(dataUrl);
    const form = new FormData();
    form.append('file', blob, filename);
    const res = await fetch('/api/inspect', { method: 'POST', body: form });
    if (!res.ok) {
      const e = await res.json().catch(() => ({}));
      throw new Error(e.detail || `Server error ${res.status}`);
    }
    return res.json();
  }

  function dataUrlToBlob(dataUrl) {
    const [header, b64] = dataUrl.split(',');
    const mime = header.match(/:(.*?);/)[1];
    const bin  = atob(b64);
    const arr  = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
    return new Blob([arr], { type: mime });
  }

  // =========================================================
  // Render result
  // =========================================================
  function renderResult(data) {
    const isGood = data.status === 'GOOD';

    // Annotated image
    if (data.annotated_image) {
      detectionImage.src = data.annotated_image.startsWith('data:')
        ? data.annotated_image
        : `data:image/jpeg;base64,${data.annotated_image}`;
    }

    // Store defects — normalise bbox key names
    allDefects = (data.defects || []).map(d => ({
      class_name: d.class_name || d.label || 'unknown',
      confidence: d.confidence,
      bbox: d.bbox || d.xyxy || d.box
            || (d.xmin != null ? [d.xmin, d.ymin, d.xmax, d.ymax] : null)
            || null,
    }));

    // Classification
    classCard.classList.remove('status-good', 'status-bad');
    classCard.classList.add(isGood ? 'status-good' : 'status-bad');
    classVal.textContent = data.status;
    classVal.classList.add('pop');

    // Confidence — guard against pre-multiplied value
    const rawConf = parseFloat(data.confidence) || 0;
    const dispConf = rawConf > 1 ? rawConf.toFixed(2) : (rawConf * 100).toFixed(2);
    confVal.textContent = (isGood && parseFloat(data.confidence) === 0) ? '100%' : `${dispConf}%`;
    confVal.classList.add('pop');

    // Defect tags
    if (allDefects.length > 0) {
      const names = [...new Set(allDefects.map(d => d.class_name))];
      defectTypeVal.innerHTML = names.map(n => `<span class="defect-tag">${n}</span>`).join('');
    } else {
      defectTypeVal.textContent = data.defect_type || 'No Defect';
    }
    defectTypeVal.classList.add('pop');

    resultSubtitle.textContent = isGood
      ? 'No defects found.'
      : `Defect detected.`;
  }

  function renderError(msg) {
    resultSubtitle.textContent = `Analysis failed: ${msg}`;
    resultSubtitle.style.color = '#FF4B4B';
    classVal.textContent = 'ERROR';
    confVal.textContent  = '—';
    defectTypeVal.textContent = '—';
  }
});