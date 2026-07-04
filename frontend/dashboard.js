/* =========================================================
   dashboard.js — Result Dashboard Logic
   Now connected to the FastAPI backend.

   Flow:
     1. Load the uploaded image from sessionStorage
     2. POST it to /api/inspect
     3. Render real YOLOv8 detection results
     4. Show annotated image with bounding boxes
   ========================================================= */

document.addEventListener('DOMContentLoaded', async () => {

  // ---- Element references ----------------------------------------------
  const originalImage      = document.getElementById('originalImage');
  const detectionImage     = document.getElementById('detectionImage');
  const resultSubtitle     = document.getElementById('resultSubtitle');
  const classificationCard = document.getElementById('classificationCard');
  const classificationVal  = document.getElementById('classificationValue');
  const confidenceVal      = document.getElementById('confidenceValue');
  const defectTypeVal      = document.getElementById('defectTypeValue');
  const uploadAnotherBtn   = document.getElementById('uploadAnotherBtn');

  // ---- Load the uploaded image from sessionStorage ---------------------
  const imageData = sessionStorage.getItem('uploadedImageData');
  const imageName = sessionStorage.getItem('uploadedImageName') || 'upload';
  const imageBlob = sessionStorage.getItem('uploadedImageBlob');

  if (!imageData) {
    window.location.href = 'upload.html';
    return;
  }

  // Show original image immediately while API call is in progress
  originalImage.src = imageData;
  originalImage.alt = `Original: ${imageName}`;
  detectionImage.src = imageData;   // placeholder until annotated result arrives

  resultSubtitle.textContent = 'Running defect analysis — please wait ...';

  // ---- Call the real API -----------------------------------------------
  try {
    const result = await runInspection(imageData, imageName);
    renderResult(result);
  } catch (err) {
    console.error('Inspection API error:', err);
    renderError(err.message);
  }

  // ---- Navigation -------------------------------------------------------
  uploadAnotherBtn.addEventListener('click', () => {
    sessionStorage.removeItem('uploadedImageData');
    sessionStorage.removeItem('uploadedImageName');
    window.location.href = 'upload.html';
  });

  // =======================================================================
  // API Call
  // =======================================================================

  async function runInspection(imageDataUrl, filename) {
    // Convert the base64 data URL back to a Blob so we can send it
    // as multipart/form-data — exactly what FastAPI's UploadFile expects.
    const blob = dataUrlToBlob(imageDataUrl);
    const formData = new FormData();
    formData.append('file', blob, filename);

    const response = await fetch('/api/inspect', {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(
        errorData.detail || `Server error: ${response.status} ${response.statusText}`
      );
    }

    return await response.json();
  }

  function dataUrlToBlob(dataUrl) {
    const [header, base64Data] = dataUrl.split(',');
    const mimeType = header.match(/:(.*?);/)[1];
    const binaryStr = atob(base64Data);
    const bytes = new Uint8Array(binaryStr.length);
    for (let i = 0; i < binaryStr.length; i++) {
      bytes[i] = binaryStr.charCodeAt(i);
    }
    return new Blob([bytes], { type: mimeType });
  }

  // =======================================================================
  // Render Results
  // =======================================================================

  function renderResult(data) {
    const isGood = data.status === 'GOOD';

    // Show annotated image from the API (has bounding boxes drawn on it)
    if (data.annotated_image) {
      detectionImage.src = data.annotated_image;
      detectionImage.alt = `Detection result: ${data.defect_type}`;
    }

    // Classification card — color coded
    classificationCard.classList.remove('status-good', 'status-bad');
    classificationCard.classList.add(isGood ? 'status-good' : 'status-bad');
    classificationVal.textContent = data.status;
    classificationVal.classList.add('pop');

    // Confidence — already a percentage from the API
    if (data.status === 'GOOD' && data.confidence === 0) {
    confidenceVal.textContent = '100%';
    } else {
    confidenceVal.textContent = `${data.confidence}%`;
}
    confidenceVal.classList.add('pop');

    // Defect type
    defectTypeVal.textContent = data.defect_type || 'No Defect';
    defectTypeVal.classList.add('pop');

    // Subtitle
    resultSubtitle.textContent = isGood
      ? 'No significant surface defects detected on this component.'
      : `Defect detected: ${data.defect_type}. Review recommended.`;
  }

  function renderError(message) {
    resultSubtitle.textContent = `Analysis failed: ${message}`;
    resultSubtitle.style.color = '#FF4B4B';

    classificationVal.textContent = 'ERROR';
    confidenceVal.textContent = '—';
    defectTypeVal.textContent = '—';
  }

});