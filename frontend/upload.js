/* =========================================================
   upload.js — Upload Page Logic
   Handles:
     - Click-to-browse file selection
     - Drag-and-drop file selection
     - File type validation (jpg, jpeg, png, bmp)
     - Image preview rendering
     - Passing the selected image to the dashboard page
   ========================================================= */

document.addEventListener('DOMContentLoaded', () => {

  // ---- Element references ----------------------------------------------
  const dropzone        = document.getElementById('dropzone');
  const fileInput        = document.getElementById('fileInput');
  const selectBtn         = document.getElementById('selectBtn');
  const dropzoneContent  = document.getElementById('dropzoneContent');
  const previewContent   = document.getElementById('previewContent');
  const previewImage      = document.getElementById('previewImage');
  const fileNameEl        = document.getElementById('fileName');
  const fileSizeEl        = document.getElementById('fileSize');
  const removeBtn          = document.getElementById('removeBtn');
  const errorMessage      = document.getElementById('errorMessage');
  const continueBtn       = document.getElementById('continueBtn');

  // Allowed image types per the brief: JPG, JPEG, PNG, BMP
  const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/bmp'];
  const ALLOWED_EXTENSIONS = /\.(jpe?g|png|bmp)$/i;

  let selectedFile = null;

  // ---- Helpers ------------------------------------------------------------

  function isValidImageFile(file) {
    if (!file) return false;
    const typeOk = ALLOWED_TYPES.includes(file.type);
    const extOk = ALLOWED_EXTENSIONS.test(file.name);
    // Some browsers report an empty MIME type for BMP — fall back to extension check.
    return typeOk || extOk;
  }

  function formatFileSize(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  }

  function showError(message) {
    errorMessage.querySelector('span').textContent = message;
    errorMessage.hidden = false;
  }

  function hideError() {
    errorMessage.hidden = true;
  }

  function handleFile(file) {
    hideError();

    if (!isValidImageFile(file)) {
      showError('Unsupported file type. Please upload a JPG, JPEG, PNG, or BMP image.');
      resetSelection();
      return;
    }

    selectedFile = file;

    const reader = new FileReader();
    reader.onload = (e) => {
      previewImage.src = e.target.result;

      // Store the image data so the dashboard page can display it
      // without re-uploading. This will later be replaced by an
      // actual upload call to the FastAPI backend.
      sessionStorage.setItem('uploadedImageData', e.target.result);
      sessionStorage.setItem('uploadedImageName', file.name);
    };
    reader.readAsDataURL(file);

    fileNameEl.textContent = file.name;
    fileSizeEl.textContent = formatFileSize(file.size);

    dropzoneContent.hidden = true;
    previewContent.hidden = false;

    continueBtn.disabled = false;
  }

  function resetSelection() {
    selectedFile = null;
    fileInput.value = '';
    previewImage.src = '';
    dropzoneContent.hidden = false;
    previewContent.hidden = true;
    continueBtn.disabled = true;
    sessionStorage.removeItem('uploadedImageData');
    sessionStorage.removeItem('uploadedImageName');
  }

  // ---- Click-to-browse ----------------------------------------------------

  selectBtn.addEventListener('click', () => fileInput.click());

  // Clicking anywhere on the dropzone (while empty) also opens the file picker
  dropzone.addEventListener('click', (e) => {
    if (!dropzoneContent.hidden && e.target !== selectBtn && !selectBtn.contains(e.target)) {
      fileInput.click();
    }
  });

  fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) handleFile(file);
  });

  // ---- Drag and drop --------------------------------------------------------

  ['dragenter', 'dragover'].forEach((eventName) => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add('drag-active');
    });
  });

  ['dragleave', 'drop'].forEach((eventName) => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove('drag-active');
    });
  });

  dropzone.addEventListener('drop', (e) => {
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  });

  // ---- Remove selected image ------------------------------------------------

  removeBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    resetSelection();
  });

  // ---- Continue to dashboard --------------------------------------------------

  continueBtn.addEventListener('click', () => {
    if (!selectedFile) return;
    window.location.href = 'dashboard.html';
  });

});
