/* =========================================================
   home.js — Home Page Logic
   Handles navigation from Home → Upload page.
   ========================================================= */

document.addEventListener('DOMContentLoaded', () => {
  const selectImageBtn = document.getElementById('selectImageBtn');

  selectImageBtn.addEventListener('click', () => {
    // Navigate to the upload page.
    // Using a short delay lets the click/hover animation complete
    // before the page transitions, which feels more deliberate.
    selectImageBtn.disabled = true;
    window.location.href = 'upload.html';
  });
});
