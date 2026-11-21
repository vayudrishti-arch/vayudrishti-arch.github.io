// Presentation only: load the reviewed stylesheet, then disable the previous inline stylesheet.
(() => {
  const previous = [...document.head.querySelectorAll('style')];
  const link = document.createElement('link');
  link.rel = 'stylesheet';
  link.href = new URL('airmate-app.css', document.currentScript.src).href;
  link.onload = () => previous.forEach(style => { if (style.sheet) style.sheet.disabled = true; });
  document.head.append(link);
})();
