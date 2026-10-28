(function () {
  'use strict';
  const MI = window.MI;
  MI.applyStoredTheme();
  MI.renderShell();
  MI.loadBatch(MI.D.SAMPLE_CSV, 'morning_batch.csv');
  const seed = new Date(Date.now() - 12 * 60 * 1000);
  MI.state.batch.loadedAt = new Date(seed.getTime() - 45 * 1000);
  MI.prepareCases();
  MI.startRun({ instant: true, seedTime: seed });
  const start = (location.hash || '').slice(1);
  MI.navigate(MI.screens[start] ? start : 'overview', { noFocus: true });
  MI.checkDownloads();
  if (window.matchMedia) {
    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    const onChange = () => MI.renderTopbar();
    if (mq.addEventListener) mq.addEventListener('change', onChange);
  }
})();
