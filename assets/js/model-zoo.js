(() => {
  'use strict';
  const root = document.querySelector('[data-model-zoo]');
  if (!root) return;
  const grid = document.getElementById('model-grid');
  const count = document.getElementById('result-count');
  const empty = document.getElementById('model-empty');
  const error = document.getElementById('model-error');
  const controls = { search: document.getElementById('model-search'), task: document.getElementById('task-filter'), scale: document.getElementById('scale-filter'), status: document.getElementById('status-filter'), sort: document.getElementById('model-sort') };
  const labels = { evaluated: 'Evaluated', evaluating: 'Evaluating', pending: 'Pending' };
  let models = [];
  const format = (value, digits = 2) => value == null ? '—' : Number(value).toFixed(digits).replace(/\.00$/, '');
  const metric = (label, value, digits) => `<div class="zoo-metric"><span class="zoo-metric-value">${format(value, digits)}</span><span class="zoo-metric-label">${label}</span></div>`;
  function card(model) {
    const weights = model.weights ? `<a class="btn btn-primary btn-sm" href="${model.weights}" target="_blank" rel="noopener noreferrer" aria-label="Download weights for ${model.name}"><i class="fas fa-download" aria-hidden="true"></i> Weights</a>` : '<span class="btn btn-disabled btn-sm" aria-disabled="true">Weights pending</span>';
    return `<article class="zoo-card"><div class="zoo-card-heading"><div><span class="zoo-family">${model.family}</span><h2>${model.name}</h2></div><span class="zoo-status status-${model.status}">${labels[model.status] || model.status}</span></div><p>${model.description}</p><div class="zoo-tags"><span>${model.task}</span><span>Scale ${model.scale}</span><span>${model.dataset}</span></div><div class="zoo-metrics">${metric('mAP50–95', model.map5095, 3)}${metric('mAP50', model.map50, 3)}${metric('Params (M)', model.params, 2)}${metric('GFLOPs', model.gflops, 1)}${metric('TRT FPS', model.fps, 2)}${metric('Precision', model.precision, 3)}</div><div class="zoo-card-actions">${weights}<a class="btn btn-secondary btn-sm" href="${model.config}" target="_blank" rel="noopener noreferrer" aria-label="Open config for ${model.name}"><i class="fas fa-code" aria-hidden="true"></i> Config</a></div></article>`;
  }
  function compare(a, b) {
    const sort = controls.sort.value;
    if (sort === 'name') return a.name.localeCompare(b.name);
    const key = sort === 'map' ? 'map5095' : sort === 'speed' ? 'fps' : 'params';
    const fallback = sort === 'params' ? Infinity : -Infinity;
    const av = a[key] == null ? fallback : a[key], bv = b[key] == null ? fallback : b[key];
    return sort === 'params' ? av - bv : bv - av;
  }
  function render() {
    const query = controls.search.value.trim().toLowerCase();
    const visible = models.filter(model => `${model.name} ${model.family} ${model.description} ${model.dataset}`.toLowerCase().includes(query) && (!controls.task.value || model.task === controls.task.value) && (!controls.scale.value || model.scale === controls.scale.value) && (!controls.status.value || model.status === controls.status.value)).sort(compare);
    grid.innerHTML = visible.map(card).join(''); count.textContent = `${visible.length} model${visible.length === 1 ? '' : 's'}`; empty.hidden = visible.length !== 0;
  }
  Object.values(controls).forEach(control => control.addEventListener(control === controls.search ? 'input' : 'change', render));
  document.getElementById('clear-filters').addEventListener('click', () => { controls.search.value = ''; controls.task.value = ''; controls.scale.value = ''; controls.status.value = ''; controls.sort.value = 'map'; render(); controls.search.focus(); });
  fetch('./model-zoo/models.json').then(response => { if (!response.ok) throw new Error(); return response.json(); }).then(data => { models = Array.isArray(data.models) ? data.models : []; root.setAttribute('aria-busy', 'false'); render(); }).catch(() => { root.setAttribute('aria-busy', 'false'); error.hidden = false; count.textContent = 'Unavailable'; });
})();
