const state = {
  runs: [],
  roots: [],
  selected: null,
  currentView: "overview",
  runStatus: "all",
  runQuery: "",
  familyFilter: "all",
  layerStatus: "all",
  layerQuery: "",
  selectedLayer: null,
  benchmarkFamily: null,
};

const $ = (id) => document.getElementById(id);
const escapeHtml = (value) => String(value ?? "—").replace(/[&<>'"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[char]);
const finite = (value) => typeof value === "number" && Number.isFinite(value);
const pct = (value, digits = 3) => finite(value) ? `${value > 0 ? "+" : ""}${value.toFixed(digits)}%` : "—";
const decimal = (value, digits = 3) => finite(value) ? value.toFixed(digits) : "—";
const bytes = (value) => {
  if (!finite(value)) return "—";
  if (value < 1024) return `${value} B`;
  if (value < 1024 ** 2) return `${(value / 1024).toFixed(1)} KiB`;
  return `${(value / 1024 ** 2).toFixed(1)} MiB`;
};
const average = (values) => {
  const valid = values.filter(finite);
  return valid.length ? valid.reduce((total, value) => total + value, 0) / valid.length : null;
};
const statusLabel = { healthy: "正常", warning: "需要关注", empty: "无路由数据" };
const auxLabel = {
  active_training: "训练中生效",
  configured_inactive_eval: "仅训练时生效",
  configured_zero_observed: "已配置，当前为零",
  not_configured: "未配置",
  unavailable: "不可用",
  unknown: "未知",
};
const axisLabel = { spatial: "空间", token: "Token", expert: "专家" };
const dispatchLabel = { dense: "全量", topk: "Top-k", sparse: "稀疏", unknown: "未记录" };
const familyColors = ["#0052d9", "#00a870", "#7a5af8", "#ed7b2f", "#d54941"];

async function api(path, options) {
  const response = await fetch(path, { cache: "no-store", ...options });
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json();
}

function showToast(message) {
  const toast = $("toast");
  toast.textContent = message;
  toast.hidden = false;
  window.clearTimeout(showToast.timeout);
  showToast.timeout = window.setTimeout(() => { toast.hidden = true; }, 2200);
}

function setConnection(online, text) {
  document.querySelector(".connection").classList.toggle("online", online);
  $("connection-text").textContent = text;
}

function runSearchText(run) {
  const metadata = run.metadata || {};
  return [run.name, run.relative_path, run.root, metadata.device, metadata.resolved_device, metadata.dataset].join(" ").toLowerCase();
}

function filteredRuns() {
  return state.runs.filter((run) => {
    const statusMatches = state.runStatus === "all" || run.status === state.runStatus;
    return statusMatches && runSearchText(run).includes(state.runQuery.toLowerCase());
  });
}

function renderRuns() {
  const list = $("run-list");
  const runs = filteredRuns();
  $("run-count").textContent = state.runs.length;
  list.replaceChildren();
  if (!runs.length) {
    list.innerHTML = '<div class="run-list-empty">没有符合当前筛选条件的运行记录。<br>点击右上角刷新可重新扫描。</div>';
    return;
  }
  runs.forEach((run) => {
    const button = document.createElement("button");
    const device = run.metadata?.device || run.metadata?.resolved_device || "unknown";
    const families = Object.keys(run.summary?.families || {}).length;
    button.type = "button";
    button.className = `run-item ${run.status}${state.selected?.id === run.id ? " active" : ""}`;
    button.innerHTML = `
      <span class="run-item-title"><i></i><strong>${escapeHtml(run.name)}</strong></span>
      <span class="run-item-path">${escapeHtml(run.relative_path)}</span>
      <span class="run-item-meta"><span>${escapeHtml(device)}</span><i aria-hidden="true">·</i><span>${run.summary?.layers || 0} 层</span><i aria-hidden="true">·</i><span>${families} 族</span></span>`;
    button.addEventListener("click", () => selectRun(run.id));
    list.append(button);
  });
}

async function loadRuns({ refresh = false, keepSelection = true } = {}) {
  const button = $("refresh-button");
  button.classList.add("loading");
  try {
    const payload = await api(refresh ? "/api/v1/refresh" : "/api/v1/runs", refresh ? { method: "POST" } : undefined);
    state.runs = payload.runs || [];
    state.roots = payload.roots || [];
    setConnection(true, `${state.runs.length} 条运行记录 · 本地只读`);
    $("fatal-state").hidden = true;
    renderRuns();
    if (keepSelection && state.selected && state.runs.some((run) => run.id === state.selected.id)) {
      await selectRun(state.selected.id);
    } else if (state.runs.length) {
      await selectRun(state.runs[0].id);
    } else {
      state.selected = null;
      $("workspace-empty").hidden = false;
      $("workspace-content").hidden = true;
    }
    if (payload.scan_errors?.length) showToast(`已跳过 ${payload.scan_errors.length} 个无效或越界文件`);
  } catch (error) {
    setConnection(false, "本地数据服务不可用");
    $("fatal-message").textContent = error.message;
    $("fatal-state").hidden = false;
  } finally {
    button.classList.remove("loading");
  }
}

async function selectRun(id) {
  try {
    state.selected = await api(`/api/v1/runs/${id}`);
    state.selectedLayer = state.selected.layers?.[0]?.id || null;
    state.familyFilter = "all";
    state.layerStatus = "all";
    state.layerQuery = "";
    state.benchmarkFamily = Object.keys(state.selected.benchmark?.families || {})[0] || null;
    $("family-filter").value = "all";
    $("layer-status-filter").value = "all";
    $("layer-search").value = "";
    $("workspace-empty").hidden = true;
    $("workspace-content").hidden = false;
    renderRuns();
    renderWorkspace();
  } catch (error) {
    showToast(`运行读取失败：${error.message}`);
  }
}

function renderWorkspace() {
  const run = state.selected;
  if (!run) return;
  $("run-path").textContent = run.relative_path;
  $("run-title").textContent = run.name;
  $("layer-tab-count").textContent = run.layers.length;
  const badge = $("run-status");
  badge.className = `health-badge ${run.status}`;
  badge.textContent = statusLabel[run.status] || run.status;
  const meta = {
    设备: run.metadata.device || run.metadata.resolved_device,
    数据集: run.metadata.dataset || run.metadata.data,
    代码版本: run.git.commit ? run.git.commit.slice(0, 10) : null,
  };
  $("run-meta").innerHTML = Object.entries(meta).filter(([, value]) => value != null).map(([key, value]) => `<span>${escapeHtml(key)} <b>${escapeHtml(value)}</b></span>`).join("");
  renderOverview();
  renderAnalysis();
  renderLayerFilters();
  renderLayers();
  renderBenchmark();
  renderEvidence();
}

function metric(label, value, note) {
  return `<div class="metric"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong><small>${escapeHtml(note)}</small></div>`;
}

function renderOverview() {
  const run = state.selected;
  const summary = run.summary;
  const auxActive = run.layers.filter((layer) => layer.aux.status === "active_training").length;
  const warningLayers = run.layers.filter(layerIsWarning).length;

  $("metric-row").innerHTML = [
    metric("采集到的路由层", summary.layers, "个"),
    metric("覆盖的路由族", Object.keys(summary.families).length, "种"),
    metric("需要关注的层", warningLayers, warningLayers ? "请打开层级负载" : "未发现异常"),
    metric("辅助损失生效", auxActive, auxActive ? "个训练态路由层" : "当前不是训练态或未配置"),
  ].join("");

  const familyEntries = Object.entries(summary.families);
  $("family-overview").innerHTML = familyEntries.length
    ? familyEntries.map(([family, count], index) => `<div class="family-row"><div class="family-name"><i style="opacity:${Math.max(.38, 1 - index * .12)}"></i><div><strong>${escapeHtml(family)}</strong><small>单层最多 ${run.layers.filter((layer) => layer.family === family).reduce((max, layer) => Math.max(max, layer.num_experts), 0)} 个专家</small></div></div><b>${count} 层</b></div>`).join("")
    : '<div class="run-list-empty">没有可显示的路由族</div>';

  const lowest = [...run.layers].filter((layer) => finite(layer.normalized_entropy)).sort((a, b) => a.normalized_entropy - b.normalized_entropy).slice(0, 6);
  $("entropy-overview").innerHTML = lowest.length
    ? lowest.map((layer) => `<button class="entropy-item" type="button" data-layer="${layer.id}"><div class="entropy-label"><strong title="${escapeHtml(layer.name)}">${escapeHtml(layer.name)}</strong><small>${escapeHtml(layer.family)}</small></div><div class="entropy-track"><i style="width:${Math.max(0, Math.min(100, layer.normalized_entropy * 100))}%"></i></div><b>${decimal(layer.normalized_entropy)}</b></button>`).join("")
    : '<div class="run-list-empty">当前运行没有可用熵数据</div>';
  $("entropy-overview").querySelectorAll("[data-layer]").forEach((button) => button.addEventListener("click", () => {
    state.selectedLayer = button.dataset.layer;
    renderLayers();
    switchView("layers");
  }));

  renderTrainingSummary();
  renderPerformanceSummary();

  const findings = [];
  if (summary.invalid_layers) findings.push(["字段校验异常", `${summary.invalid_layers} 个层存在专家向量长度或熵范围问题。`]);
  if (summary.unsupported_layers) findings.push(["存在不支持的层", `${summary.unsupported_layers} 个层被显式标记为 unsupported。`]);
  if (summary.collapsed_layers) findings.push(["可能发生路由坍塌", `${summary.collapsed_layers} 个层触发了运行时坍塌检测。`]);
  if (run.status === "empty") findings.push(["未记录路由层", "若这不是基线组，请检查 routing_enabled 和采样间隔。"]);
  $("finding-panel").hidden = findings.length === 0;
  $("finding-list").innerHTML = findings.map(([title, copy]) => `<div class="finding"><i>i</i><div><strong>${escapeHtml(title)}</strong><p>${escapeHtml(copy)}</p></div></div>`).join("");
}

function renderTrainingSummary() {
  const runs = state.selected.training_runs || {};
  const entries = Object.entries(runs);
  const panel = $("training-summary-panel");
  panel.hidden = entries.length === 0;
  if (!entries.length) return;
  $("training-summary").innerHTML = entries.map(([family, run]) => {
    if (finite(run.epochs)) {
      const valMap = run.metrics?.["metrics/mAP50-95(B)"];
      const testMap = run.test_metrics?.["metrics/mAP50-95(B)"];
      return `<div class="training-row"><div class="summary-family"><i></i><div><strong>${escapeHtml(family)}</strong><small>${escapeHtml(run.routed_layers)} 个路由层</small></div></div><dl><div><dt>完整训练</dt><dd>${escapeHtml(run.epochs)} epochs</dd></div><div><dt>验证 mAP50-95</dt><dd>${decimal(valMap, 4)}</dd></div><div><dt>测试 mAP50-95</dt><dd>${decimal(testMap, 4)}</dd></div></dl><span class="summary-state">训练完成</span></div>`;
    }
    const clean = !(run.invalid_layers || run.unsupported_layers);
    const auxActive = (run.aux_statuses || []).includes("active_training");
    return `<div class="training-row"><div class="summary-family"><i></i><div><strong>${escapeHtml(family)}</strong><small>${escapeHtml(run.routed_layers)} 个路由层</small></div></div><dl><div><dt>路由观测</dt><dd>${escapeHtml(run.routing_observations)} 次</dd></div><div><dt>日志指标</dt><dd>${escapeHtml(run.tensorboard_routing_scalar_tags)} 项</dd></div><div><dt>辅助损失</dt><dd>${auxActive ? "训练中生效" : "未生效"}</dd></div></dl><span class="summary-state ${clean ? "" : "warning"}">${clean ? "数据完整" : "需要检查"}</span></div>`;
  }).join("");
}

function renderPerformanceSummary() {
  const families = state.selected.benchmark?.families || {};
  const entries = Object.entries(families);
  const panel = $("performance-summary-panel");
  panel.hidden = entries.length === 0;
  const hasTrainingSummary = Object.keys(state.selected.training_runs || {}).length > 0;
  $("overview-details-grid").hidden = !hasTrainingSummary && entries.length === 0;
  if (!entries.length) return;
  $("performance-summary").innerHTML = entries.map(([family, result]) => {
    const interval = result.paired_bootstrap_95_ci_percent || [];
    return `<button type="button" data-family="${escapeHtml(family)}"><div class="summary-family"><i></i><div><strong>${escapeHtml(family)}</strong><small>95% 区间 ${pct(interval[0])} 至 ${pct(interval[1])}</small></div></div><b>${pct(result.mean_overhead_percent)}</b><span class="summary-state ${result.passed ? "" : "warning"}">${result.passed ? "通过" : "未通过"}</span></button>`;
  }).join("");
  $("performance-summary").querySelectorAll("button").forEach((button) => button.addEventListener("click", () => {
    state.benchmarkFamily = button.dataset.family;
    renderBenchmark();
    switchView("benchmark");
  }));
}

function familyColor(family) {
  const families = Object.keys(state.selected.summary?.families || {}).sort();
  return familyColors[Math.max(0, families.indexOf(family)) % familyColors.length];
}

function openLayer(layerId) {
  state.selectedLayer = layerId;
  renderLayers();
  switchView("layers");
}

function renderAnalysis() {
  const run = state.selected;
  const layers = run.layers || [];
  const layerDataMissing = layers.length === 0;
  $("matrix-empty").hidden = !layerDataMissing;
  $("matrix-content").hidden = layerDataMissing;
  $("balance-empty").hidden = !layerDataMissing;
  $("balance-content").hidden = layerDataMissing;
  if (!layerDataMissing) {
    renderLoadMatrix(layers);
    renderBalanceChart(layers);
  }
  renderVisualGallery(run);
}

function renderLoadMatrix(layers) {
  const maxExperts = Math.max(...layers.map((layer) => layer.num_experts), 1);
  const relativeDeltas = layers.flatMap((layer) => {
    const uniform = layer.num_experts ? 1 / layer.num_experts : 0;
    return layer.expert_usage.filter(finite).map((value) => uniform ? (value - uniform) / uniform : 0);
  });
  const maxDelta = Math.max(...relativeDeltas.map(Math.abs), .001);
  const header = `<div class="matrix-corner">路由层</div>${Array.from({ length: maxExperts }, (_, index) => `<div class="matrix-expert">E${index + 1}</div>`).join("")}`;
  const rows = layers.map((layer) => {
    const uniform = layer.num_experts ? 1 / layer.num_experts : 0;
    const cells = Array.from({ length: maxExperts }, (_, index) => {
      if (index >= layer.num_experts) return '<span class="matrix-cell unavailable" aria-hidden="true"></span>';
      const value = finite(layer.expert_usage[index]) ? layer.expert_usage[index] : 0;
      const relativeDelta = uniform ? (value - uniform) / uniform : 0;
      const intensity = Math.sqrt(Math.min(1, Math.abs(relativeDelta) / maxDelta));
      const lightness = 97 - intensity * 48;
      const background = relativeDelta >= 0 ? `hsl(216 78% ${lightness}%)` : `hsl(27 88% ${lightness}%)`;
      const foreground = intensity > .62 ? "#fff" : "#1f3349";
      const deltaPoints = (value - uniform) * 100;
      const display = Math.abs(deltaPoints) < .005 ? "0" : `${deltaPoints > 0 ? "+" : ""}${deltaPoints.toFixed(1)}`;
      return `<button class="matrix-cell" type="button" data-layer="${layer.id}" style="--cell-bg:${background};--cell-text:${foreground}" aria-label="${escapeHtml(layer.name)}，专家 ${index + 1}，实际负载 ${(value * 100).toFixed(2)}%，相对均匀份额偏差 ${display} 个百分点" title="实际 ${(value * 100).toFixed(2)}% · 均匀 ${(uniform * 100).toFixed(2)}% · 偏差 ${display}pp">${display}</button>`;
    }).join("");
    return `<button class="matrix-layer" type="button" data-layer="${layer.id}" title="${escapeHtml(layer.name)}"><strong>${escapeHtml(layer.name)}</strong><small>${escapeHtml(layer.family)} · ${decimal(layer.normalized_entropy)}</small></button>${cells}`;
  }).join("");
  $("load-matrix").style.setProperty("--experts", maxExperts);
  $("load-matrix").innerHTML = header + rows;
  $("load-matrix").querySelectorAll("[data-layer]").forEach((button) => button.addEventListener("click", () => openLayer(button.dataset.layer)));
}

function renderBalanceChart(layers) {
  const points = layers.filter((layer) => finite(layer.normalized_entropy) && finite(layer.load_spread)).map((layer) => {
    const uniform = layer.num_experts ? 1 / layer.num_experts : 1;
    return {
      layer,
      entropyGap: Math.max(1e-8, 1 - layer.normalized_entropy),
      relativeSpread: Math.max(1e-8, layer.load_spread / uniform),
    };
  });
  const xLogs = points.map((point) => Math.log10(point.entropyGap));
  const yLogs = points.map((point) => Math.log10(point.relativeSpread));
  const xMin = Math.min(...xLogs), xMax = Math.max(...xLogs);
  const yMin = Math.min(...yLogs), yMax = Math.max(...yLogs);
  const position = (value, low, high) => high === low ? .5 : (value - low) / (high - low);
  const legend = Object.keys(state.selected.summary.families).sort().map((family) => `<span><i style="--dot:${familyColor(family)}"></i>${escapeHtml(family)}</span>`).join("");
  const dots = points.map((point) => {
    const x = 74 + position(Math.log10(point.entropyGap), xMin, xMax) * 786;
    const y = 350 - position(Math.log10(point.relativeSpread), yMin, yMax) * 300;
    const title = `${point.layer.name} · 熵缺口 ${point.entropyGap.toExponential(2)} · 相对负载差 ${(point.relativeSpread * 100).toFixed(2)}%`;
    return `<circle class="balance-dot" data-layer="${point.layer.id}" cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="7" fill="${familyColor(point.layer.family)}" tabindex="0" role="button"><title>${escapeHtml(title)}</title></circle>`;
  }).join("");
  const grid = [0, .25, .5, .75, 1].map((ratio) => `<line x1="74" y1="${350 - ratio * 300}" x2="860" y2="${350 - ratio * 300}"/><line x1="${74 + ratio * 786}" y1="50" x2="${74 + ratio * 786}" y2="350"/>`).join("");
  $("balance-chart").innerHTML = `<div class="balance-legend">${legend}</div><svg class="balance-svg" viewBox="0 0 900 410" role="img" aria-label="熵缺口与相对专家负载差二维分布"><g class="balance-grid">${grid}</g><line class="balance-axis" x1="74" y1="350" x2="860" y2="350"/><line class="balance-axis" x1="74" y1="50" x2="74" y2="350"/><text class="balance-axis-title" x="467" y="397">熵缺口（对数展开，越右偏离均匀越多）</text><text class="balance-axis-title" transform="translate(20 200) rotate(-90)">相对峰谷差（对数展开）</text><text class="balance-tick" x="74" y="373">${(10 ** xMin).toExponential(1)}</text><text class="balance-tick" x="860" y="373" text-anchor="end">${(10 ** xMax).toExponential(1)}</text><text class="balance-tick" x="66" y="354" text-anchor="end">${(10 ** yMin * 100).toFixed(2)}%</text><text class="balance-tick" x="66" y="54" text-anchor="end">${(10 ** yMax * 100).toFixed(1)}%</text>${dots}</svg>`;
  $("balance-chart").querySelectorAll("[data-layer]").forEach((button) => button.addEventListener("click", () => openLayer(button.dataset.layer)));
}

function renderVisualGallery(run) {
  const dashboards = run.evidence.filter((file) => file.media_type === "image/png" && file.name.includes("routing_dashboard"));
  const available = dashboards.length > 0;
  $("spatial-empty").hidden = available;
  $("visual-artifacts-panel").hidden = !available;
  if (!available) return;
  const familyOf = (file) => (file.relative_path || file.name).match(/routing_visualizations[/\\]([^/\\]+)/)?.[1] || "routing";
  const cards = dashboards.map((file) => `<article class="spatial-card"><header><strong>${escapeHtml(familyOf(file))}</strong><span>代表层空间路由</span></header><a href="/api/v1/runs/${run.id}/files/${file.id}" target="_blank" rel="noreferrer"><img src="/api/v1/runs/${run.id}/files/${file.id}" alt="${escapeHtml(visualLabel(file.name))}" loading="lazy"></a></article>`).join("");
  $("visual-gallery").innerHTML = `<div class="spatial-showcase">${cards}</div>`;
}

function visualLabel(name) {
  const layer = name.replace(/\.png$/i, "").replace(/_/g, ".");
  if (name.includes("expert_usage")) return `${name.split("_")[0]} · 各层专家负载`;
  if (name.includes("routing_distribution")) return `${layer.replace(".routing.distribution", "")} · 路由分布`;
  if (name.includes("routing_dashboard")) return `${layer.replace(".routing.dashboard", "")} · 路由总览`;
  if (name.includes("assignment_map")) return `${layer.replace(".assignment.map", "")} · 专家分配图`;
  if (name.includes("confidence_heatmap")) return `${layer.replace(".confidence.heatmap", "")} · 路由置信度`;
  const expert = name.match(/expert_(\d+)_heatmap/);
  if (expert) return `${layer.replace(/\.expert\.\d+\.heatmap/, "")} · 专家 ${Number(expert[1]) + 1} 权重图`;
  return layer;
}

function renderLayerFilters() {
  const families = [...new Set(state.selected.layers.map((layer) => layer.family))].sort();
  $("family-filter").innerHTML = '<option value="all">全部</option>' + families.map((family) => `<option value="${escapeHtml(family)}">${escapeHtml(family)}</option>`).join("");
  $("family-filter").value = state.familyFilter;
}

function layerIsWarning(layer) {
  return layer.issues.length > 0 || (finite(layer.normalized_entropy) && layer.normalized_entropy < 0.5);
}

function visibleLayers() {
  return state.selected.layers.filter((layer) => {
    const family = state.familyFilter === "all" || layer.family === state.familyFilter;
    const status = state.layerStatus === "all" || (state.layerStatus === "warning") === layerIsWarning(layer);
    const query = `${layer.name} ${layer.module_type}`.toLowerCase().includes(state.layerQuery.toLowerCase());
    return family && status && query;
  });
}

function renderLayers() {
  const layers = visibleLayers();
  const body = $("layer-table");
  $("layer-table-empty").hidden = layers.length > 0;
  body.innerHTML = layers.map((layer) => {
    const warning = layerIsWarning(layer);
    return `<tr data-layer="${layer.id}" class="${layer.id === state.selectedLayer ? "selected" : ""}" tabindex="0">
      <td class="layer-cell"><strong title="${escapeHtml(layer.name)}">${escapeHtml(layer.name)}</strong><small>${escapeHtml(layer.module_type)}</small></td>
      <td><span class="family-tag">${escapeHtml(layer.family)}</span></td><td class="number-cell">${layer.num_experts}</td>
      <td class="number-cell">${decimal(layer.normalized_entropy)}</td><td class="number-cell">${finite(layer.load_spread) ? `${(layer.load_spread * 100).toFixed(2)}%` : "—"}</td>
      <td><span class="aux-tag" title="原始状态：${escapeHtml(layer.aux.status)}">${escapeHtml(auxLabel[layer.aux.status] || layer.aux.status)}</span></td>
      <td><span class="state-tag ${warning ? "warning" : ""}">${warning ? "关注" : "正常"}</span></td></tr>`;
  }).join("");
  body.querySelectorAll("tr").forEach((row) => {
    const open = () => { state.selectedLayer = row.dataset.layer; renderLayers(); };
    row.addEventListener("click", open);
    row.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") open(); });
  });
  const selected = state.selected.layers.find((layer) => layer.id === state.selectedLayer);
  renderInspector(selected || layers[0]);
}

function renderInspector(layer) {
  const inspector = $("layer-inspector");
  if (!layer) {
    inspector.innerHTML = '<div class="inspector-empty">当前筛选没有可检查的路由层</div>';
    return;
  }
  const uniform = layer.num_experts ? 1 / layer.num_experts : 0;
  const bars = Array.from({ length: layer.num_experts }, (_, index) => {
    const usage = finite(layer.expert_usage[index]) ? layer.expert_usage[index] : 0;
    const probability = finite(layer.mean_router_probs[index]) ? layer.mean_router_probs[index] : 0;
    return `<div class="expert-comparison"><label>专家 ${index + 1}</label><div class="expert-series"><div><span>负载</span><div class="expert-track usage"><i style="width:${Math.max(0, Math.min(100, usage * 100))}%"></i><em style="left:${Math.min(100, uniform * 100)}%" title="均匀分配位置"></em></div><b>${(usage * 100).toFixed(1)}%</b></div><div><span>权重</span><div class="expert-track probability"><i style="width:${Math.max(0, Math.min(100, probability * 100))}%"></i><em style="left:${Math.min(100, uniform * 100)}%" title="均匀分配位置"></em></div><b>${(probability * 100).toFixed(1)}%</b></div></div></div>`;
  }).join("");
  inspector.innerHTML = `
    <div class="inspector-head"><span>${escapeHtml(layer.family)} · ${escapeHtml(axisLabel[layer.routing_axis] || layer.routing_axis)}路由</span><h2>${escapeHtml(layer.name)}</h2><small>${escapeHtml(layer.module_type)}</small></div>
    <div class="inspector-section"><div class="section-title"><span>专家负载与平均权重</span><small><i></i>均匀分配位置</small></div><div class="expert-bars">${bars || '<div class="run-list-empty">无专家向量</div>'}</div></div>
    <div class="inspector-section"><dl class="inspector-dl">
      <div><dt>均衡度</dt><dd>${decimal(layer.normalized_entropy)} / 1</dd></div><div><dt>原始路由熵</dt><dd>${decimal(layer.entropy)}</dd></div>
      <div><dt>专家数量</dt><dd>${layer.num_experts}</dd></div><div><dt>最大专家占比</dt><dd>${finite(layer.dominant_share) ? `${(layer.dominant_share * 100).toFixed(1)}%` : "—"}</dd></div>
      <div><dt>Top-k</dt><dd>${escapeHtml(layer.top_k)}</dd></div><div><dt>负载差</dt><dd>${finite(layer.load_spread) ? `${(layer.load_spread * 100).toFixed(2)}%` : "—"}</dd></div>
      <div><dt>分发方式</dt><dd>${escapeHtml(dispatchLabel[layer.dispatch_policy] || layer.dispatch_policy)}</dd></div><div><dt>辅助损失</dt><dd>${escapeHtml(auxLabel[layer.aux.status] || layer.aux.status)}</dd></div>
      <div><dt>Aux 已配置</dt><dd>${layer.aux.configured ? "是" : "否"}</dd></div><div><dt>Aux 观测值</dt><dd>${decimal(layer.aux.observed, 6)}</dd></div>
      <div><dt>概率张量</dt><dd>${escapeHtml(layer.probability_shape.join(" × "))}</dd></div>
    </dl></div>`;
}

function renderBenchmark() {
  const benchmark = state.selected.benchmark;
  const empty = !benchmark || !benchmark.families || !Object.keys(benchmark.families).length;
  $("benchmark-empty").hidden = !empty;
  $("benchmark-content").hidden = empty;
  if (empty) {
    const benchmarkRun = state.runs.find((run) => run.summary?.has_benchmark);
    $("benchmark-locate").hidden = !benchmarkRun;
    $("benchmark-empty-copy").textContent = benchmarkRun
      ? "当前是单次运行记录；配对汇总保存在另一条 benchmark 记录中。"
      : "路由诊断仍可正常使用。完成配对 benchmark 后，此处会自动出现。";
    return;
  }
  const families = Object.keys(benchmark.families);
  if (!families.includes(state.benchmarkFamily)) state.benchmarkFamily = families[0];
  $("benchmark-families").innerHTML = families.map((family) => `<button type="button" data-family="${escapeHtml(family)}" class="${family === state.benchmarkFamily ? "active" : ""}">${escapeHtml(family)}</button>`).join("");
  $("benchmark-families").querySelectorAll("button").forEach((button) => button.addEventListener("click", () => { state.benchmarkFamily = button.dataset.family; renderBenchmark(); }));
  const summary = benchmark.families[state.benchmarkFamily];
  const pairs = summary.pairs || [];
  const values = pairs.map((pair) => pair.overhead_percent).filter(finite);
  const threshold = finite(summary.threshold_percent) ? summary.threshold_percent : 10;
  const min = Math.min(-2, ...values);
  const max = Math.max(threshold + 1, ...values);
  const position = (value) => `${(value - min) / (max - min) * 100}%`;
  $("benchmark-title").textContent = `${state.benchmarkFamily} · ${pairs.length} 轮配对结果`;
  $("pair-chart").innerHTML = pairs.map((pair) => `<div class="pair-row"><span>第 ${pair.pair} 轮<small>${decimal(pair.baseline_mean_milliseconds, 1)} → ${decimal(pair.e3_mean_milliseconds, 1)} ms</small></span><div class="pair-axis" style="--zero:${position(0)};--threshold:${position(threshold)}"><i class="pair-dot" style="--position:${position(pair.overhead_percent)}"></i></div><b>${pct(pair.overhead_percent)}</b></div>`).join("") + `<div class="axis-labels" style="--zero:${position(0)};--threshold:${position(threshold)}"><span>${min.toFixed(0)}%</span><span class="axis-zero">0% 基线</span><span class="axis-threshold">${threshold}% 上限</span></div>`;
  const familyRuns = benchmark.runs?.filter((run) => run.family === state.benchmarkFamily && run.condition === "e3") || [];
  const ci = summary.paired_bootstrap_95_ci_percent || [];
  const memory = Math.max(...familyRuns.map((run) => run.memory?.peak_device_memory_bytes || run.memory?.sampled_current_memory_bytes_max || 0), 0);
  $("benchmark-summary").innerHTML = `<span>平均额外开销</span><strong>${pct(summary.mean_overhead_percent)}</strong><div class="benchmark-verdict ${summary.passed ? "" : "failed"}">${summary.passed ? `✓ 低于 ${threshold}% 验收上限` : `! 高于 ${threshold}% 验收上限`}</div><dl class="benchmark-details">
    <div><dt>95% 置信区间</dt><dd>[${pct(ci[0])}, ${pct(ci[1])}]</dd></div><div><dt>测试轮数</dt><dd>${pairs.length} 轮</dd></div>
    <div><dt>平均延迟</dt><dd>${decimal(average(familyRuns.map((run) => run.mean_milliseconds)), 1)} ms</dd></div><div><dt>平均 p95</dt><dd>${decimal(average(familyRuns.map((run) => run.p95_milliseconds)), 1)} ms</dd></div>
    <div><dt>平均吞吐</dt><dd>${decimal(average(familyRuns.map((run) => run.samples_per_second)), 2)} 样本/秒</dd></div><div><dt>设备内存</dt><dd>${bytes(memory)}</dd></div>
  </dl><p class="benchmark-note">负值表示这轮开启采集反而更快，通常视为正常测量波动，不代表采集能加速训练。</p>`;
}

function renderEvidence() {
  const run = state.selected;
  const metadata = run.metadata || {};
  const snapshot = run.snapshot_metadata || {};
  const sample = snapshot.sample || {};
  const provenance = {
    "相对路径": run.relative_path,
    "数据格式": run.schemas.join(", "),
    "代码提交": run.git.commit,
    "代码分支": run.git.branch,
    "存在未提交改动": run.git.dirty === true ? "是" : run.git.dirty === false ? "否" : run.git.dirty,
    "设备": metadata.device || metadata.resolved_device,
    "系统": metadata.platform,
    "数据集": metadata.dataset || metadata.data,
    "输入尺寸": snapshot.imgsz ? `${snapshot.imgsz} × ${snapshot.imgsz}` : metadata.imgsz,
    "Batch": metadata.batch,
    "随机种子": snapshot.seed ?? metadata.seed,
    "路由采样间隔": metadata.routing_interval_steps ? `每 ${metadata.routing_interval_steps} step` : undefined,
    "预热/测量步数": metadata.warmup_steps != null && metadata.measured_steps != null ? `${metadata.warmup_steps} / ${metadata.measured_steps}` : undefined,
    "权重来源": snapshot.weights,
    "采集用途": snapshot.scope,
    "数据样本": sample.path,
    "样本 SHA-256": sample.sha256,
    "Python": metadata.python,
    "PyTorch": metadata.torch,
    ...Object.fromEntries(Object.entries(run.models || {}).map(([family, model]) => [`${family} 模型`, model.model])),
  };
  $("provenance-list").innerHTML = Object.entries(provenance).filter(([, value]) => value !== undefined && value !== null && value !== "").map(([key, value]) => `<div><dt>${escapeHtml(key)}</dt><dd>${escapeHtml(value)}</dd></div>`).join("") || '<div><dt>状态</dt><dd>当前证据没有来源元数据</dd></div>';
  const limitations = run.limitations || [];
  $("limitations").hidden = limitations.length === 0;
  $("limitations-list").innerHTML = limitations.map((item) => `<li>${escapeHtml(item)}</li>`).join("");
  const sourceFiles = run.evidence.filter((file) => file.media_type !== "image/png");
  $("evidence-list").innerHTML = sourceFiles.map((file) => `<a class="evidence-file" href="/api/v1/runs/${run.id}/files/${file.id}" target="_blank" rel="noreferrer"><span class="file-icon">JSON</span><div><strong>${escapeHtml(file.name)}</strong><small>${escapeHtml(file.schema)} · ${bytes(file.size)}</small></div><em>打开 ↗</em></a>`).join("");
}

function switchView(view) {
  state.currentView = view;
  document.querySelectorAll(".view-tabs button").forEach((button) => button.classList.toggle("active", button.dataset.view === view));
  document.querySelectorAll(".view").forEach((section) => {
    const active = section.id === `view-${view}`;
    section.hidden = !active;
    section.classList.toggle("active", active);
  });
}

function bindEvents() {
  $("refresh-button").addEventListener("click", () => loadRuns({ refresh: true }));
  $("retry-button").addEventListener("click", () => loadRuns());
  $("benchmark-locate").addEventListener("click", () => {
    const run = state.runs.find((item) => item.summary?.has_benchmark);
    if (run) selectRun(run.id);
  });
  $("run-search").addEventListener("input", (event) => { state.runQuery = event.target.value; renderRuns(); });
  document.querySelectorAll(".status-filters button").forEach((button) => button.addEventListener("click", () => {
    state.runStatus = button.dataset.status;
    document.querySelectorAll(".status-filters button").forEach((item) => item.classList.toggle("active", item === button));
    renderRuns();
  }));
  document.querySelectorAll(".view-tabs button").forEach((button) => button.addEventListener("click", () => switchView(button.dataset.view)));
  document.querySelectorAll("[data-open-view]").forEach((button) => button.addEventListener("click", () => switchView(button.dataset.openView)));
  $("family-filter").addEventListener("change", (event) => { state.familyFilter = event.target.value; state.selectedLayer = null; renderLayers(); });
  $("layer-status-filter").addEventListener("change", (event) => { state.layerStatus = event.target.value; state.selectedLayer = null; renderLayers(); });
  $("layer-search").addEventListener("input", (event) => { state.layerQuery = event.target.value; state.selectedLayer = null; renderLayers(); });
  $("copy-path").addEventListener("click", async () => {
    if (!state.selected) return;
    try { await navigator.clipboard.writeText(state.selected.relative_path); showToast("相对路径已复制"); }
    catch { showToast(state.selected.relative_path); }
  });
}

bindEvents();
loadRuns();
