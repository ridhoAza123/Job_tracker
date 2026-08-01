/* Speed Tracker AI — dashboard frontend.
 *
 * Talks to the existing Flask API only (no backend contract changes):
 *   /api/status /api/live /api/statistics /api/detections /api/cameras
 *   /api/cameras/refresh /api/camera /api/settings /api/roi /api/capture
 *   /api/stats/reset /stream /healthz
 */
(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const isDashboard = !!$('liveVideo');
  const isHistory = !!$('historyTable');

  /* =================================================================== */
  /* Fetch helpers                                                       */
  /* =================================================================== */
  const api = async (url, options) => {
    try {
      const response = await fetch(url, options);
      if (!response.ok) return null;
      return await response.json();
    } catch (error) {
      return null;
    }
  };
  const postJSON = (url, body) => api(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}),
  });
  const deleteJSON = (url) => api(url, { method: 'DELETE' });

  const escapeHtml = (value) => String(value ?? '').replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  const setText = (id, value) => { const el = $(id); if (el) el.textContent = value; };
  const setHtml = (id, value) => { const el = $(id); if (el) el.innerHTML = value; };

  /* =================================================================== */
  /* Toast notifications                                                  */
  /* =================================================================== */
  const TOAST_ICONS = { success: 'check-circle', danger: 'x-circle', warning: 'alert-triangle', info: 'sparkles' };

  const showToast = ({ type = 'info', title, message, duration = 5000 }) => {
    const container = $('toastContainer');
    if (!container) return;
    const node = document.createElement('div');
    node.className = `toast ${type}`;
    node.innerHTML = `
      <i data-lucide="${TOAST_ICONS[type] || 'info'}" class="icon toast-icon"></i>
      <div class="toast-body">
        <div class="toast-title">${escapeHtml(title)}</div>
        ${message ? `<div class="toast-message">${escapeHtml(message)}</div>` : ''}
      </div>
      <i data-lucide="x" class="icon icon-sm toast-close"></i>`;
    container.appendChild(node);
    if (window.lucide) window.lucide.createIcons();

    const remove = () => {
      node.classList.add('leaving');
      setTimeout(() => node.remove(), 260);
    };
    node.querySelector('.toast-close')?.addEventListener('click', remove);
    setTimeout(remove, duration);
  };

  /* =================================================================== */
  /* Number count-up animation                                           */
  /* =================================================================== */
  const animateValue = (id, toValue, decimals = 0) => {
    const el = $(id);
    if (!el) return;
    const from = parseFloat(el.dataset.raw || el.textContent) || 0;
    const to = Number(toValue) || 0;
    if (Math.abs(to - from) < 10 ** -decimals) {
      el.textContent = to.toFixed(decimals);
      el.dataset.raw = to;
      return;
    }
    const duration = 450;
    const start = performance.now();
    const step = (now) => {
      const progress = Math.min((now - start) / duration, 1);
      const eased = 1 - (1 - progress) ** 3;
      const current = from + (to - from) * eased;
      el.textContent = current.toFixed(decimals);
      if (progress < 1) requestAnimationFrame(step);
      else el.dataset.raw = to;
    };
    requestAnimationFrame(step);
    el.classList.remove('count-flash');
    void el.offsetWidth;
    el.classList.add('count-flash');
  };

  /* =================================================================== */
  /* Clock                                                                */
  /* =================================================================== */
  const tickClock = () => {
    const now = new Date();
    setText('headerClock', now.toLocaleTimeString('id-ID', { hour12: false }));
    setText('headerDate', now.toLocaleDateString('id-ID', {
      weekday: 'long', day: 'numeric', month: 'long', year: 'numeric',
    }));
  };

  /* =================================================================== */
  /* Status pill / dot helpers                                           */
  /* =================================================================== */
  const dotClass = (status) => {
    if (status === 'ONLINE') return 'online';
    if (['OFFLINE', 'NOT FOUND', 'FAILED', 'STOPPED'].includes(status)) return 'offline';
    return 'pending';
  };
  const badgeClass = (status) => {
    if (status === 'ONLINE') return 'badge-success';
    if (['OFFLINE', 'NOT FOUND', 'FAILED', 'STOPPED'].includes(status)) return 'badge-danger';
    return 'badge-warning';
  };
  const speedClass = (status) => {
    if (status === 'Overspeed') return 'speed-over';
    if (status === 'Normal') return 'speed-normal';
    return 'speed-idle';
  };
  const applyDot = (id, status) => {
    const el = $(id);
    if (el) el.className = `status-dot ${dotClass(status)}${id.includes('Cctv') || id === 'ssStatusDot' ? ' pulse' : ''}`;
  };

  /* =================================================================== */
  /* Sidebar (mobile drawer) + settings modal + smooth-scroll nav         */
  /* =================================================================== */
  const initChrome = () => {
    const sidebar = $('sidebar');
    const backdrop = $('sidebarBackdrop');
    const toggle = $('sidebarToggle');
    const openSidebar = () => { sidebar?.classList.add('open'); backdrop?.classList.add('open'); };
    const closeSidebar = () => { sidebar?.classList.remove('open'); backdrop?.classList.remove('open'); };
    toggle?.addEventListener('click', () => {
      sidebar?.classList.contains('open') ? closeSidebar() : openSidebar();
    });
    backdrop?.addEventListener('click', closeSidebar);
    document.querySelectorAll('.nav-item[data-scroll]').forEach((link) => {
      link.addEventListener('click', (event) => {
        if (window.location.pathname !== '/') return; // let it navigate to dashboard first
        event.preventDefault();
        closeSidebar();
        document.getElementById(link.dataset.scroll)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      });
    });

    const modal = $('settingsModal');
    const openModal = () => { modal?.classList.add('open'); loadSettingsIntoModal(); };
    const closeModal = () => modal?.classList.remove('open');
    $('navSettings')?.addEventListener('click', (event) => { event.preventDefault(); closeSidebar(); openModal(); });
    $('settingsClose')?.addEventListener('click', closeModal);
    $('settingsDone')?.addEventListener('click', closeModal);
    modal?.addEventListener('click', (event) => { if (event.target === modal) closeModal(); });
    document.addEventListener('keydown', (event) => { if (event.key === 'Escape') closeModal(); });
  };

  /* =================================================================== */
  /* Settings modal <-> /api/settings                                     */
  /* =================================================================== */
  const bindDebouncedSlider = (sliderId, labelId, key, format) => {
    const slider = $(sliderId);
    if (!slider) return;
    let timer = null;
    slider.addEventListener('input', () => {
      const value = Number(slider.value);
      setText(labelId, format ? format(value) : value.toFixed(2));
      clearTimeout(timer);
      timer = setTimeout(() => postJSON('/api/settings', { [key]: value }), 150);
    });
  };

  const loadSettingsIntoModal = async () => {
    const data = await api('/api/settings');
    if (data) {
      $('settingsConfSlider').value = data.confidence;
      setText('settingsConfValue', Number(data.confidence).toFixed(2));
      $('settingsIouSlider').value = data.iou;
      setText('settingsIouValue', Number(data.iou).toFixed(2));
      $('settingsLimitSlider').value = data.speed_limit;
      setText('settingsLimitValue', String(Math.round(data.speed_limit)));
      $('settingsPreprocessToggle').checked = !!data.preprocess;
    }
    const status = await api('/api/status');
    if (status) {
      setText('settingsModelValue', status.model || '–');
      setText('settingsTrackerValue', status.tracker || '–');
      setText('settingsResolutionValue', status.resolution || '–');
      setText('settingsFpsValue', status.fps !== undefined ? `${status.fps} fps` : '–');
    }
  };

  const initSettingsModal = () => {
    bindDebouncedSlider('settingsConfSlider', 'settingsConfValue', 'confidence');
    bindDebouncedSlider('settingsIouSlider', 'settingsIouValue', 'iou');
    bindDebouncedSlider('settingsLimitSlider', 'settingsLimitValue', 'speed_limit', (v) => String(Math.round(v)));
    $('settingsPreprocessToggle')?.addEventListener('change', (event) => {
      postJSON('/api/settings', { preprocess: event.target.checked });
    });
    $('settingsResetStats')?.addEventListener('click', async () => {
      await postJSON('/api/stats/reset', {});
      trend.speed = []; trend.total = []; trend.labels = [];
      showToast({ type: 'info', title: 'Statistik direset', message: 'Counter live telah dikosongkan.' });
      refreshLive();
    });
  };

  /* =================================================================== */
  /* Status polling: header pills, camera info, system status, overlay   */
  /* =================================================================== */
  let previousCctvStatus = null;
  let aiLoadedToastShown = false;
  let dbConnectedToastShown = false;

  const refreshStatus = async () => {
    const data = await api('/api/status');
    if (!data) return;
    const status = data.cctv_status || 'OFFLINE';

    // Header pills.
    applyDot('pillAiDot', data.model ? 'ONLINE' : 'OFFLINE');
    applyDot('pillDbDot', data.database ? 'ONLINE' : 'OFFLINE');
    applyDot('pillCctvDot', status);

    // Video card header + overlay bar.
    setText('videoStatusText', status);
    const videoPill = $('videoStatusPill');
    if (videoPill) videoPill.className = `status-pill ${badgeClass(status)}`;
    applyDot('videoStatusDot', status);
    applyDot('ovStatusDot', status);
    setText('ovName', data.camera_name || 'Tidak ada kamera');
    setText('ovStatus', status);
    setText('ovFps', data.fps !== undefined ? `${data.fps}` : '–');
    setText('ovModel', data.model || '–');
    setText('ovTracker', data.tracker || '–');
    setText('ovInference', data.inference_ms !== undefined ? `${data.inference_ms} ms` : '–');
    setText('ovResolution', data.resolution || '–');
    setText('ovStreamType', data.stream_type || '–');

    // Camera information card.
    setText('ciName', data.camera_name || '–');
    setText('ciStatus', status);
    setText('ciResolution', data.resolution || '–');
    setText('ciStreamType', data.stream_type || '–');
    setText('ciUrl', data.stream_url || '–');
    const reasonAlert = $('ciReasonAlert');
    if (reasonAlert) {
      const message = data.reason && status !== 'ONLINE' ? data.reason : '';
      reasonAlert.textContent = message;
      reasonAlert.classList.toggle('show', !!message);
    }

    // System status card.
    applyDot('ssStatusDot', status);
    setText('ssStatusText', status);
    setText('ssCamera', data.camera_name || '–');
    setText('ssResolution', data.resolution || '–');
    setText('ssFps', data.fps !== undefined ? `${data.fps} fps` : '–');
    setText('ssInference', data.inference_ms !== undefined ? `${data.inference_ms} ms` : '–');
    setText('ssTracker', data.tracker || '–');
    setText('ssModel', data.model || '–');
    setText('ssDatabase', data.database || '–');
    setText('ssStream', data.stream_url || '–');
    setText('ssLatency', data.fps > 0 ? `${Math.round(1000 / data.fps)} ms` : '–');
    setText('ssUptime', data.uptime !== undefined ? formatUptime(data.uptime) : '–');

    // KPI: resolution.
    setText('kpiResolution', data.resolution || '–');

    // Toasts on meaningful state transitions.
    if (data.model && !aiLoadedToastShown) {
      aiLoadedToastShown = true;
      showToast({ type: 'success', title: 'AI Loaded', message: `Model ${data.model} siap digunakan.` });
    }
    if (data.database && !dbConnectedToastShown) {
      dbConnectedToastShown = true;
      showToast({ type: 'success', title: 'Database Connected', message: 'Koneksi database berhasil.' });
    }
    if (previousCctvStatus !== null && previousCctvStatus !== status) {
      if (status === 'ONLINE') {
        showToast({ type: 'success', title: 'Camera Connected', message: data.camera_name || '' });
      } else if (previousCctvStatus === 'ONLINE') {
        showToast({ type: 'danger', title: 'Camera Lost', message: data.reason || status });
      }
    }
    previousCctvStatus = status;

    // Line-crossing diagnostics (only meaningful once ROI lines are set).
    const crossing = data.crossing;
    const info = $('crossingInfo');
    if (info && crossing) {
      if (crossing.lines_active) {
        const stuck = crossing.entry_crossings > 0 && crossing.measured_tracks === 0;
        info.classList.toggle('show', true);
        info.classList.toggle('info-alert', true);
        info.style.color = stuck ? '#fde68a' : '';
        info.textContent = stuck
          ? `Entry dilewati ${crossing.entry_crossings}× tapi belum ada yang mencapai EXIT — geser garis EXIT searah lintasan kendaraan.`
          : `Line gate aktif · entry ${crossing.entry_crossings} · exit ${crossing.exit_crossings} · terukur ${crossing.measured_tracks}`;
      } else {
        info.classList.remove('show');
      }
    }
  };

  const formatUptime = (seconds) => {
    seconds = Math.floor(seconds);
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = seconds % 60;
    return h > 0 ? `${h}j ${m}m` : m > 0 ? `${m}m ${s}d` : `${s}d`;
  };

  /* =================================================================== */
  /* Live counters, mini list, recent table, charts                      */
  /* =================================================================== */
  const trend = { labels: [], speed: [], total: [] };
  let lastOverspeedCount = null;

  const refreshLive = async () => {
    const data = await api('/api/live');
    if (!data) return;

    animateValue('kpiTotal', data.total || 0, 0);
    animateValue('kpiOverspeed', data.overspeed || 0, 0);
    animateValue('kpiAvgSpeed', data.average_speed || 0, 1);
    animateValue('kpiFps', data.fps || 0, 1);
    animateValue('kpiInference', data.inference_ms || 0, 0);

    if (lastOverspeedCount !== null && data.overspeed > lastOverspeedCount) {
      const worst = (data.detections || []).find((d) => d.status === 'Overspeed');
      showToast({
        type: 'danger', title: 'Overspeed Detected',
        message: worst ? `${worst.vehicle_type} #${worst.track_id} · ${worst.speed} km/j` : 'Kendaraan melebihi batas kecepatan.',
      });
    }
    lastOverspeedCount = data.overspeed || 0;

    renderLiveDetectionList(data.detections || []);
    updateCharts(data);

    const stamp = new Date().toLocaleTimeString('id-ID', { hour12: false });
    trend.labels.push(stamp);
    trend.speed.push(data.average_speed || 0);
    trend.total.push(data.total || 0);
    if (trend.labels.length > 30) { trend.labels.shift(); trend.speed.shift(); trend.total.shift(); }
    updateTrendCharts();
  };

  const renderLiveDetectionList = (rows) => {
    const container = $('liveDetectionList');
    setText('liveDetectionCount', rows.length);
    if (!container) return;
    container.innerHTML = rows.length
      ? rows.slice(0, 12).map((item) => `
          <div class="mini-item">
            <div class="mini-item-left">
              <span class="mini-item-id">#${item.track_id}</span>
              <span class="mini-item-type">${escapeHtml(item.vehicle_type)}</span>
            </div>
            <span class="mini-item-speed ${speedClass(item.status)}">${item.speed > 0 ? `${item.speed}` : '…'}</span>
          </div>`).join('')
      : '<div class="loading-panel" style="padding:20px;">Tidak ada kendaraan terdeteksi.</div>';
  };

  const refreshSavedTable = async () => {
    const data = await api('/api/detections?page=1&limit=12');
    const table = $('recentDetectionTable');
    if (!data || !table) return;
    table.innerHTML = data.items.length
      ? data.items.map((item) => `
          <tr>
            <td class="mono">${escapeHtml(item.timestamp || '-')}</td>
            <td class="mono">#${item.track_id ?? '-'}</td>
            <td>${escapeHtml(item.vehicle_type)}</td>
            <td class="${speedClass(item.status)}">${item.speed} km/j</td>
            <td><span class="badge ${item.status === 'Overspeed' ? 'badge-danger' : item.status === 'Measuring' ? 'badge-warning' : 'badge-success'}">${escapeHtml(item.status)}</span></td>
            <td class="muted">–</td>
            <td class="muted">–</td>
            <td>${escapeHtml(item.camera_name || '-')}</td>
            <td>${item.snapshot ? `<a href="/${escapeHtml(item.snapshot)}" target="_blank" rel="noopener">Lihat</a>` : '<span class="muted">-</span>'}</td>
          </tr>`).join('')
      : '<tr class="empty-row"><td colspan="9">Belum ada data.</td></tr>';
  };

  /* ------------------------------ ApexCharts --------------------------- */
  const CHART_BASE = {
    chart: { background: 'transparent', foreColor: '#8b9bb4', toolbar: { show: false }, animations: { easing: 'easeinout', speed: 350 } },
    tooltip: { theme: 'dark' },
    grid: { borderColor: '#334155', strokeDashArray: 4 },
  };
  const charts = {};

  const initCharts = () => {
    if (typeof ApexCharts === 'undefined') return;

    if ($('chartVehicleType')) {
      charts.vehicleType = new ApexCharts($('chartVehicleType'), {
        ...CHART_BASE,
        chart: { ...CHART_BASE.chart, type: 'donut', height: 230 },
        series: [0, 0, 0, 0],
        labels: ['Mobil', 'Motor', 'Bus', 'Truck'],
        colors: ['#3b82f6', '#22c55e', '#f59e0b', '#a855f7'],
        stroke: { width: 2, colors: ['#1e293b'] },
        legend: { position: 'bottom', labels: { colors: '#8b9bb4' } },
        dataLabels: { style: { colors: ['#0f172a'] } },
        plotOptions: { pie: { donut: { size: '62%', labels: { show: true, total: { show: true, label: 'Total', color: '#8b9bb4' } } } } },
      });
      charts.vehicleType.render();
    }

    if ($('chartSpeedTrend')) {
      charts.speedTrend = new ApexCharts($('chartSpeedTrend'), {
        ...CHART_BASE,
        chart: { ...CHART_BASE.chart, type: 'area', height: 230 },
        series: [{ name: 'Rata-rata (km/j)', data: [] }],
        xaxis: { categories: [], labels: { show: false }, axisTicks: { show: false }, axisBorder: { show: false } },
        yaxis: { labels: { style: { colors: '#8b9bb4' } } },
        colors: ['#3b82f6'],
        stroke: { curve: 'smooth', width: 2 },
        fill: { type: 'gradient', gradient: { shadeIntensity: 1, opacityFrom: 0.35, opacityTo: 0, stops: [0, 90, 100] } },
        dataLabels: { enabled: false },
      });
      charts.speedTrend.render();
    }

    if ($('chartTrafficDensity')) {
      charts.trafficDensity = new ApexCharts($('chartTrafficDensity'), {
        ...CHART_BASE,
        chart: { ...CHART_BASE.chart, type: 'line', height: 230 },
        series: [{ name: 'Total kendaraan', data: [] }],
        xaxis: { categories: [], labels: { show: false }, axisTicks: { show: false }, axisBorder: { show: false } },
        yaxis: { labels: { style: { colors: '#8b9bb4' } } },
        colors: ['#22c55e'],
        stroke: { curve: 'smooth', width: 2 },
        dataLabels: { enabled: false },
      });
      charts.trafficDensity.render();
    }

    if ($('chartOverspeedRatio')) {
      charts.overspeedRatio = new ApexCharts($('chartOverspeedRatio'), {
        ...CHART_BASE,
        chart: { ...CHART_BASE.chart, type: 'radialBar', height: 230 },
        series: [0],
        labels: ['Overspeed'],
        colors: ['#ef4444'],
        plotOptions: {
          radialBar: {
            hollow: { size: '60%' },
            track: { background: '#24334a' },
            dataLabels: {
              value: { color: '#f8fafc', fontSize: '26px', fontWeight: 800, formatter: (v) => `${v.toFixed(0)}%` },
              name: { color: '#8b9bb4', fontSize: '12px' },
            },
          },
        },
      });
      charts.overspeedRatio.render();
    }

    if ($('chartBusyHours')) {
      charts.busyHours = new ApexCharts($('chartBusyHours'), {
        ...CHART_BASE,
        chart: { ...CHART_BASE.chart, type: 'bar', height: 240 },
        series: [{ name: 'Kendaraan', data: new Array(24).fill(0) }],
        xaxis: {
          categories: Array.from({ length: 24 }, (_, h) => `${String(h).padStart(2, '0')}:00`),
          labels: { style: { colors: '#8b9bb4' } },
        },
        yaxis: { labels: { style: { colors: '#8b9bb4' } } },
        colors: ['#3b82f6'],
        plotOptions: { bar: { borderRadius: 4, columnWidth: '55%' } },
        dataLabels: { enabled: false },
      });
      charts.busyHours.render();
    }
  };

  const updateCharts = (live) => {
    charts.vehicleType?.updateSeries([live.car || 0, live.motorcycle || 0, live.bus || 0, live.truck || 0]);
    const overspeedPct = live.total > 0 ? (live.overspeed / live.total) * 100 : 0;
    charts.overspeedRatio?.updateSeries([Math.round(overspeedPct)]);
  };

  const updateTrendCharts = () => {
    charts.speedTrend?.updateOptions({ xaxis: { categories: trend.labels } }, false, true);
    charts.speedTrend?.updateSeries([{ name: 'Rata-rata (km/j)', data: trend.speed }]);
    charts.trafficDensity?.updateOptions({ xaxis: { categories: trend.labels } }, false, true);
    charts.trafficDensity?.updateSeries([{ name: 'Total kendaraan', data: trend.total }]);
  };

  const refreshBusyHours = async () => {
    if (!charts.busyHours) return;
    const data = await api('/api/detections?page=1&limit=200');
    if (!data || !data.items) return;
    const buckets = new Array(24).fill(0);
    data.items.forEach((item) => {
      if (!item.timestamp) return;
      const parsed = new Date(item.timestamp.replace(' ', 'T'));
      if (!Number.isNaN(parsed.getTime())) buckets[parsed.getHours()] += 1;
    });
    charts.busyHours.updateSeries([{ name: 'Kendaraan', data: buckets }]);
  };

  /* =================================================================== */
  /* Camera search panel                                                  */
  /* =================================================================== */
  let cameras = [];
  let selectedCameraUrl = null;

  const renderCameraList = () => {
    const container = $('cameraList');
    if (!container) return;
    const filter = ($('cameraSearchInput')?.value || '').toLowerCase().trim();
    const visible = cameras.filter((camera) =>
      !filter || camera.name.toLowerCase().includes(filter) || camera.url.toLowerCase().includes(filter));

    container.innerHTML = visible.length
      ? visible.map((camera) => `
          <div class="camera-item ${camera.url === selectedCameraUrl ? 'selected' : ''}" data-url="${escapeHtml(camera.url)}" title="${escapeHtml(camera.reason || '')}">
            <span class="status-dot ${dotClass(camera.status)}"></span>
            <div class="camera-item-body">
              <div class="camera-item-name">${escapeHtml(camera.name)}</div>
              <div class="camera-item-meta">${escapeHtml(camera.status)}<span class="sep">&middot;</span>${escapeHtml(camera.resolution || '-')}<span class="sep">&middot;</span>${escapeHtml(camera.stream_type)}</div>
            </div>
          </div>`).join('')
      : '<div class="loading-panel" style="padding:20px;">Tidak ada kamera yang cocok.</div>';

    container.querySelectorAll('.camera-item').forEach((el) => {
      el.addEventListener('click', () => selectCamera(el.dataset.url));
    });

    const online = cameras.filter((c) => c.status === 'ONLINE').length;
    setText('cameraCountBadge', cameras.length);
    setText('cameraListSummary', cameras.length
      ? `${cameras.length} kamera · ${online} online · menampilkan ${visible.length}`
      : 'Tidak ada kamera ditemukan.');
  };

  const refreshCameras = async () => {
    const data = await api('/api/cameras');
    if (!data) return;
    cameras = data.cameras || [];
    selectedCameraUrl = data.selected_url;
    renderCameraList();
  };

  const selectCamera = async (url) => {
    if (!url || url === selectedCameraUrl) return;
    setText('cameraListSummary', 'Mengganti kamera…');
    const result = await postJSON('/api/camera', { url });
    if (result && result.success) {
      selectedCameraUrl = result.selected_url;
      const video = $('liveVideo');
      if (video) video.src = `/stream?t=${Date.now()}`;
      renderCameraList();
    }
    await refreshStatus();
    await refreshCameras();
  };

  /* =================================================================== */
  /* ROI / entry / exit line editor                                       */
  /* =================================================================== */
  const roi = { polygon: [], entry: [], exit: [], mode: null };
  const canvas = $('roiCanvas');
  const context = canvas ? canvas.getContext('2d') : null;

  const resizeCanvas = () => {
    const video = $('liveVideo');
    if (!canvas || !video) return;
    const rect = video.getBoundingClientRect();
    if (rect.width && rect.height) {
      canvas.width = rect.width;
      canvas.height = rect.height;
      drawOverlay();
    }
  };

  const drawOverlay = () => {
    if (!context || !canvas) return;
    const { width, height } = canvas;
    context.clearRect(0, 0, width, height);
    const toPixels = ([x, y]) => [x * width, y * height];

    if (roi.polygon.length) {
      context.beginPath();
      roi.polygon.forEach((point, index) => {
        const [x, y] = toPixels(point);
        index === 0 ? context.moveTo(x, y) : context.lineTo(x, y);
      });
      if (roi.polygon.length > 2) context.closePath();
      context.fillStyle = 'rgba(34,197,94,0.14)';
      context.strokeStyle = '#22c55e';
      context.lineWidth = 2;
      if (roi.polygon.length > 2) context.fill();
      context.stroke();
      roi.polygon.forEach((point) => {
        const [x, y] = toPixels(point);
        context.beginPath(); context.arc(x, y, 4, 0, Math.PI * 2);
        context.fillStyle = '#22c55e'; context.fill();
      });
    }

    const drawLine = (points, colour, caption) => {
      if (!points.length) return;
      context.strokeStyle = colour; context.fillStyle = colour; context.lineWidth = 3;
      if (points.length === 2) {
        const [x1, y1] = toPixels(points[0]);
        const [x2, y2] = toPixels(points[1]);
        context.beginPath(); context.moveTo(x1, y1); context.lineTo(x2, y2); context.stroke();
        context.font = '600 12px Inter, sans-serif';
        context.fillText(caption, x1 + 6, y1 - 6);
      }
      points.forEach((point) => {
        const [x, y] = toPixels(point);
        context.beginPath(); context.arc(x, y, 4, 0, Math.PI * 2); context.fill();
      });
    };
    drawLine(roi.entry, '#3b82f6', 'ENTRY');
    drawLine(roi.exit, '#f59e0b', 'EXIT');
  };

  const setToolActive = (id, active) => $(id)?.classList.toggle('active', active);

  const setMode = (mode) => {
    roi.mode = roi.mode === mode ? null : mode;
    if (roi.mode === 'roi') roi.polygon = [];
    if (roi.mode === 'entry') roi.entry = [];
    if (roi.mode === 'exit') roi.exit = [];

    setToolActive('btnDrawRoi', roi.mode === 'roi');
    setToolActive('btnEntryLine', roi.mode === 'entry');
    setToolActive('btnExitLine', roi.mode === 'exit');
    canvas?.classList.toggle('drawing', roi.mode !== null);

    const hint = $('roiHint');
    if (hint) {
      const messages = {
        roi: 'Klik minimal 3 titik untuk area ROI, lalu klik <strong>Draw ROI</strong> lagi untuk menyimpan.',
        entry: 'Klik 2 titik pada video untuk menggambar <strong>ENTRY LINE</strong>.',
        exit: 'Klik 2 titik pada video untuk menggambar <strong>EXIT LINE</strong>.',
      };
      if (roi.mode) { hint.innerHTML = messages[roi.mode]; hint.classList.remove('hidden'); }
      else hint.classList.add('hidden');
    }
    if (roi.mode === null) saveRoi();
    drawOverlay();
  };

  const saveRoi = async () => {
    const payload = {
      polygon: roi.polygon.length >= 3 ? roi.polygon : [],
      entry_line: roi.entry.length === 2 ? roi.entry : [],
      exit_line: roi.exit.length === 2 ? roi.exit : [],
    };
    const result = await postJSON('/api/roi', payload);
    if (result) {
      roi.polygon = result.polygon || []; roi.entry = result.entry_line || []; roi.exit = result.exit_line || [];
      drawOverlay();
    }
  };

  const loadRoi = async () => {
    const data = await api('/api/roi');
    if (!data) return;
    roi.polygon = data.polygon || []; roi.entry = data.entry_line || []; roi.exit = data.exit_line || [];
    drawOverlay();
  };

  const onCanvasClick = (event) => {
    if (!roi.mode || !canvas) return;
    const rect = canvas.getBoundingClientRect();
    const point = [
      Math.min(Math.max((event.clientX - rect.left) / rect.width, 0), 1),
      Math.min(Math.max((event.clientY - rect.top) / rect.height, 0), 1),
    ];
    if (roi.mode === 'roi') roi.polygon.push(point);
    else if (roi.mode === 'entry') { roi.entry.push(point); if (roi.entry.length === 2) { setMode('entry'); return; } }
    else if (roi.mode === 'exit') { roi.exit.push(point); if (roi.exit.length === 2) { setMode('exit'); return; } }
    drawOverlay();
  };

  /* =================================================================== */
  /* Toolbar: capture / refresh stream / screenshot / fullscreen          */
  /* =================================================================== */
  const initToolbar = () => {
    $('btnDrawRoi')?.addEventListener('click', () => setMode('roi'));
    $('btnEntryLine')?.addEventListener('click', () => setMode('entry'));
    $('btnExitLine')?.addEventListener('click', () => setMode('exit'));
    $('btnResetRoi')?.addEventListener('click', async () => {
      roi.polygon = []; roi.entry = []; roi.exit = []; roi.mode = null;
      await deleteJSON('/api/roi');
      setMode(null);
      drawOverlay();
      showToast({ type: 'info', title: 'ROI direset' });
    });

    $('btnCapture')?.addEventListener('click', async () => {
      const result = await postJSON('/api/capture', {});
      if (result && result.snapshot) {
        showToast({ type: 'success', title: 'Snapshot tersimpan', message: result.snapshot });
        refreshSavedTable();
      } else {
        showToast({ type: 'warning', title: 'Capture gagal', message: 'Belum ada frame yang tersedia.' });
      }
    });

    $('btnRefreshStream')?.addEventListener('click', () => {
      const video = $('liveVideo');
      if (video) video.src = `/stream?t=${Date.now()}`;
      showToast({ type: 'info', title: 'Stream disegarkan' });
    });

    $('btnScreenshot')?.addEventListener('click', () => {
      const video = $('liveVideo');
      if (!video) return;
      try {
        const off = document.createElement('canvas');
        off.width = video.naturalWidth || video.width;
        off.height = video.naturalHeight || video.height;
        off.getContext('2d').drawImage(video, 0, 0, off.width, off.height);
        const link = document.createElement('a');
        link.download = `speed-tracker-${Date.now()}.png`;
        link.href = off.toDataURL('image/png');
        link.click();
        showToast({ type: 'success', title: 'Screenshot diunduh' });
      } catch (error) {
        showToast({ type: 'warning', title: 'Screenshot gagal', message: 'Coba lagi setelah stream aktif.' });
      }
    });

    $('btnFullscreen')?.addEventListener('click', () => {
      const shell = $('videoShell');
      if (!shell) return;
      if (!document.fullscreenElement) shell.requestFullscreen?.();
      else document.exitFullscreen?.();
    });
    document.addEventListener('fullscreenchange', () => {
      const icon = document.querySelector('#btnFullscreen i');
      if (icon) icon.setAttribute('data-lucide', document.fullscreenElement ? 'minimize' : 'maximize');
      if (window.lucide) window.lucide.createIcons();
    });
  };

  /* =================================================================== */
  /* History page                                                        */
  /* =================================================================== */
  const loadHistory = async (page = 1) => {
    const type = $('filterType')?.value || '';
    const status = $('filterStatus')?.value || '';
    const search = $('searchInput')?.value.trim() || '';
    const query = new URLSearchParams({ page, limit: '25' });
    if (type) query.set('vehicle_type', type);
    if (status) query.set('status', status);
    if (search) query.set('search', search);

    const data = await api(`/api/detections?${query}`);
    const table = $('historyTable');
    if (!data || !table) return;

    table.innerHTML = data.items.length
      ? data.items.map((item) => `
          <tr>
            <td class="mono">${item.id}</td>
            <td class="mono">${item.track_id ?? '-'}</td>
            <td>${escapeHtml(item.vehicle_type)}</td>
            <td class="${speedClass(item.status)}">${item.speed} km/j</td>
            <td><span class="badge ${item.status === 'Overspeed' ? 'badge-danger' : item.status === 'Measuring' ? 'badge-warning' : 'badge-success'}">${escapeHtml(item.status)}</span></td>
            <td>${escapeHtml(item.camera_name || '-')}</td>
            <td class="mono url-text">${escapeHtml(item.stream_url || '-')}</td>
            <td class="mono">${escapeHtml(item.timestamp || '-')}</td>
            <td>${item.snapshot ? `<a href="/${escapeHtml(item.snapshot)}" target="_blank" rel="noopener">Lihat</a>` : '<span class="muted">-</span>'}</td>
          </tr>`).join('')
      : '<tr class="empty-row"><td colspan="9">Belum ada data.</td></tr>';

    setText('historySummary', `${data.total} data · halaman ${data.page} dari ${data.pages || 1}`);

    const pagination = $('pagination');
    if (pagination) {
      const pages = data.pages || 1;
      const from = Math.max(1, data.page - 2);
      const to = Math.min(pages, from + 4);
      let html = '';
      for (let i = from; i <= to; i += 1) {
        html += `<button class="page-btn ${i === data.page ? 'active' : ''}" data-page="${i}">${i}</button>`;
      }
      pagination.innerHTML = html;
      pagination.querySelectorAll('button[data-page]').forEach((button) => {
        button.addEventListener('click', () => loadHistory(Number(button.dataset.page)));
      });
    }
  };

  /* =================================================================== */
  /* Bootstrap                                                            */
  /* =================================================================== */
  if (window.lucide) window.lucide.createIcons();

  initChrome();

  if (isDashboard) {
    initCharts();
    initSettingsModal();
    initToolbar();

    tickClock();
    setInterval(tickClock, 1000);

    refreshStatus();
    refreshLive();
    refreshCameras();
    refreshSavedTable();
    loadRoi();
    refreshBusyHours();
    resizeCanvas();

    setInterval(refreshLive, 1000);
    setInterval(refreshStatus, 3000);
    setInterval(refreshSavedTable, 5000);
    setInterval(refreshCameras, 30000);
    setInterval(refreshBusyHours, 60000);

    window.addEventListener('resize', resizeCanvas);
    $('liveVideo')?.addEventListener('load', resizeCanvas);
    $('cameraSearchInput')?.addEventListener('input', renderCameraList);
    $('btnRefreshCameras')?.addEventListener('click', async () => {
      setText('cameraListSummary', 'Memindai portal CCTV…');
      await postJSON('/api/cameras/refresh', {});
      setTimeout(refreshCameras, 4000);
    });
    canvas?.addEventListener('click', onCanvasClick);
  }

  if (isHistory) {
    tickClock();
    setInterval(tickClock, 1000);
    refreshStatus();
    setInterval(refreshStatus, 5000);

    loadHistory(1);
    let timer = null;
    const reload = () => { clearTimeout(timer); timer = setTimeout(() => loadHistory(1), 300); };
    $('filterType')?.addEventListener('change', reload);
    $('filterStatus')?.addEventListener('change', reload);
    $('searchInput')?.addEventListener('input', reload);
  }
})();
