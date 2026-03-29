// Video Analytics Dashboard — SSE client for detection stats

const statusEl = document.getElementById('status');
const fpsEl = document.getElementById('fps');
const latencyEl = document.getElementById('latency');
const detCountEl = document.getElementById('det-count');
const frameIdEl = document.getElementById('frame-id');
const detListEl = document.getElementById('det-list');

// Update stats display from SSE data
function updateStats(data) {
  fpsEl.textContent = data.fps?.toFixed(1) ?? '--';
  latencyEl.textContent = data.latency_ms?.toFixed(0) ?? '--';
  detCountEl.textContent = data.detections?.length ?? 0;
  frameIdEl.textContent = data.frame_id ?? '--';

  // Detection list chips
  detListEl.innerHTML = (data.detections || [])
    .map(function(d) {
      return '<div class="det-item">' + d.class + ' ' +
             (d.confidence * 100).toFixed(0) + '%</div>';
    })
    .join('');
}

// Connect to SSE stream with auto-reconnect
function connectSSE() {
  var source = new EventSource('/events');

  source.onopen = function() {
    statusEl.textContent = 'Connected';
    statusEl.className = 'status connected';
  };

  source.onmessage = function(event) {
    try {
      var data = JSON.parse(event.data);
      updateStats(data);
    } catch (e) {
      console.warn('Failed to parse SSE data:', e);
    }
  };

  source.onerror = function() {
    statusEl.textContent = 'Disconnected';
    statusEl.className = 'status disconnected';
    source.close();
    setTimeout(connectSSE, 2000);
  };
}

// Initialize — MJPEG stream loads automatically via <img src="/video">
connectSSE();
