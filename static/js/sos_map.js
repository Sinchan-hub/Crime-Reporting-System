// static/js/sos_map.js
document.addEventListener('DOMContentLoaded', function() {
  if (!document.getElementById('sos-live-map')) return;
  const map = L.map('sos-live-map').setView([12.97,77.59], 12);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; OpenStreetMap contributors'
  }).addTo(map);

  const cluster = L.markerClusterGroup();
  map.addLayer(cluster);

  async function fetchAndPlot() {
    try {
      const res = await fetch('/sos/active');
      const list = await res.json();
      cluster.clearLayers();
      list.forEach(it => {
        if (it.lat && it.lon) {
          const mk = L.marker([it.lat, it.lon]);
          mk.bindPopup(`<b>SOS</b><br>ID: ${it.id}<br>Time: ${it.created_at}<br>
            <button onclick="ack('${it.id}')">Acknowledge</button>`);
          cluster.addLayer(mk);
        }
      });
    } catch(e) {
      console.error('failed fetch sos', e);
    }
  }

  window.ack = async function(sid) {
    try {
      await fetch(`/sos/ack/${sid}`, { method: 'POST' });
      fetchAndPlot();
      alert('Acknowledged');
    } catch(e){ alert('Ack failed') }
  };

  fetchAndPlot();
  setInterval(fetchAndPlot, 7000);
});
