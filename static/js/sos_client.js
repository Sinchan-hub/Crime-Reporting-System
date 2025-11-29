// static/js/sos_client.js
document.addEventListener('DOMContentLoaded', function() {
  const sosBtn = document.getElementById('sosBtn');
  const overlay = document.getElementById('sos-overlay');
  const cancelBtn = document.getElementById('sos-cancel');
  const confirmBtn = document.getElementById('sos-confirm');
  const countEl = document.getElementById('sos-count');
  const timerEl = document.getElementById('sos-timer');
  const audio = document.getElementById('sos-audio');

  let counter = 3;
  let timer = null;

  function openSOS() {
    counter = 3;
    countEl.textContent = counter;
    timerEl.textContent = counter;
    overlay.style.display = 'flex';
    if (audio) { audio.currentTime = 0; audio.play().catch(()=>{}); }
    timer = setInterval(() => {
      counter--;
      countEl.textContent = counter;
      timerEl.textContent = counter;
      if (counter <= 0) {
        clearInterval(timer);
        doSendSOS();
      }
    }, 1000);
  }

  function closeSOS() {
    overlay.style.display = 'none';
    if (audio) try { audio.pause(); audio.currentTime = 0; } catch(e){}
    if (timer) clearInterval(timer);
  }

  async function doSendSOS() {
    if (timer) clearInterval(timer);
    if (audio) try { audio.pause(); audio.currentTime = 0; } catch(e){}
    try {
      const pos = await new Promise((resolve, reject) => {
        if (!navigator.geolocation) return reject('no geo');
        navigator.geolocation.getCurrentPosition(resolve, reject, {timeout:8000});
      });
      const lat = pos.coords.latitude;
      const lon = pos.coords.longitude;
      await fetch('/sos', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({lat, lon})
      });
      closeSOS();
      alert('SOS sent. Help is on the way.');
    } catch (err) {
      await fetch('/sos', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({}) });
      closeSOS();
      alert('SOS sent without location.');
    }
  }

  sosBtn && sosBtn.addEventListener('click', openSOS);
  cancelBtn && cancelBtn.addEventListener('click', closeSOS);
  confirmBtn && confirmBtn.addEventListener('click', doSendSOS);
});
