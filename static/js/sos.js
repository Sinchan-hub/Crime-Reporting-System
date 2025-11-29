// ---------------------
// SOS FEATURE JS
// ---------------------

let sosTimer = null;
let sosCount = 3;

// Open SOS modal
function openSOS() {
    sosCount = 3;
    document.getElementById("sos-count").textContent = sosCount;
    document.getElementById("sos-timer").textContent = sosCount;
    document.getElementById("sos-overlay").style.display = "flex";

    sosTimer = setInterval(() => {
        sosCount--;
        document.getElementById("sos-count").textContent = sosCount;
        document.getElementById("sos-timer").textContent = sosCount;
        if (sosCount <= 0) {
            clearInterval(sosTimer);
            sendSOS();
        }
    }, 1000);
}

// Close SOS modal
function closeSOS() {
    document.getElementById("sos-overlay").style.display = "none";
    if (sosTimer) clearInterval(sosTimer);
}

// Send SOS (with GPS)
function sendSOS() {
    if (sosTimer) clearInterval(sosTimer);

    if (!navigator.geolocation) {
        alert("GPS unavailable. Sending SOS without location.");
        window.location.href = "/sos";
        return;
    }

    navigator.geolocation.getCurrentPosition(
        pos => {
            const lat = pos.coords.latitude;
            const lon = pos.coords.longitude;

            window.location.href = `/sos?lat=${lat}&lon=${lon}`;
        },
        err => {
            alert("GPS failed. Sending SOS without location.");
            window.location.href = "/sos";
        },
        { timeout: 6000 }
    );
}

// Attach HTML buttons
document.addEventListener("DOMContentLoaded", () => {
    const sosBtn = document.getElementById("sosBtn");
    const cancelBtn = document.getElementById("sos-cancel");
    const confirmBtn = document.getElementById("sos-confirm");

    if (sosBtn) sosBtn.onclick = openSOS;
    if (cancelBtn) cancelBtn.onclick = closeSOS;
    if (confirmBtn) confirmBtn.onclick = sendSOS;
});
