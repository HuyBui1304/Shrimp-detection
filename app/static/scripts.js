// ===================== IMAGE =====================
const fileInput = document.getElementById('file');
const preview   = document.getElementById('preview');
const resultImg = document.getElementById('result');
const boxesPre  = document.getElementById('boxes');

fileInput.addEventListener('change', () => {
  const f = fileInput.files[0];
  if (f) preview.src = URL.createObjectURL(f);
});

document.getElementById('run').onclick = async () => {
  const f = fileInput.files[0];
  if (!f) return alert('Select an image first.');

  const fd = new FormData();
  fd.append('file', f);
  fd.append('imgsz', document.getElementById('imgsz').value);
  fd.append('conf',  document.getElementById('conf').value);
  fd.append('save_annotated', 'true');

  try {
    const res = await fetch('/predict', { method: 'POST', body: fd });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    resultImg.src = data.image || "";
    boxesPre.textContent = JSON.stringify(data.boxes, null, 2);
  } catch (err) {
    alert("Error: " + err.message);
  }
};

// ===================== VIDEO =====================
const vidInput  = document.getElementById('vidfile');
const runBtn    = document.getElementById('runVideo');
const stopBtn   = document.getElementById('stopVideo');
const origVideo = document.getElementById('origVideo');
const annotImg  = document.getElementById('annotStream');
const logSummary= document.getElementById('logSummary');

let currentVidMeta = null;

// Keep the action available after page restoration.
if (runBtn) runBtn.disabled = false;

if (runBtn) runBtn.onclick = async () => {
  const f = vidInput?.files?.[0];
  if (!f) return alert('Select a video first.');

  const imgsz  = document.getElementById('v_imgsz')?.value || '960';
  const conf   = document.getElementById('v_conf')?.value  || '0.25';
  const stride = document.getElementById('v_stride')?.value|| '2';

  runBtn.disabled = true;
  if (stopBtn) stopBtn.disabled = false;
  if (logSummary) logSummary.textContent = 'Uploading and starting the stream...';

  try {
    // Upload video
    const fd = new FormData();
    fd.append('file', f);
    const up = await fetch('/upload_video', { method: 'POST', body: fd });
    if (!up.ok) throw new Error(await up.text());
    const meta = await up.json(); // { id, orig_url, annot_url }
    currentVidMeta = meta;

    // Inference parameters
    const qs = new URLSearchParams({ conf, imgsz, stride }).toString();

    // Playback sources
    if (origVideo) origVideo.src = meta.orig_url;
    if (annotImg)  annotImg.src  = `${meta.annot_url}?${qs}`;

    // Play the original video.
    if (origVideo) {
      origVideo.onloadedmetadata = () => origVideo.play().catch(()=>{});
    }

    if (logSummary) logSummary.textContent = 'Streaming original and annotated video.';
  } catch (e) {
    alert('Video error: ' + e.message);
    if (logSummary) logSummary.textContent = 'Video processing failed.';
  } finally {
    runBtn.disabled = false;
  }
};

if (stopBtn) stopBtn.onclick = () => {
  if (annotImg) annotImg.src = '';
  try { if (origVideo) origVideo.pause(); } catch(_) {}
  if (logSummary) logSummary.textContent = 'Stopped.';
};

// ====== Sync tua ======
if (origVideo) {
  origVideo.addEventListener('seeked', () => {
    if (!currentVidMeta) return;
    const imgsz  = document.getElementById('v_imgsz')?.value || '960';
    const conf   = document.getElementById('v_conf')?.value  || '0.25';
    const stride = document.getElementById('v_stride')?.value|| '2';

    const qs = new URLSearchParams({
      conf, imgsz, stride,
      start: Math.floor(origVideo.currentTime)
    }).toString();

    annotImg.src = `${currentVidMeta.annot_url}?${qs}`;
  });
}

// ===================== LIVE CAMERA =====================
const camStartBtn = document.getElementById('camStart');
const camStopBtn  = document.getElementById('camStop');
const camVideo    = document.getElementById('camVideo');
const camOverlay  = document.getElementById('camOverlay');
const camCtx      = camOverlay.getContext('2d');
const camInfo     = document.getElementById('camInfo');

let camStream = null;
let camTimer  = null;
let camBusy   = false;

function camResizeOverlay() {
  const rect = camVideo.getBoundingClientRect();
  if (rect.width === 0 || rect.height === 0) return;
  camOverlay.width  = Math.floor(rect.width);
  camOverlay.height = Math.floor(rect.height);
}

function camDrawBoxes(boxes) {
  camCtx.clearRect(0, 0, camOverlay.width, camOverlay.height);
  const vw = camVideo.videoWidth  || camOverlay.width;
  const vh = camVideo.videoHeight || camOverlay.height;
  const sx = camOverlay.width  / vw;
  const sy = camOverlay.height / vh;

  camCtx.lineWidth = Math.max(1, camOverlay.width / 400);
  camCtx.font = `${Math.max(10, camOverlay.width / 48)}px ui-monospace, monospace`;
  camCtx.textBaseline = 'top';

  boxes.forEach(b => {
    const x = b.x1 * sx, y = b.y1 * sy, w = (b.x2 - b.x1) * sx, h = (b.y2 - b.y1) * sy;
    camCtx.strokeStyle = 'rgba(0,255,0,0.95)';
    camCtx.strokeRect(x, y, w, h);

    const label = `${b.cls_name ?? b.cls} ${b.conf.toFixed(2)}`;
    const pad = 2;
    const tw = camCtx.measureText(label).width + pad * 2;
    const th = parseInt(camCtx.font, 10) + pad * 2;

    camCtx.fillStyle = 'rgba(0,0,0,0.65)';
    camCtx.fillRect(x, Math.max(0, y - th), tw, th);
    camCtx.fillStyle = '#fff';
    camCtx.fillText(label, x + pad, Math.max(0, y - th) + pad);
  });
}

async function camTick() {
  if (camBusy) return;
  camBusy = true;
  try {
    const tmp = document.createElement('canvas');
    tmp.width  = camVideo.videoWidth  || 640;
    tmp.height = camVideo.videoHeight || 480;
    const tctx = tmp.getContext('2d');
    tctx.drawImage(camVideo, 0, 0, tmp.width, tmp.height);

    const blob = await new Promise(res => tmp.toBlob(res, 'image/jpeg', 0.85));
    if (!blob) return;

    const fd = new FormData();
    fd.append('file', blob, 'frame.jpg');
    fd.append('imgsz', document.getElementById('cam_imgsz').value || '640');
    fd.append('conf',  document.getElementById('cam_conf').value  || '0.25');
    fd.append('save_annotated', 'false');

    const resp = await fetch('/predict', { method: 'POST', body: fd });
    if (!resp.ok) throw new Error(await resp.text());
    const data = await resp.json();

    camInfo.textContent = `Boxes: ${data.boxes.length}`;
    camDrawBoxes(data.boxes || []);
  } catch (e) {
  camInfo.textContent = 'Camera inference failed.';
    console.error(e);
  } finally {
    camBusy = false;
  }
}

camStartBtn.onclick = async () => {
  if (camStream) return;
  try {
    camStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment' }, audio: false
    });
    camVideo.srcObject = camStream;
    await camVideo.play();
    camResizeOverlay();
    window.addEventListener('resize', camResizeOverlay);

    const intervalMs = 300;
    camTimer = setInterval(camTick, intervalMs);

    camStartBtn.disabled = true;
    camStopBtn.disabled  = false;
    camInfo.textContent  = 'Scanning...';
  } catch (e) {
    alert('Could not open the camera: ' + e.message);
  }
};

camStopBtn.onclick = () => {
  if (camTimer) clearInterval(camTimer);
  camTimer = null;
  camBusy = false;
  if (camStream) {
    camStream.getTracks().forEach(t => t.stop());
    camStream = null;
  }
  camVideo.srcObject = null;
  camCtx.clearRect(0, 0, camOverlay.width, camOverlay.height);

  camStartBtn.disabled = false;
  camStopBtn.disabled  = true;
  camInfo.textContent  = 'Stopped.';
};
