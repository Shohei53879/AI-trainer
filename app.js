import { SquatAnalyzer, ISSUE_TEXT } from './squat.js';

// ---------- 設定 ----------
const MP_VER = '0.10.14';
const MP_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VER}`;
const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';
const AUTO_END_SEC = 8; // 最後のレップからこの秒数、立ったままならセット終了

const store = {
  get(k, d) { try { return localStorage.getItem(k) ?? d; } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch {} },
};
const settings = {
  apiKey: store.get('tc.apiKey', ''),
  model: store.get('tc.model', 'gemini-2.5-flash'),
  depth: Number(store.get('tc.depth', '100')),
  facing: store.get('tc.facing', 'user'),
};

// ---------- 画面 ----------
const $ = (id) => document.getElementById(id);
const video = $('video'), canvas = $('overlay'), ctx = canvas.getContext('2d');
const ui = {
  status: $('status'), reps: $('reps'), knee: $('knee'), lean: $('lean'),
  log: $('log'), start: $('startBtn'), set: $('setBtn'), talk: $('talkBtn'),
  hands: $('handsFree'), flip: $('flipBtn'), settingsBtn: $('settingsBtn'),
  dialog: $('settings'), apiKey: $('apiKey'), model: $('model'), depth: $('depth'),
  save: $('saveSettings'), stage: $('stage'),
};

function log(who, text) {
  const div = document.createElement('div');
  div.className = `msg ${who}`;
  div.textContent = text;
  ui.log.prepend(div);
}
function setStatus(t) { ui.status.textContent = t; }

// ---------- 音声出力 ----------
let speaking = false;
const synth = window.speechSynthesis;
function pickVoice() {
  return synth?.getVoices().find((v) => v.lang?.startsWith('ja')) ?? null;
}
function speak(text, { interrupt = false } = {}) {
  if (!synth || !text) return;
  if (interrupt) synth.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.lang = 'ja-JP';
  u.rate = 1.1;
  const v = pickVoice();
  if (v) u.voice = v;
  u.onstart = () => { speaking = true; pauseListening(); };
  u.onend = u.onerror = () => {
    if (!synth.speaking) { speaking = false; resumeListening(); }
  };
  synth.speak(u);
}

// ---------- 音声入力 ----------
const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
let rec = null, listening = false, wantListen = false;

function initRecognition() {
  if (!SR) { ui.talk.disabled = true; ui.hands.disabled = true; log('sys', 'このブラウザは音声認識に対応していません'); return; }
  rec = new SR();
  rec.lang = 'ja-JP';
  rec.interimResults = false;
  rec.continuous = true;
  rec.onresult = (e) => {
    const r = e.results[e.results.length - 1];
    if (r.isFinal) handleUtterance(r[0].transcript.trim());
  };
  rec.onstart = () => { listening = true; ui.talk.classList.add('on'); };
  rec.onend = () => {
    listening = false; ui.talk.classList.remove('on');
    if (wantListen && !speaking) setTimeout(startListening, 300); // iOS は無音で止まるので再開
  };
  rec.onerror = (e) => { if (e.error === 'not-allowed') { wantListen = false; log('sys', 'マイクが許可されていません'); } };
}
function startListening() {
  if (!rec || listening || speaking) return;
  try { rec.start(); } catch {}
}
function pauseListening() { if (rec && listening) try { rec.abort(); } catch {} }
function resumeListening() { if (wantListen) setTimeout(startListening, 300); }

// ---------- Gemini ----------
const SYSTEM_PROMPT = `あなたは日本語で話すパーソナルトレーナーです。
返答はスマホの音声読み上げで流れるので、話し言葉で2〜3文、100文字程度に収めてください。記号や箇条書きは使わないでください。
フォームの判定はアプリが関節角度から計測した数値で行っています。渡されたデータにないことを推測で断定しないでください。
セット後のフィードバックでは、まず良かった点を一つ、次に最も優先して直すべき点を一つだけ、具体的な体の動かし方で伝えてください。
膝・腰の痛みなど体の不調を訴えられたら、無理に続けず中止して専門家に相談するよう勧めてください。`;

const history = []; // {role, parts}
const sessionSets = []; // 各セットの要約

async function askGemini(userText) {
  if (!settings.apiKey) {
    const msg = '設定からGeminiのAPIキーを入力してください';
    log('sys', msg); speak(msg); return;
  }
  const context = sessionSets.length
    ? `【今日の記録】\n${sessionSets.map((s, i) => `第${i + 1}セット: ${s}`).join('\n')}`
    : '【今日の記録】まだセットはありません';
  const live = analyzer.reps.length ? `\n【進行中のセット】\n${analyzer.summary()}` : '';
  const contents = [
    ...history.slice(-8),
    { role: 'user', parts: [{ text: `${context}${live}\n\n${userText}` }] },
  ];
  const generationConfig = { temperature: 0.7, maxOutputTokens: 400 };
  if (settings.model.includes('2.5-flash')) generationConfig.thinkingConfig = { thinkingBudget: 0 };

  setStatus('コーチが考え中…');
  try {
    const res = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(settings.model)}:generateContent`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'x-goog-api-key': settings.apiKey },
        body: JSON.stringify({ systemInstruction: { parts: [{ text: SYSTEM_PROMPT }] }, contents, generationConfig }),
      }
    );
    const data = await res.json();
    if (!res.ok) throw new Error(data.error?.message || res.status);
    const text = data.candidates?.[0]?.content?.parts?.map((p) => p.text || '').join('').trim();
    if (!text) throw new Error('空の応答');
    history.push({ role: 'user', parts: [{ text: userText }] }, { role: 'model', parts: [{ text }] });
    log('coach', text);
    speak(text);
  } catch (e) {
    log('sys', `Gemini エラー: ${e.message}`);
    speak('コーチにつながりませんでした');
  } finally {
    setStatus(inSet ? 'セット中' : '待機中');
  }
}

// ---------- セット管理 ----------
const analyzer = new SquatAnalyzer({ depthAngle: settings.depth });
let inSet = false, lastRepTime = 0;

function startSet() {
  if (inSet) return;
  analyzer.reset();
  inSet = true; lastRepTime = performance.now() / 1000;
  ui.set.textContent = 'セット終了'; ui.reps.textContent = '0';
  setStatus('セット中');
  speak('スタート。横向きで始めてください', { interrupt: true });
  log('sys', 'セット開始');
}
function endSet() {
  if (!inSet) return;
  inSet = false;
  ui.set.textContent = 'セット開始';
  setStatus('待機中');
  if (!analyzer.reps.length) { log('sys', 'レップがなかったのでセットを取り消しました'); return; }
  const summary = analyzer.summary();
  sessionSets.push(summary);
  log('sys', summary);
  askGemini('今のセットが終わりました。フィードバックをください。');
}

function onRep(rep) {
  lastRepTime = performance.now() / 1000;
  ui.reps.textContent = String(rep.n);
  // 数を数え、問題があれば一言だけ(前傾はレップ中に注意済み)
  const issue = rep.issues.find((k) => k !== 'lean');
  speak(issue ? `${rep.n}、${ISSUE_TEXT[issue]}` : `${rep.n}`, { interrupt: true });
}

// ---------- 音声コマンド ----------
function handleUtterance(text) {
  if (!text) return;
  log('user', text);
  if (/(セット)?(開始|スタート|始め)/.test(text) && !inSet) return startSet();
  if (/(セット)?(終了|終わり|ストップ|止め)/.test(text) && inSet) return endSet();
  if (/何回/.test(text) && inSet) return speak(`${analyzer.reps.length}回です`);
  askGemini(text);
}

// ---------- カメラと姿勢推定 ----------
let landmarker = null, drawer = null, PoseLandmarker = null, stream = null;

async function loadVision() {
  let mod;
  try { mod = await import(`${MP_BASE}/vision_bundle.mjs`); }
  catch { mod = await import(`${MP_BASE}/+esm`); }
  return mod;
}

async function initPose() {
  setStatus('骨格モデルを読み込み中…');
  const vision = await loadVision();
  PoseLandmarker = vision.PoseLandmarker;
  const fileset = await vision.FilesetResolver.forVisionTasks(`${MP_BASE}/wasm`);
  const make = (delegate) => PoseLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: MODEL_URL, delegate },
    runningMode: 'VIDEO', numPoses: 1,
  });
  try { landmarker = await make('GPU'); } catch { landmarker = await make('CPU'); }
  drawer = new vision.DrawingUtils(ctx);
}

async function startCamera() {
  stream?.getTracks().forEach((t) => t.stop());
  stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: settings.facing, width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
  ui.stage.classList.toggle('mirror', settings.facing === 'user');
}

let lastVideoTime = -1;
function loop() {
  requestAnimationFrame(loop);
  if (!landmarker || video.readyState < 2 || video.currentTime === lastVideoTime) return;
  lastVideoTime = video.currentTime;
  const w = video.videoWidth, h = video.videoHeight;
  if (canvas.width !== w) { canvas.width = w; canvas.height = h; }

  const now = performance.now();
  const result = landmarker.detectForVideo(video, now);
  const lm = result.landmarks?.[0];
  ctx.clearRect(0, 0, w, h);
  if (lm) {
    drawer.drawConnectors(lm, PoseLandmarker.POSE_CONNECTIONS, { color: '#7CFFB2', lineWidth: 4 });
    drawer.drawLandmarks(lm, { color: '#ffffff', radius: 3 });
  }

  const t = now / 1000;
  const out = analyzer.update(lm, w, h, t);
  if (out.visible) {
    ui.knee.textContent = `${Math.round(out.knee)}°`;
    ui.lean.textContent = `${Math.round(out.lean)}°`;
  } else {
    ui.knee.textContent = ui.lean.textContent = '—';
  }
  if (!inSet) return;
  for (const ev of out.events) {
    if (ev.type === 'rep') onRep(ev.rep);
    if (ev.type === 'cue') speak(ev.text, { interrupt: true });
  }
  if (analyzer.reps.length && out.state === 'up' && t - lastRepTime > AUTO_END_SEC) endSet();
}

// ---------- 画面の操作 ----------
async function start() {
  ui.start.disabled = true;
  speak(' '); // iOS では最初の読み上げをタップ操作の中で行う必要がある
  try {
    await startCamera();
    await initPose();
    try { await navigator.wakeLock?.request('screen'); } catch {}
    ui.start.hidden = true;
    ui.set.disabled = ui.talk.disabled = false;
    initRecognition();
    if (ui.hands.checked) { wantListen = true; startListening(); }
    setStatus('待機中');
    log('sys', 'スマホを体の真横 2〜3m に置き、全身が映るようにしてください。「セット開始」と話しかけるか、ボタンを押すと始まります。');
    requestAnimationFrame(loop);
  } catch (e) {
    ui.start.disabled = false;
    setStatus('起動に失敗しました');
    log('sys', `エラー: ${e.message}`);
  }
}

ui.start.onclick = start;
ui.set.onclick = () => (inSet ? endSet() : startSet());
ui.talk.onclick = () => {
  // ハンズフリーでなければ1回だけ聞き取る
  if (listening) { wantListen = false; pauseListening(); return; }
  wantListen = ui.hands.checked; startListening();
  if (!ui.hands.checked) rec?.addEventListener('result', () => pauseListening(), { once: true });
};
ui.hands.onchange = () => { wantListen = ui.hands.checked; wantListen ? startListening() : pauseListening(); };
ui.flip.onclick = async () => {
  settings.facing = settings.facing === 'user' ? 'environment' : 'user';
  store.set('tc.facing', settings.facing);
  if (stream) await startCamera();
};
ui.settingsBtn.onclick = () => {
  ui.apiKey.value = settings.apiKey; ui.model.value = settings.model; ui.depth.value = settings.depth;
  ui.dialog.showModal();
};
ui.save.onclick = () => {
  settings.apiKey = ui.apiKey.value.trim();
  settings.model = ui.model.value.trim() || 'gemini-2.5-flash';
  settings.depth = Number(ui.depth.value) || 100;
  analyzer.opt.depthAngle = settings.depth;
  store.set('tc.apiKey', settings.apiKey); store.set('tc.model', settings.model); store.set('tc.depth', String(settings.depth));
};
if (!settings.apiKey) log('sys', '右上の⚙からGeminiのAPIキーを設定してください');
