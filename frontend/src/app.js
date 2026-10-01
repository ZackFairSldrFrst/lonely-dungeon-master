// Lonely Dungeon Master: projector display.
// World units are projector pixels, origin top-left, y down — the same space the
// tracker reports in, so marker positions drop straight onto the table.
import * as THREE from 'three';

const params = new URLSearchParams(location.search);
const WS_URL = params.get('ws') || `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`;
const MAX_PLAYERS = 16;

let W = 1920, H = 1080, CELL = 40, FOG_RADIUS_CELLS = 6;

// ------------------------------------------------------------------ renderer
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setClearColor(0x000000, 1);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.OrthographicCamera(0, W, 0, H, -100, 100); // top=0, bottom=H -> y grows downward
const clock = new THREE.Clock();

function resize() {
  renderer.setSize(window.innerWidth, window.innerHeight);
}
window.addEventListener('resize', resize);
resize();

function setCanvas(width, height) {
  W = width; H = height;
  camera.left = 0; camera.right = W; camera.top = 0; camera.bottom = H;
  camera.updateProjectionMatrix();
  fog.mesh.scale.set(W, H, 1);
  fog.mesh.position.set(W / 2, H / 2, 0);
}

// Layer order (back to front). Monsters hide in the fog; DM spells and players shine through it.
const LAYER = { grid: 0, monsters: 1, fog: 2, effects: 3, players: 4 };

const planeGeo = new THREE.PlaneGeometry(1, 1);
const vertexShader = /* glsl */`
  varying vec2 vUv;
  varying vec2 vWorld;
  void main() {
    vUv = uv;
    vec4 world = modelMatrix * vec4(position, 1.0);
    vWorld = world.xy;
    gl_Position = projectionMatrix * viewMatrix * world;
  }`;

function shaderMaterial(fragmentShader, uniforms, additive = true) {
  return new THREE.ShaderMaterial({
    vertexShader, fragmentShader,
    uniforms: { uTime: { value: 0 }, uAlpha: { value: 0 }, ...uniforms },
    transparent: true, depthTest: false, depthWrite: false, side: THREE.DoubleSide,
    blending: additive ? THREE.AdditiveBlending : THREE.NormalBlending,
  });
}

// ------------------------------------------------------------------ grid
let grid = null;
function buildGrid() {
  if (grid) { scene.remove(grid); grid.geometry.dispose(); }
  const pts = [];
  for (let x = 0; x <= W; x += CELL) pts.push(x, 0, 0, x, H, 0);
  for (let y = 0; y <= H; y += CELL) pts.push(0, y, 0, W, y, 0);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
  grid = new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color: 0x3a4a66, transparent: true, opacity: 0.55, depthTest: false }));
  grid.renderOrder = LAYER.grid;
  scene.add(grid);
}

// ------------------------------------------------------------------ player ring (blue glow)
const PLAYER_FRAG = /* glsl */`
  uniform float uTime; uniform float uAlpha; uniform vec3 uColor;
  varying vec2 vUv;
  void main() {
    vec2 p = vUv - 0.5;
    float d = length(p) * 2.0;                       // 0 centre .. 1 edge
    float pulse = 0.5 + 0.5 * sin(uTime * 2.2);
    float core = exp(-pow((d - 0.62) / 0.035, 2.0));
    float glow = exp(-pow((d - 0.62) / (0.16 + 0.04 * pulse), 2.0)) * 0.55;
    float a = atan(p.y, p.x) + uTime * 0.6;
    float dashes = smoothstep(0.55, 0.9, sin(a * 3.0)) * exp(-pow((d - 0.86) / 0.03, 2.0)) * 0.8;
    float inner = smoothstep(0.62, 0.0, d) * 0.12;
    float v = (core + glow + dashes + inner) * uAlpha;
    gl_FragColor = vec4(uColor * v, v);
  }`;

// ------------------------------------------------------------------ monster aura (red target)
const MONSTER_FRAG = /* glsl */`
  uniform float uTime; uniform float uAlpha; uniform vec3 uColor;
  varying vec2 vUv;
  void main() {
    vec2 p = vUv - 0.5;
    float d = length(p) * 2.0;
    float pulse = 0.5 + 0.5 * sin(uTime * 3.4);
    float fill = pow(max(0.0, 1.0 - d), 1.5) * (0.45 + 0.3 * pulse);
    float a = atan(p.y, p.x) - uTime * 0.9;
    float ring = exp(-pow((d - 0.7) / 0.04, 2.0)) * smoothstep(-0.2, 0.6, sin(a * 4.0)) * 1.4;
    // crosshair ticks
    float ticks = 0.0;
    vec2 q = abs(p) * 2.0;
    ticks += step(q.x, 0.025) * smoothstep(0.78, 0.82, q.y) * step(q.y, 0.97);
    ticks += step(q.y, 0.025) * smoothstep(0.78, 0.82, q.x) * step(q.x, 0.97);
    float v = (fill + ring + ticks * 0.9) * uAlpha;
    gl_FragColor = vec4(uColor * v, v);
  }`;

// ------------------------------------------------------------------ fog of war
const FOG_FRAG = /* glsl */`
  uniform float uTime; uniform float uAlpha; uniform float uRadius; uniform int uCount;
  uniform vec3 uPlayers[${MAX_PLAYERS}];
  varying vec2 vWorld;
  float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
  float noise(vec2 p) {
    vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1, 0)), f.x), mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), f.x), f.y);
  }
  void main() {
    float clear = 0.0;
    for (int i = 0; i < ${MAX_PLAYERS}; i++) {
      if (i >= uCount) break;
      float d = distance(vWorld, uPlayers[i].xy);
      clear = max(clear, uPlayers[i].z * (1.0 - smoothstep(uRadius * 0.65, uRadius, d)));
    }
    float wisps = noise(vWorld / 180.0 + vec2(uTime * 0.03, uTime * 0.02)) * 0.6
                + noise(vWorld / 70.0 - vec2(uTime * 0.05, 0.0)) * 0.4;
    vec3 col = vec3(0.035, 0.04, 0.06) * wisps;           // barely-lit drifting mist
    gl_FragColor = vec4(col, (1.0 - clear) * uAlpha);
  }`;

const fog = (() => {
  const players = Array.from({ length: MAX_PLAYERS }, () => new THREE.Vector3());
  const mat = shaderMaterial(FOG_FRAG, {
    uPlayers: { value: players }, uCount: { value: 0 }, uRadius: { value: 240 },
  }, false);
  mat.uniforms.uAlpha.value = 1;
  const mesh = new THREE.Mesh(planeGeo, mat);
  mesh.renderOrder = LAYER.fog;
  scene.add(mesh);
  return { mesh, mat, players };
})();

// ------------------------------------------------------------------ DM effects
const EFFECT_FRAG = /* glsl */`
  uniform float uTime; uniform float uAlpha; uniform vec3 uColor; uniform int uKind; uniform float uAge;
  varying vec2 vUv;
  void main() {
    vec2 p = vUv - 0.5;
    float d = length(p) * 2.0;
    float v = 0.0;
    if (uKind == 0) {            // spell_aura: filled glow + bright rim + slow sparkle rotation
      float a = atan(p.y, p.x) + uTime * 0.4;
      v = smoothstep(1.0, 0.2, d) * 0.35 + exp(-pow((d - 0.93) / 0.04, 2.0))
        + 0.25 * smoothstep(0.7, 1.0, sin(a * 9.0)) * smoothstep(1.0, 0.6, d);
    } else if (uKind == 1) {     // area_burst: shockwave expanding outward
      float front = clamp(uAge / 0.9, 0.0, 1.0) * 0.82;  // stop short of the quad edge so the wave never clips
      v = exp(-pow((d - front) / 0.07, 2.0)) * 1.2 + smoothstep(front, 0.0, d) * 0.3 * (1.0 - front * 0.5);
    } else if (uKind == 2) {     // highlight: square with pulsing border
      vec2 q = abs(p) * 2.0; float m = max(q.x, q.y);
      v = step(m, 1.0) * (0.18 + 0.7 * smoothstep(0.86, 0.98, m)) * (0.75 + 0.25 * sin(uTime * 4.0));
    } else if (uKind == 3) {     // line: beam along the quad's length
      float across = abs(p.y) * 2.0;
      v = exp(-pow(across / 0.25, 2.0)) * (0.8 + 0.2 * sin(vUv.x * 40.0 - uTime * 12.0));
    } else {                     // darkness: soft black disc (normal blending)
      float s = smoothstep(1.0, 0.75, d);
      gl_FragColor = vec4(0.0, 0.0, 0.0, s * uAlpha); return;
    }
    v *= uAlpha;
    gl_FragColor = vec4(uColor * v, v);
  }`;
const EFFECT_KIND = { spell_aura: 0, area_burst: 1, highlight: 2, line: 3, darkness: 4 };
const effects = [];

function cellCenter(x, y) { return new THREE.Vector2((x + 0.5) * CELL, (y + 0.5) * CELL); }

function safeColor(c) {
  const col = new THREE.Color();
  try { col.setStyle(String(c)); } catch { col.set('gold'); }
  return col;
}

function addEffect(e) {
  const kind = EFFECT_KIND[e.type] ?? 0;
  const mat = shaderMaterial(EFFECT_FRAG, {
    uColor: { value: safeColor(e.color) }, uKind: { value: kind }, uAge: { value: 0 },
  }, kind !== 4);
  const mesh = new THREE.Mesh(planeGeo, mat);
  mesh.renderOrder = LAYER.effects;
  const c = cellCenter(e.x, e.y);
  if (kind === 3) {
    const end = cellCenter(e.x2 ?? e.x, e.y2 ?? e.y);
    const len = Math.max(CELL, c.distanceTo(end));
    mesh.position.set((c.x + end.x) / 2, (c.y + end.y) / 2, 0);
    mesh.scale.set(len, CELL * Math.max(0.6, e.radius * 0.6), 1);
    mesh.rotation.z = Math.atan2(end.y - c.y, end.x - c.x);
  } else {
    const size = kind === 2 ? (2 * e.radius + 1) * CELL : 2 * e.radius * CELL + CELL;
    mesh.position.set(c.x, c.y, 0);
    mesh.scale.set(size, size, 1);
  }
  scene.add(mesh);
  effects.push({ mesh, mat, born: clock.elapsedTime, duration: e.duration || 6 });
}

function clearEffects() {
  for (const fx of effects) { scene.remove(fx.mesh); fx.mat.dispose(); }
  effects.length = 0;
}

// ------------------------------------------------------------------ miniatures
const tokens = new Map(); // entity_id -> { mesh, mat, role, target, alive }
const PLAYER_COLOR = new THREE.Color(0x3d8bff);
const MONSTER_COLOR = new THREE.Color(0xff3b30);

function updateEntities(list) {
  const seen = new Set();
  for (const e of list) {
    seen.add(e.entity_id);
    let t = tokens.get(e.entity_id);
    if (t && t.role !== e.role) { removeToken(e.entity_id); t = null; }
    if (!t) {
      const isPlayer = e.role === 'player';
      const mat = shaderMaterial(isPlayer ? PLAYER_FRAG : MONSTER_FRAG, {
        uColor: { value: (isPlayer ? PLAYER_COLOR : MONSTER_COLOR).clone() },
      });
      const mesh = new THREE.Mesh(planeGeo, mat);
      mesh.renderOrder = isPlayer ? LAYER.players : LAYER.monsters;
      mesh.position.set(e.pixel_x, e.pixel_y, 0);
      scene.add(mesh);
      t = { mesh, mat, role: e.role, target: new THREE.Vector2(), alive: true };
      tokens.set(e.entity_id, t);
    }
    t.alive = true;
    t.target.set(e.pixel_x, e.pixel_y);
  }
  for (const [id, t] of tokens) if (!seen.has(id)) t.alive = false; // fades out in the frame loop
}

function removeToken(id) {
  const t = tokens.get(id);
  if (!t) return;
  scene.remove(t.mesh); t.mat.dispose(); tokens.delete(id);
}

// ------------------------------------------------------------------ narration + status
const narrationEl = document.getElementById('narration');
const statusEl = document.getElementById('status');
let narrationTimer = 0;

function showNarration(text) {
  if (!text) return;
  narrationEl.textContent = text;
  narrationEl.classList.add('show');
  clearTimeout(narrationTimer);
  narrationTimer = setTimeout(() => narrationEl.classList.remove('show'), 4000 + text.length * 55);
}

function setStatus(text) {
  statusEl.textContent = text || '';
  statusEl.classList.toggle('show', !!text);
}

// ------------------------------------------------------------------ websocket
let ws = null;
function connect() {
  ws = new WebSocket(WS_URL);
  ws.onopen = () => setStatus('');
  ws.onclose = () => { setStatus('reconnecting…'); setTimeout(connect, 1500); };
  ws.onerror = () => ws.close();
  ws.onmessage = (ev) => {
    let msg;
    try { msg = JSON.parse(ev.data); } catch { return; }
    switch (msg.type) {
      case 'config':
        CELL = msg.cell_px; FOG_RADIUS_CELLS = msg.fog_radius_cells ?? FOG_RADIUS_CELLS;
        setCanvas(msg.width, msg.height);
        buildGrid();
        break;
      case 'entities':
        updateEntities(msg.entities || []);
        break;
      case 'thinking':
        setStatus('The DM is thinking…');
        break;
      case 'dm':
        setStatus('');
        if (msg.replay) break;
        for (const e of msg.grid_effects || []) addEffect(e);
        showNarration(msg.narration);
        break;
      case 'reset':
        clearEffects();
        break;
    }
  };
}

// ------------------------------------------------------------------ keys
const helpEl = document.getElementById('help');
setTimeout(() => helpEl.classList.add('hide'), 8000);
window.addEventListener('keydown', (e) => {
  if (e.key === 'f' || e.key === 'F') {
    if (!document.fullscreenElement) document.documentElement.requestFullscreen?.();
    else document.exitFullscreen?.();
  } else if (e.code === 'Space') {
    e.preventDefault();
    if (ws?.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'listen' }));
    setStatus('Listening…');
  } else if (e.key === 'g' || e.key === 'G') {
    if (grid) grid.visible = !grid.visible;
  }
});

// ------------------------------------------------------------------ frame loop
function frame() {
  const dt = Math.min(clock.getDelta(), 0.1);
  const t = clock.elapsedTime;
  const ease = 1 - Math.exp(-dt * 10); // smooth the 5 FPS tracker updates into 60 FPS motion

  let count = 0;
  for (const [id, tok] of tokens) {
    tok.mesh.position.x += (tok.target.x - tok.mesh.position.x) * ease;
    tok.mesh.position.y += (tok.target.y - tok.mesh.position.y) * ease;
    const a = tok.mat.uniforms.uAlpha;
    a.value += ((tok.alive ? 1 : 0) - a.value) * (1 - Math.exp(-dt * 6));
    if (!tok.alive && a.value < 0.02) { removeToken(id); continue; }
    tok.mat.uniforms.uTime.value = t;
    const size = (tok.role === 'player' ? 1.6 : 1.5) * CELL * (0.9 + 0.1 * a.value);
    tok.mesh.scale.set(size, size, 1);
    if (tok.role === 'player' && count < MAX_PLAYERS) {
      fog.players[count++].set(tok.mesh.position.x, tok.mesh.position.y, a.value);
    }
  }
  fog.mat.uniforms.uCount.value = count;
  fog.mat.uniforms.uRadius.value = FOG_RADIUS_CELLS * CELL;
  fog.mat.uniforms.uTime.value = t;

  for (let i = effects.length - 1; i >= 0; i--) {
    const fx = effects[i];
    const age = t - fx.born;
    if (age > fx.duration) { scene.remove(fx.mesh); fx.mat.dispose(); effects.splice(i, 1); continue; }
    const fadeIn = Math.min(1, age / 0.4);
    const fadeOut = Math.min(1, (fx.duration - age) / 0.8);
    fx.mat.uniforms.uAlpha.value = fadeIn * fadeOut;
    fx.mat.uniforms.uTime.value = t;
    fx.mat.uniforms.uAge.value = age;
  }

  renderer.render(scene, camera);
  requestAnimationFrame(frame);
}

setCanvas(W, H);
buildGrid();
connect();
requestAnimationFrame(frame);
