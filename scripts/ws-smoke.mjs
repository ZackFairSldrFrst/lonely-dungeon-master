// Prints the live server stream: `npm run smoke [-- ws://host:port/ws]`
import WebSocket from 'ws';

const url = process.argv[2] || 'ws://127.0.0.1:8765/ws';
const ws = new WebSocket(url);
const counts = {};
const started = Date.now();

ws.on('open', () => console.log(`connected to ${url}`));
ws.on('message', (data) => {
  const msg = JSON.parse(data);
  counts[msg.type] = (counts[msg.type] || 0) + 1;
  if (msg.type !== 'entities' || counts.entities <= 2) console.log(JSON.stringify(msg).slice(0, 300));
});
ws.on('error', (e) => { console.error(e.message); process.exit(1); });

setTimeout(() => {
  const secs = (Date.now() - started) / 1000;
  console.log(`\n${secs.toFixed(1)} s: ${JSON.stringify(counts)} (${((counts.entities || 0) / secs).toFixed(1)} entity updates per second)`);
  process.exit(0);
}, Number(process.env.SMOKE_SECONDS || 5) * 1000);
