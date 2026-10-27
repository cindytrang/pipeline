const { test, before, after } = require('node:test');
const assert = require('node:assert/strict');
const app = require('../index');
let server, base;
before(async () => {
  server = await new Promise(resolve => {
    const listener = app.listen(0, '127.0.0.1', () => resolve(listener));
  });
  base = `http://127.0.0.1:${server.address().port}`;
});
after(() => new Promise(resolve => server.close(resolve)));
test('home page lists the synthetic datapack', async () => {
  const response = await fetch(base);
  assert.equal(response.status, 200);
  assert.match(await response.text(), /synthetic-demo/);
});
test('annotated messages render predictions and labels', async () => {
  const response = await fetch(`${base}/visualizations/synthetic-demo/chat/example-chat/`);
  assert.equal(response.status, 200);
  const html = await response.text();
  assert.match(html, /data-prediction="0.72"/);
  assert.match(html, /Synthetic example/);
});
test('missing datapack returns the existing 404 view', async () => {
  assert.equal((await fetch(`${base}/visualizations/not-found/`)).status, 404);
});
test('CSS and chart assets are served from the frontend directory', async () => {
  for (const asset of ['/css/main.css', '/vendor/Chart.min.js']) {
    assert.equal((await fetch(base + asset)).status, 200, asset);
  }
});
