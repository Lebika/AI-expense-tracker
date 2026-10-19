// Minimal service worker: makes the site installable as an app.
// It caches NOTHING, because the pages show private spending data.
self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', event => event.waitUntil(self.clients.claim()));
self.addEventListener('fetch', event => {
  if (event.request.mode === 'navigate') {
    event.respondWith(
      fetch(event.request).catch(() => new Response(
        '<meta name="viewport" content="width=device-width,initial-scale=1">' +
        '<body style="font-family:sans-serif;text-align:center;padding:40px">' +
        '<h2>You are offline</h2><p>Please check your internet and try again.</p></body>',
        { headers: { 'Content-Type': 'text/html' } }
      ))
    );
  }
});
