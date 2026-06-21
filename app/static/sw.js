self.addEventListener("install", (e) => {
  console.log("Service Worker: Installed");
  e.waitUntil(
    caches.open("yolo-cache-v2").then((cache) => {
      return cache.addAll([
        "/",
        "/static/index.html",
        "/static/scripts.js",
        "/static/styles.css"
      ]);
    })
  );
});

self.addEventListener("fetch", (e) => {
  e.respondWith(
    caches.match(e.request).then((resp) => {
      return resp || fetch(e.request);
    })
  );
});