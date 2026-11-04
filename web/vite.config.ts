import { resolve } from "node:path";
import { defineConfig } from "vite";

export default defineConfig({
  // onnxruntime-web finds its WASM file at runtime; Vite shouldn't pre-bundle it
  optimizeDeps: { exclude: ["onnxruntime-web"] },
  server: {
    proxy: {
      "/api": "http://localhost:8000",      // Hastaak backend (server/main.py)
      "/vision": "http://localhost:8001",   // native camera mode (python/vision_server.py)
    },
  },
  build: {
    rollupOptions: {
      input: { playground: resolve(__dirname, "index.html"), demo: resolve(__dirname, "demo.html") },
    },
  },
});
