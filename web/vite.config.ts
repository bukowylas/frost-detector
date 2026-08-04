import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Build to web/dist, which the FastAPI backend serves as static files. In dev,
// proxy /api to the backend so the single-origin setup matches production.
export default defineConfig({
  plugins: [react()],
  build: { outDir: "dist" },
  server: {
    proxy: { "/api": "http://localhost:8000" },
  },
});
