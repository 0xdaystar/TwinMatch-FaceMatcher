const fs = require("fs");
const path = require("path");

const src = path.join(__dirname, "..", "node_modules", "onnxruntime-web", "dist");
const dst = path.join(__dirname, "..", "public", "onnx");

if (!fs.existsSync(dst)) fs.mkdirSync(dst, { recursive: true });

for (const file of fs.readdirSync(src)) {
  if (file.startsWith("ort-wasm-") && (file.endsWith(".wasm") || file.endsWith(".mjs"))) {
    fs.copyFileSync(path.join(src, file), path.join(dst, file));
    console.log(`Copied ${file}`);
  }
}
