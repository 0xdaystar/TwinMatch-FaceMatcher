import * as faceapi from "face-api.js";
import * as ort from "onnxruntime-web";

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let faceModelsLoaded = false;
let arcfaceSession: ort.InferenceSession | null = null;

// ---------------------------------------------------------------------------
// ArcFace 112x112 alignment template (standard InsightFace coordinates)
// ---------------------------------------------------------------------------
const ARCFACE_TEMPLATE: [number, number][] = [
  [38.2946, 51.6963], // left eye
  [73.5318, 51.5014], // right eye
  [56.0252, 71.7366], // nose tip
  [41.5493, 92.3655], // left mouth corner
  [70.7299, 92.2041], // right mouth corner
];

// ---------------------------------------------------------------------------
// Model loading
// ---------------------------------------------------------------------------
export async function loadModels(
  onProgress?: (model: string) => void
): Promise<void> {
  const MODEL_URL = "/models";

  if (!faceModelsLoaded) {
    onProgress?.("Face Detector");
    await faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL);

    onProgress?.("Landmarks");
    await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);

    faceModelsLoaded = true;
  }

  if (!arcfaceSession) {
    onProgress?.("ArcFace (ONNX)");
    ort.env.wasm.wasmPaths = "/onnx/";
    arcfaceSession = await ort.InferenceSession.create(
      `${MODEL_URL}/w600k_mbf.onnx`,
      { executionProviders: ["wasm"] }
    );
  }
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
export interface FaceDetectionResult {
  detection: faceapi.WithFaceLandmarks<{ detection: faceapi.FaceDetection }>;
  descriptor: Float32Array; // 512-D ArcFace embedding
}

// ---------------------------------------------------------------------------
// Detection + embedding pipeline
// ---------------------------------------------------------------------------
export async function detectFace(
  input: HTMLImageElement | HTMLCanvasElement
): Promise<FaceDetectionResult | null> {
  // 1. Detect face + 68-point landmarks via face-api.js
  const result = await faceapi
    .detectSingleFace(input, new faceapi.SsdMobilenetv1Options({ minConfidence: 0.5 }))
    .withFaceLandmarks();

  if (!result) return null;

  // 2. Extract 5 key points from the 68-point set
  const pts68 = result.landmarks.positions;
  const fivePoints = extractFivePoints(pts68);

  // 3. Align face to 112x112 via similarity transform
  const aligned = alignFace(input, fivePoints);

  // 4. Run ArcFace ONNX inference to get 512-D embedding
  const descriptor = await getEmbedding(aligned);

  return { detection: result, descriptor };
}

// ---------------------------------------------------------------------------
// 5-point extraction from 68-point landmarks
// ---------------------------------------------------------------------------
function avg(points: faceapi.Point[]): [number, number] {
  let sx = 0,
    sy = 0;
  for (const p of points) {
    sx += p.x;
    sy += p.y;
  }
  return [sx / points.length, sy / points.length];
}

function extractFivePoints(
  pts: faceapi.Point[]
): [number, number][] {
  return [
    avg(pts.slice(36, 42)), // left eye center
    avg(pts.slice(42, 48)), // right eye center
    [pts[30].x, pts[30].y], // nose tip
    [pts[48].x, pts[48].y], // left mouth corner
    [pts[54].x, pts[54].y], // right mouth corner
  ];
}

// ---------------------------------------------------------------------------
// Similarity transform (4-DOF: uniform scale + rotation + translation)
// Solved via least-squares over 5 point correspondences.
// ---------------------------------------------------------------------------
function computeSimilarityTransform(
  src: [number, number][],
  dst: [number, number][]
): { a: number; b: number; tx: number; ty: number } {
  const n = src.length;
  // Build normal equations for [ a  b  tx ]
  //                              [-b  a  ty ]
  // minimising  sum || dst_i - M * src_i ||^2
  let sx = 0, sy = 0, dx = 0, dy = 0;
  for (let i = 0; i < n; i++) {
    sx += src[i][0]; sy += src[i][1];
    dx += dst[i][0]; dy += dst[i][1];
  }
  sx /= n; sy /= n; dx /= n; dy /= n;

  let num = 0, den = 0;
  for (let i = 0; i < n; i++) {
    const sxc = src[i][0] - sx;
    const syc = src[i][1] - sy;
    const dxc = dst[i][0] - dx;
    const dyc = dst[i][1] - dy;
    num += sxc * dyc - syc * dxc;
    den += sxc * dxc + syc * dyc;
  }

  let ssrc = 0;
  for (let i = 0; i < n; i++) {
    ssrc += (src[i][0] - sx) ** 2 + (src[i][1] - sy) ** 2;
  }

  const a = den / ssrc;
  const b = num / ssrc;
  const tx = dx - a * sx + b * sy;
  const ty = dy - b * sx - a * sy;

  return { a, b, tx, ty };
}

// ---------------------------------------------------------------------------
// Warp face to 112x112 aligned crop using canvas
// ---------------------------------------------------------------------------
function alignFace(
  image: HTMLImageElement | HTMLCanvasElement,
  fivePoints: [number, number][]
): HTMLCanvasElement {
  const SIZE = 112;
  const canvas = document.createElement("canvas");
  canvas.width = SIZE;
  canvas.height = SIZE;
  const ctx = canvas.getContext("2d")!;

  // Compute transform: source landmarks → ArcFace template
  const { a, b, tx, ty } = computeSimilarityTransform(
    fivePoints,
    ARCFACE_TEMPLATE
  );

  // Canvas setTransform maps from destination to source, so we need the
  // INVERSE of our transform.  For a similarity transform:
  //   M     = [ a  -b  tx ]     M_inv = (1/det) * [ a   b  -(a*tx + b*ty)  ]
  //           [ b   a  ty ]                        [ -b  a   (b*tx - a*ty)  ]
  // det = a^2 + b^2
  const det = a * a + b * b;
  const ai = a / det;
  const bi = b / det;
  const txi = -(ai * tx + bi * ty);
  const tyi = (bi * tx - ai * ty);

  // setTransform(a, b, c, d, e, f) sets the matrix:
  //   [ a  c  e ]
  //   [ b  d  f ]
  // We want the inverse mapping (dst pixel → src pixel) because canvas
  // drawImage uses the transform to map the *source* into the *destination*.
  // Actually canvas transform maps user-space to device-space, which means
  // we provide the FORWARD transform and canvas applies it.
  ctx.setTransform(a, b, -b, a, tx, ty);

  // Draw the source image — canvas will apply the affine transform
  ctx.drawImage(image, 0, 0);

  // Reset transform for any later drawing
  ctx.setTransform(1, 0, 0, 1, 0, 0);

  return canvas;
}

// ---------------------------------------------------------------------------
// ONNX ArcFace inference → 512-D L2-normalised embedding
// ---------------------------------------------------------------------------
async function getEmbedding(
  aligned: HTMLCanvasElement
): Promise<Float32Array> {
  if (!arcfaceSession) throw new Error("ArcFace model not loaded");

  const SIZE = 112;
  const ctx = aligned.getContext("2d")!;
  const imageData = ctx.getImageData(0, 0, SIZE, SIZE);
  const { data } = imageData; // RGBA uint8

  // Convert to NCHW float32 tensor [1, 3, 112, 112]
  // InsightFace MobileFaceNet expects pixel values in [0, 255] range
  // with standard normalization: (pixel - 127.5) / 127.5
  const floats = new Float32Array(3 * SIZE * SIZE);
  for (let i = 0; i < SIZE * SIZE; i++) {
    floats[i] = (data[i * 4] - 127.5) / 127.5;                     // R
    floats[SIZE * SIZE + i] = (data[i * 4 + 1] - 127.5) / 127.5;   // G
    floats[2 * SIZE * SIZE + i] = (data[i * 4 + 2] - 127.5) / 127.5; // B
  }

  const inputTensor = new ort.Tensor("float32", floats, [1, 3, SIZE, SIZE]);

  // The model's input name — InsightFace MobileFaceNet uses "input.1"
  // We'll detect it dynamically from the session
  const inputName = arcfaceSession.inputNames[0];
  const outputName = arcfaceSession.outputNames[0];

  const results = await arcfaceSession.run({ [inputName]: inputTensor });
  const raw = results[outputName].data as Float32Array;

  // L2-normalise the embedding
  let norm = 0;
  for (let i = 0; i < raw.length; i++) norm += raw[i] * raw[i];
  norm = Math.sqrt(norm);

  const embedding = new Float32Array(raw.length);
  for (let i = 0; i < raw.length; i++) embedding[i] = raw[i] / norm;

  return embedding;
}

// ---------------------------------------------------------------------------
// Similarity & matching (cosine similarity — higher is better)
// ---------------------------------------------------------------------------
export function computeSimilarity(
  emb1: Float32Array,
  emb2: Float32Array
): number {
  let dot = 0;
  for (let i = 0; i < emb1.length; i++) dot += emb1[i] * emb2[i];
  return dot; // already L2-normalised, so dot = cosine similarity
}

export function similarityToConfidence(similarity: number): number {
  // Sigmoid: midpoint at 0.45, steepness 12
  const midpoint = 0.45;
  const k = 12;
  const confidence = 100 / (1 + Math.exp(-k * (similarity - midpoint)));
  return Math.round(confidence * 10) / 10;
}

export function isMatch(similarity: number): boolean {
  return similarity > 0.45;
}

// ---------------------------------------------------------------------------
// Landmark drawing (unchanged)
// ---------------------------------------------------------------------------
function objectCoverTransform(
  srcW: number,
  srcH: number,
  dstW: number,
  dstH: number
): { scale: number; offsetX: number; offsetY: number } {
  const scale = Math.max(dstW / srcW, dstH / srcH);
  const offsetX = (dstW - srcW * scale) / 2;
  const offsetY = (dstH - srcH * scale) / 2;
  return { scale, offsetX, offsetY };
}

export function drawLandmarks(
  canvas: HTMLCanvasElement,
  detection: FaceDetectionResult["detection"],
  sourceWidth: number,
  sourceHeight: number
): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const { scale, offsetX, offsetY } = objectCoverTransform(
    sourceWidth,
    sourceHeight,
    canvas.width,
    canvas.height
  );

  const mapX = (x: number) => x * scale + offsetX;
  const mapY = (y: number) => y * scale + offsetY;

  const box = detection.detection.box;
  ctx.strokeStyle = "#00ff88";
  ctx.lineWidth = 2;
  ctx.strokeRect(
    mapX(box.x),
    mapY(box.y),
    box.width * scale,
    box.height * scale
  );

  const positions = detection.landmarks.positions;
  for (const point of positions) {
    ctx.beginPath();
    ctx.arc(mapX(point.x), mapY(point.y), 2.5, 0, 2 * Math.PI);
    ctx.fillStyle = "#00ff88";
    ctx.fill();
  }

  const landmarks = detection.landmarks;
  const features = [
    landmarks.getJawOutline(),
    landmarks.getLeftEyeBrow(),
    landmarks.getRightEyeBrow(),
    landmarks.getNose(),
    landmarks.getLeftEye(),
    landmarks.getRightEye(),
    landmarks.getMouth(),
  ];

  ctx.strokeStyle = "rgba(0, 255, 136, 0.5)";
  ctx.lineWidth = 1;

  for (const feature of features) {
    ctx.beginPath();
    ctx.moveTo(mapX(feature[0].x), mapY(feature[0].y));
    for (let i = 1; i < feature.length; i++) {
      ctx.lineTo(mapX(feature[i].x), mapY(feature[i].y));
    }
    ctx.stroke();
  }
}
