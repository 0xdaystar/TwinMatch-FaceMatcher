import * as faceapi from "face-api.js";

let modelsLoaded = false;

export async function loadModels(
  onProgress?: (model: string) => void
): Promise<void> {
  if (modelsLoaded) return;

  const MODEL_URL = "/models";

  onProgress?.("SSD MobileNet");
  await faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL);

  onProgress?.("Face Landmarks");
  await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);

  onProgress?.("Face Recognition");
  await faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL);

  modelsLoaded = true;
}

export interface FaceDetectionResult {
  detection: faceapi.WithFaceDescriptor<
    faceapi.WithFaceLandmarks<{ detection: faceapi.FaceDetection }>
  >;
  descriptor: Float32Array;
}

export async function detectFace(
  input: HTMLImageElement | HTMLCanvasElement
): Promise<FaceDetectionResult | null> {
  const result = await faceapi
    .detectSingleFace(input, new faceapi.SsdMobilenetv1Options({ minConfidence: 0.5 }))
    .withFaceLandmarks()
    .withFaceDescriptor();

  if (!result) return null;

  return {
    detection: result,
    descriptor: result.descriptor,
  };
}

export function computeDistance(
  descriptor1: Float32Array,
  descriptor2: Float32Array
): number {
  return faceapi.euclideanDistance(
    Array.from(descriptor1),
    Array.from(descriptor2)
  );
}

export function distanceToConfidence(distance: number): number {
  // Distance of 0 = perfect match, ~0.6 = threshold, >1.0 = definitely different
  const MATCH_THRESHOLD = 0.6;
  const confidence = Math.max(0, (1 - distance / MATCH_THRESHOLD) * 100);
  return Math.min(100, Math.round(confidence * 10) / 10);
}

export function isMatch(distance: number): boolean {
  return distance < 0.6;
}

/**
 * Compute the object-cover transform: how CSS object-cover maps
 * source image coordinates to the displayed container.
 */
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

  // Compute the same transform CSS object-cover applies
  const { scale, offsetX, offsetY } = objectCoverTransform(
    sourceWidth,
    sourceHeight,
    canvas.width,
    canvas.height
  );

  // Helper: map a point from source image coords to canvas coords
  const mapX = (x: number) => x * scale + offsetX;
  const mapY = (y: number) => y * scale + offsetY;

  // Draw face bounding box
  const box = detection.detection.box;
  ctx.strokeStyle = "#00ff88";
  ctx.lineWidth = 2;
  ctx.strokeRect(
    mapX(box.x),
    mapY(box.y),
    box.width * scale,
    box.height * scale
  );

  // Draw landmark points
  const positions = detection.landmarks.positions;
  for (const point of positions) {
    ctx.beginPath();
    ctx.arc(mapX(point.x), mapY(point.y), 2.5, 0, 2 * Math.PI);
    ctx.fillStyle = "#00ff88";
    ctx.fill();
  }

  // Draw connections for key features
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
