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

export function drawLandmarks(
  canvas: HTMLCanvasElement,
  detection: FaceDetectionResult["detection"],
  sourceWidth: number,
  sourceHeight: number
): void {
  const displaySize = { width: canvas.width, height: canvas.height };
  faceapi.matchDimensions(canvas, { width: sourceWidth, height: sourceHeight });

  const resizedDetection = faceapi.resizeResults(detection, displaySize);
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Draw face bounding box
  const box = resizedDetection.detection.box;
  ctx.strokeStyle = "#00ff88";
  ctx.lineWidth = 2;
  ctx.strokeRect(box.x, box.y, box.width, box.height);

  // Draw landmarks
  const landmarks = resizedDetection.landmarks;
  const positions = landmarks.positions;

  for (const point of positions) {
    ctx.beginPath();
    ctx.arc(point.x, point.y, 2, 0, 2 * Math.PI);
    ctx.fillStyle = "#00ff88";
    ctx.fill();
  }

  // Draw connections for key features
  const jawline = landmarks.getJawOutline();
  const leftEyeBrow = landmarks.getLeftEyeBrow();
  const rightEyeBrow = landmarks.getRightEyeBrow();
  const nose = landmarks.getNose();
  const leftEye = landmarks.getLeftEye();
  const rightEye = landmarks.getRightEye();
  const mouth = landmarks.getMouth();

  const features = [jawline, leftEyeBrow, rightEyeBrow, nose, leftEye, rightEye, mouth];

  ctx.strokeStyle = "rgba(0, 255, 136, 0.5)";
  ctx.lineWidth = 1;

  for (const feature of features) {
    ctx.beginPath();
    ctx.moveTo(feature[0].x, feature[0].y);
    for (let i = 1; i < feature.length; i++) {
      ctx.lineTo(feature[i].x, feature[i].y);
    }
    ctx.stroke();
  }
}
