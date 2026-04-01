"use client";

import { useState, useCallback, useRef } from "react";
import ImageUploader from "@/components/ImageUploader";
import FaceCanvas from "@/components/FaceCanvas";
import MatchResult from "@/components/MatchResult";
import {
  loadModels,
  detectFace,
  computeDistance,
  distanceToConfidence,
  isMatch,
  type FaceDetectionResult,
} from "@/lib/faceMatch";

type Status =
  | "idle"
  | "loading-models"
  | "detecting"
  | "done"
  | "error";

interface FaceData {
  src: string;
  width: number;
  height: number;
  detection: FaceDetectionResult | null;
}

export default function Home() {
  const [status, setStatus] = useState<Status>("idle");
  const [modelProgress, setModelProgress] = useState("");
  const [error, setError] = useState("");
  const [face1, setFace1] = useState<FaceData | null>(null);
  const [face2, setFace2] = useState<FaceData | null>(null);
  const [result, setResult] = useState<{
    distance: number;
    confidence: number;
    match: boolean;
  } | null>(null);

  const img1Ref = useRef<HTMLImageElement | null>(null);
  const img2Ref = useRef<HTMLImageElement | null>(null);

  const runComparison = useCallback(async () => {
    const img1 = img1Ref.current;
    const img2 = img2Ref.current;
    if (!img1 || !img2) return;

    setError("");
    setResult(null);

    try {
      setStatus("loading-models");
      await loadModels((model) => setModelProgress(model));

      setStatus("detecting");
      const [det1, det2] = await Promise.all([
        detectFace(img1),
        detectFace(img2),
      ]);

      if (!det1 || !det2) {
        setError(
          !det1 && !det2
            ? "No face detected in either image. Try clearer, front-facing photos."
            : !det1
            ? "No face detected in the first image."
            : "No face detected in the second image."
        );
        setStatus("error");

        setFace1((prev) =>
          prev ? { ...prev, detection: det1 } : null
        );
        setFace2((prev) =>
          prev ? { ...prev, detection: det2 } : null
        );
        return;
      }

      setFace1((prev) =>
        prev ? { ...prev, detection: det1 } : null
      );
      setFace2((prev) =>
        prev ? { ...prev, detection: det2 } : null
      );

      const distance = computeDistance(det1.descriptor, det2.descriptor);
      const confidence = distanceToConfidence(distance);
      const match = isMatch(distance);

      setResult({ distance, confidence, match });
      setStatus("done");
    } catch (err) {
      console.error(err);
      setError("Something went wrong. Please try again.");
      setStatus("error");
    }
  }, []);

  const handleImage1 = useCallback(
    (img: HTMLImageElement) => {
      img1Ref.current = img;
      setFace1({
        src: img.src,
        width: img.naturalWidth,
        height: img.naturalHeight,
        detection: null,
      });
      setResult(null);
    },
    []
  );

  const handleImage2 = useCallback(
    (img: HTMLImageElement) => {
      img2Ref.current = img;
      setFace2({
        src: img.src,
        width: img.naturalWidth,
        height: img.naturalHeight,
        detection: null,
      });
      setResult(null);
    },
    []
  );

  const bothSelected = !!img1Ref.current && !!img2Ref.current && !!face1 && !!face2;
  const isProcessing = status === "loading-models" || status === "detecting";

  return (
    <div className="flex flex-col flex-1 items-center bg-zinc-50 dark:bg-zinc-950 font-sans">
      <main className="flex flex-col items-center w-full max-w-3xl px-4 py-12 gap-8">
        {/* Header */}
        <div className="text-center space-y-2">
          <h1 className="text-4xl font-bold tracking-tight text-zinc-900 dark:text-zinc-50">
            TwinMatch
          </h1>
          <p className="text-zinc-500 dark:text-zinc-400 max-w-md">
            Upload two face photos and find out if they match.
            Everything runs in your browser — images never leave your device.
          </p>
        </div>

        {/* Image upload area */}
        <div className="flex flex-col sm:flex-row items-center sm:items-start gap-6 w-full justify-center">
          <div className="w-full max-w-[320px]">
            {face1?.detection ? (
              <div className="flex flex-col items-center gap-3">
                <span className="text-sm font-medium text-zinc-500 uppercase tracking-wider">
                  Face 1
                </span>
                <FaceCanvas
                  imageSrc={face1.src}
                  detection={face1.detection}
                  imageWidth={face1.width}
                  imageHeight={face1.height}
                />
              </div>
            ) : (
              <ImageUploader
                label="Face 1"
                onImageSelect={handleImage1}
                disabled={isProcessing}
              />
            )}
          </div>

          {/* VS divider */}
          <div className="flex items-center justify-center sm:pt-10">
            <span className="text-2xl font-bold text-zinc-300 dark:text-zinc-700">
              vs
            </span>
          </div>

          <div className="w-full max-w-[320px]">
            {face2?.detection ? (
              <div className="flex flex-col items-center gap-3">
                <span className="text-sm font-medium text-zinc-500 uppercase tracking-wider">
                  Face 2
                </span>
                <FaceCanvas
                  imageSrc={face2.src}
                  detection={face2.detection}
                  imageWidth={face2.width}
                  imageHeight={face2.height}
                />
              </div>
            ) : (
              <ImageUploader
                label="Face 2"
                onImageSelect={handleImage2}
                disabled={isProcessing}
              />
            )}
          </div>
        </div>

        {/* Compare button */}
        <button
          onClick={runComparison}
          disabled={!bothSelected || isProcessing}
          className={`
            px-8 py-3 rounded-full font-semibold text-white transition-all
            ${
              !bothSelected || isProcessing
                ? "bg-zinc-300 dark:bg-zinc-700 cursor-not-allowed"
                : "bg-zinc-900 dark:bg-white dark:text-zinc-900 hover:bg-zinc-700 dark:hover:bg-zinc-200 active:scale-95"
            }
          `}
        >
          {isProcessing
            ? status === "loading-models"
              ? `Loading ${modelProgress}...`
              : "Analyzing faces..."
            : "Compare Faces"}
        </button>

        {/* Error message */}
        {error && (
          <p className="text-red-600 dark:text-red-400 text-sm text-center max-w-md">
            {error}
          </p>
        )}

        {/* Result */}
        {result && (
          <MatchResult
            distance={result.distance}
            confidence={result.confidence}
            match={result.match}
          />
        )}

        {/* Reset button */}
        {(face1 || face2 || result) && !isProcessing && (
          <button
            onClick={() => {
              setFace1(null);
              setFace2(null);
              setResult(null);
              setError("");
              setStatus("idle");
              img1Ref.current = null;
              img2Ref.current = null;
            }}
            className="text-sm text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300 underline underline-offset-4"
          >
            Start over
          </button>
        )}

        {/* Footer */}
        <footer className="text-xs text-zinc-400 text-center mt-8">
          Powered by face-api.js &amp; TensorFlow.js. No data is sent to any server.
        </footer>
      </main>
    </div>
  );
}
