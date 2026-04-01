"use client";

import { useEffect, useRef } from "react";
import type { FaceDetectionResult } from "@/lib/faceMatch";

interface FaceCanvasProps {
  imageSrc: string;
  detection: FaceDetectionResult | null;
  imageWidth: number;
  imageHeight: number;
}

export default function FaceCanvas({
  imageSrc,
  detection,
  imageWidth,
  imageHeight,
}: FaceCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container || !detection) return;

    // Match canvas to displayed image size
    const rect = container.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;

    // Import dynamically to avoid SSR issues
    import("@/lib/faceMatch").then(({ drawLandmarks }) => {
      drawLandmarks(canvas, detection.detection, imageWidth, imageHeight);
    });
  }, [detection, imageWidth, imageHeight]);

  return (
    <div ref={containerRef} className="relative w-full aspect-square max-w-[320px] rounded-2xl overflow-hidden">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={imageSrc}
        alt="Face with landmarks"
        className="w-full h-full object-cover"
      />
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full"
      />
    </div>
  );
}
