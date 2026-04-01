"use client";

import { useEffect, useRef, useCallback } from "react";
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

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container || !detection) return;

    const rect = container.getBoundingClientRect();
    // Set canvas pixel dimensions to match the container's CSS size
    canvas.width = rect.width;
    canvas.height = rect.height;

    import("@/lib/faceMatch").then(({ drawLandmarks }) => {
      drawLandmarks(canvas, detection.detection, imageWidth, imageHeight);
    });
  }, [detection, imageWidth, imageHeight]);

  useEffect(() => {
    draw();

    // Redraw if the container resizes (e.g. window resize)
    const container = containerRef.current;
    if (!container) return;
    const observer = new ResizeObserver(draw);
    observer.observe(container);
    return () => observer.disconnect();
  }, [draw]);

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
