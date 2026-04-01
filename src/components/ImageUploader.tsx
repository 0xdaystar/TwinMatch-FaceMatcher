"use client";

import { useCallback, useRef, useState } from "react";

interface ImageUploaderProps {
  label: string;
  onImageSelect: (image: HTMLImageElement, file: File) => void;
  disabled?: boolean;
}

export default function ImageUploader({
  label,
  onImageSelect,
  disabled,
}: ImageUploaderProps) {
  const [preview, setPreview] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    (file: File) => {
      if (!file.type.startsWith("image/")) return;

      const url = URL.createObjectURL(file);
      setPreview(url);

      const img = new Image();
      img.onload = () => {
        onImageSelect(img, file);
      };
      img.src = url;
    },
    [onImageSelect]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  return (
    <div className="flex flex-col items-center gap-3 w-full">
      <span className="text-sm font-medium text-zinc-500 uppercase tracking-wider">
        {label}
      </span>
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        onClick={() => !disabled && inputRef.current?.click()}
        className={`
          relative w-full aspect-square max-w-[320px] rounded-2xl border-2 border-dashed
          flex items-center justify-center cursor-pointer transition-all overflow-hidden
          ${dragging ? "border-emerald-400 bg-emerald-400/10" : "border-zinc-300 dark:border-zinc-700 hover:border-zinc-400 dark:hover:border-zinc-600"}
          ${disabled ? "opacity-50 cursor-not-allowed" : ""}
        `}
      >
        {preview ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={preview}
            alt="Uploaded face"
            className="w-full h-full object-cover"
          />
        ) : (
          <div className="flex flex-col items-center gap-2 p-6 text-center">
            <svg
              className="w-10 h-10 text-zinc-400"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={1.5}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M12 16v-8m0 0l-3 3m3-3l3 3M6.75 19.5a4.5 4.5 0 01-1.41-8.775 5.25 5.25 0 0110.338-2.32 3.75 3.75 0 013.07 5.876A3 3 0 0118.75 19.5H6.75z"
              />
            </svg>
            <p className="text-sm text-zinc-500">
              Drop an image here or click to upload
            </p>
          </div>
        )}
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          onChange={handleChange}
          className="hidden"
          disabled={disabled}
        />
      </div>
    </div>
  );
}
