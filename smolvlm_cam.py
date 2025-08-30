#!/usr/bin/env python3
import argparse
import sys
import time
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq


MODEL_ID = "HuggingFaceTB/SmolVLM-Instruct"

def pick_device() -> str:
    # Prioritize CUDA, then Apple Metal (MPS), then CPU
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def load_model(device: str, dtype_str: Optional[str] = None):
    # Default dtype choices by device
    if dtype_str is None:
        if device == "cuda":
            dtype = torch.bfloat16
        elif device == "mps":
            # bf16 is mixed on MPS; fp16 often works better than bf16 here
            dtype = torch.float16
        else:
            dtype = torch.float32
    else:
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_str]

    attn_impl = "flash_attention_2" if device == "cuda" else "eager"

    print(f"[load] device={device}, dtype={dtype}, attn_impl={attn_impl}")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        _attn_implementation=attn_impl,
    ).to(device)

    # On MPS, certain models are more stable with full precision for generate().
    # You can force float32 with: model = model.to(torch.float32)
    return processor, model

def bgr_to_pil(frame_bgr: np.ndarray) -> Image.Image:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)

def generate_caption(
    processor,
    model,
    image: Image.Image,
    device: str,
    user_text: str = "Describe the scene briefly."
) -> str:
    # Build a chat-style message with one image + one text question
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]}
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

    # Prepare inputs
    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    # Generate
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
        )
        out = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # The chat template usually prefixes "Assistant: "
    # Strip that for overlay neatness.
    return out.replace("Assistant:", "").strip()

def overlay_multiline_text(
    frame: np.ndarray,
    text: str,
    origin=(10, 30),
    line_height=28,
    max_width_px: int = 1000,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=0.7,
    thickness=2,
):
    # Very simple word wrap for OpenCV overlay
    words = text.split()
    lines = []
    cur = ""
    for w in words:
        test = (cur + " " + w).strip()
        size, _ = cv2.getTextSize(test, font, font_scale, thickness)
        if size[0] > max_width_px and cur:
            lines.append(cur)
            cur = w
        else:
            cur = test
    if cur:
        lines.append(cur)

    x, y = origin
    # semi-transparent box behind text for readability
    if lines:
        # compute box size
        widths = [cv2.getTextSize(line, font, font_scale, thickness)[0][0] for line in lines]
        box_w = min(max_width_px, max(widths) + 16)
        box_h = line_height * len(lines) + 16
        overlay = frame.copy()
        cv2.rectangle(overlay, (x - 8, y - 24), (x - 8 + box_w, y - 24 + box_h), (0, 0, 0), -1)
        # 60% opacity
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    for i, line in enumerate(lines):
        yy = y + i * line_height
        cv2.putText(frame, line, (x, yy), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

def main():
    ap = argparse.ArgumentParser(description="SmolVLM webcam describer")
    ap.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    ap.add_argument("--prompt", type=str, default="Describe the scene briefly.",
                    help="Text prompt to pair with the frame")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default=None,
                    help="Force model dtype (default auto)")
    ap.add_argument("--res", type=int, default=None,
                    help="Optional longest_edge multiple (N*384) for processor size; e.g., 3 -> 1152")
    args = ap.parse_args()

    device = pick_device()
    processor, model = load_model(device, dtype_str=args.dtype)

    if args.res is not None:
        # Re-create processor with a smaller visual resolution if you need memory savings
        processor.image_processor.size = {"longest_edge": args.res * 384}
        print(f"[processor] Set longest_edge to {processor.image_processor.size['longest_edge']}")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("ERROR: Could not open webcam. Try a different --camera index.", file=sys.stderr)
        sys.exit(1)

    print("Controls: SPACE = analyze current frame · Q = quit")
    latest_text = "Press SPACE to analyze the scene."

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("WARN: Failed to read from camera.")
                break

            # Show previous result overlaid
            overlay_multiline_text(frame, latest_text)

            cv2.imshow("SmolVLM Webcam", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' '):
                snapshot = frame.copy()
                pil_img = bgr_to_pil(snapshot)

                t0 = time.time()
                try:
                    text = generate_caption(processor, model, pil_img, device, user_text=args.prompt)
                except Exception as e:
                    text = f"[error] {type(e).__name__}: {e}"
                dt = time.time() - t0
                latest_text = f"{text}  (in {dt:.2f}s)"
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()