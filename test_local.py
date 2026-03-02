#!/usr/bin/env python3
"""
Local test script for the RunPod handler.
Run this to test image generation without deploying.

Usage:
    python test_local.py
    python test_local.py --prompt "custom prompt"
    python test_local.py --lora-url "https://example.com/lora.safetensors"
"""

import argparse
import base64
import sys
from pathlib import Path

# Import the handler
from handler import handler, pipe


def save_image(data_uri: str, output_path: str = "output.png"):
    """Save a base64 data URI to a file."""
    # Remove the data URI prefix
    if data_uri.startswith("data:"):
        data_uri = data_uri.split(",", 1)[1]
    
    # Decode and save
    image_data = base64.b64decode(data_uri)
    with open(output_path, "wb") as f:
        f.write(image_data)
    
    print(f"Image saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Test the RunPod SDXL handler locally")
    parser.add_argument("--prompt", type=str, default="beautiful woman, professional photo, studio lighting, 8k uhd",
                        help="Image generation prompt")
    parser.add_argument("--negative-prompt", type=str, default=None,
                        help="Negative prompt (uses default if not specified)")
    parser.add_argument("--lora-url", type=str, default=None,
                        help="URL to LoRA weights")
    parser.add_argument("--lora-scale", type=float, default=0.8,
                        help="LoRA strength (0.0-1.0)")
    parser.add_argument("--width", type=int, default=1024,
                        help="Image width")
    parser.add_argument("--height", type=int, default=1024,
                        help="Image height")
    parser.add_argument("--steps", type=int, default=30,
                        help="Number of inference steps")
    parser.add_argument("--cfg", type=float, default=7.0,
                        help="Guidance scale (CFG)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")
    parser.add_argument("--output", type=str, default="output.png",
                        help="Output file path")
    
    args = parser.parse_args()
    
    # Build job input
    job_input = {
        "prompt": args.prompt,
        "width": args.width,
        "height": args.height,
        "num_inference_steps": args.steps,
        "guidance_scale": args.cfg,
    }
    
    if args.negative_prompt:
        job_input["negative_prompt"] = args.negative_prompt
    
    if args.lora_url:
        job_input["lora_url"] = args.lora_url
        job_input["lora_scale"] = args.lora_scale
    
    if args.seed is not None:
        job_input["seed"] = args.seed
    
    # Create job object
    job = {"input": job_input}
    
    print("\n" + "=" * 60)
    print("Running local test...")
    print("=" * 60)
    print(f"Prompt: {args.prompt}")
    print(f"Size: {args.width}x{args.height}")
    print(f"Steps: {args.steps}, CFG: {args.cfg}")
    if args.lora_url:
        print(f"LoRA: {args.lora_url} (scale: {args.lora_scale})")
    print("=" * 60 + "\n")
    
    # Call handler
    result = handler(job)
    
    # Check result
    if "error" in result:
        print(f"\n❌ Error: {result['error']}")
        sys.exit(1)
    
    print(f"\n✅ Generation successful!")
    print(f"   Seed: {result['seed']}")
    print(f"   Size: {result['width']}x{result['height']}")
    
    # Save image
    save_image(result["image"], args.output)
    print(f"\n📷 Image saved to: {args.output}")


if __name__ == "__main__":
    main()
