"""
RunPod Serverless Handler for SDXL Image Generation
Supports RealVisXL V5.0 with dynamic LoRA loading
"""

import os
import io
import base64
import hashlib
import requests
import torch
import runpod
from pathlib import Path
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from safetensors.torch import load_file

# ============================================================================
# Configuration
# ============================================================================

MODEL_ID = os.getenv("MODEL_ID", "SG161222/RealVisXL_V5.0")
LORA_CACHE_DIR = Path("/tmp/lora_cache")
LORA_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Default generation parameters
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
DEFAULT_NUM_STEPS = 30
DEFAULT_GUIDANCE_SCALE = 7.0
DEFAULT_LORA_SCALE = 0.8

# Default negative prompt for photorealism
DEFAULT_NEGATIVE_PROMPT = (
    "bad hands, bad anatomy, ugly, deformed, face asymmetry, eyes asymmetry, "
    "deformed eyes, deformed mouth, open mouth, blurry, low quality, "
    "watermark, text, signature, cartoon, anime, illustration"
)

# ============================================================================
# Model Loading (done once at startup)
# ============================================================================

print(f"Loading model: {MODEL_ID}")

pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)

# Use DPM++ SDE Karras scheduler (recommended for RealVisXL)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config,
    algorithm_type="sde-dpmsolver++",
    use_karras_sigmas=True,
)

# Move to GPU
pipe = pipe.to("cuda")

# Enable memory optimizations
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()

print("Model loaded successfully!")

# Track currently loaded LoRA
current_lora_url = None


# ============================================================================
# Helper Functions
# ============================================================================

def download_lora(lora_url: str) -> Path:
    """Download LoRA from URL and cache it locally."""
    # Create hash of URL for cache filename
    url_hash = hashlib.md5(lora_url.encode()).hexdigest()[:12]
    
    # Determine file extension
    if ".safetensors" in lora_url.lower():
        ext = ".safetensors"
    else:
        ext = ".bin"
    
    cache_path = LORA_CACHE_DIR / f"lora_{url_hash}{ext}"
    
    # Return cached if exists
    if cache_path.exists():
        print(f"Using cached LoRA: {cache_path}")
        return cache_path
    
    print(f"Downloading LoRA from: {lora_url}")
    
    # Handle CivitAI URLs
    headers = {}
    if "civitai.com" in lora_url:
        api_key = os.getenv("CIVITAI_API_KEY", "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
    
    # Handle HuggingFace URLs
    if "huggingface.co" in lora_url:
        hf_token = os.getenv("HF_TOKEN", "")
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"
    
    response = requests.get(lora_url, headers=headers, stream=True, timeout=300)
    response.raise_for_status()
    
    # Save to cache
    with open(cache_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"LoRA downloaded and cached: {cache_path}")
    return cache_path


def load_lora(lora_url: str, lora_scale: float = DEFAULT_LORA_SCALE):
    """Load LoRA weights into the pipeline."""
    global current_lora_url
    
    # Unload previous LoRA if different
    if current_lora_url is not None and current_lora_url != lora_url:
        print("Unloading previous LoRA...")
        pipe.unload_lora_weights()
        current_lora_url = None
    
    # Skip if already loaded
    if current_lora_url == lora_url:
        print(f"LoRA already loaded, adjusting scale to {lora_scale}")
        pipe.set_adapters(["default"], adapter_weights=[lora_scale])
        return
    
    # Download and load new LoRA
    lora_path = download_lora(lora_url)
    
    print(f"Loading LoRA with scale {lora_scale}...")
    pipe.load_lora_weights(
        str(lora_path),
        adapter_name="default",
    )
    pipe.set_adapters(["default"], adapter_weights=[lora_scale])
    
    current_lora_url = lora_url
    print("LoRA loaded successfully!")


def unload_lora():
    """Unload LoRA weights from the pipeline."""
    global current_lora_url
    
    if current_lora_url is not None:
        print("Unloading LoRA weights...")
        pipe.unload_lora_weights()
        current_lora_url = None


def generate_image(
    prompt: str,
    negative_prompt: str = None,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    num_inference_steps: int = DEFAULT_NUM_STEPS,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    seed: int = None,
) -> bytes:
    """Generate an image and return as PNG bytes."""
    
    # Use default negative prompt if not provided
    if negative_prompt is None:
        negative_prompt = DEFAULT_NEGATIVE_PROMPT
    
    # Set seed for reproducibility
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(seed)
    else:
        seed = torch.randint(0, 2**32, (1,)).item()
        generator = torch.Generator(device="cuda").manual_seed(seed)
    
    print(f"Generating image with seed: {seed}")
    print(f"Prompt: {prompt[:100]}...")
    print(f"Size: {width}x{height}, Steps: {num_inference_steps}, CFG: {guidance_scale}")
    
    # Generate
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]
    
    # Convert to PNG bytes
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    
    return buffer.getvalue(), seed


# ============================================================================
# RunPod Handler
# ============================================================================

def handler(job):
    """
    RunPod serverless handler function.
    
    Input parameters:
        - prompt (str, required): The image generation prompt
        - negative_prompt (str, optional): Negative prompt
        - lora_url (str, optional): URL to LoRA weights (.safetensors or .bin)
        - lora_scale (float, optional): LoRA strength (0.0-1.0), default 0.8
        - width (int, optional): Image width, default 1024
        - height (int, optional): Image height, default 1024
        - num_inference_steps (int, optional): Number of steps, default 30
        - guidance_scale (float, optional): CFG scale, default 7.0
        - seed (int, optional): Random seed for reproducibility
    
    Returns:
        - image: Base64-encoded PNG image (data URI)
        - seed: The seed used for generation
    """
    job_input = job["input"]
    
    # Validate required parameters
    prompt = job_input.get("prompt")
    if not prompt:
        return {"error": "Missing required parameter: prompt"}
    
    # Extract optional parameters
    negative_prompt = job_input.get("negative_prompt")
    lora_url = job_input.get("lora_url")
    lora_scale = job_input.get("lora_scale", DEFAULT_LORA_SCALE)
    width = job_input.get("width", DEFAULT_WIDTH)
    height = job_input.get("height", DEFAULT_HEIGHT)
    num_inference_steps = job_input.get("num_inference_steps", DEFAULT_NUM_STEPS)
    guidance_scale = job_input.get("guidance_scale", DEFAULT_GUIDANCE_SCALE)
    seed = job_input.get("seed")
    
    # Validate dimensions (must be divisible by 8)
    width = (width // 8) * 8
    height = (height // 8) * 8
    
    # Clamp dimensions to reasonable range
    width = max(512, min(width, 2048))
    height = max(512, min(height, 2048))
    
    try:
        # Handle LoRA loading/unloading
        if lora_url:
            load_lora(lora_url, lora_scale)
        else:
            unload_lora()
        
        # Generate image
        image_bytes, used_seed = generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        
        # Encode as base64 data URI
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        image_data_uri = f"data:image/png;base64,{image_base64}"
        
        return {
            "image": image_data_uri,
            "seed": used_seed,
            "width": width,
            "height": height,
        }
    
    except Exception as e:
        print(f"Error during generation: {e}")
        return {"error": str(e)}


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
