# RunPod Serverless Deployment Guide

## Overview

This guide covers deploying the SDXL image generation worker to RunPod Serverless.

**Base Model:** RealVisXL V5.0 (photorealistic SDXL)  
**Features:** Dynamic LoRA loading, configurable parameters

---

## Prerequisites

1. **RunPod Account** with API access
2. **Docker** installed locally (for building)
3. **Docker Hub** or other container registry account

---

## Option 1: Build & Push to Docker Hub

### Step 1: Build the Docker Image

```bash
cd ~/workspace/auraglow/runpod

# Build with model baked in (larger image, faster cold start)
docker build -t yourusername/auraglow-sdxl:latest .

# OR build without model (smaller image, downloads on first run)
docker build --build-arg DOWNLOAD_MODEL=false -t yourusername/auraglow-sdxl:latest .
```

### Step 2: Push to Docker Hub

```bash
docker login
docker push yourusername/auraglow-sdxl:latest
```

### Step 3: Create RunPod Serverless Endpoint

1. Go to [RunPod Console](https://www.runpod.io/console/serverless)
2. Click **"+ New Endpoint"**
3. Configure:
   - **Name:** `auraglow-sdxl`
   - **Container Image:** `yourusername/auraglow-sdxl:latest`
   - **GPU Type:** RTX 4090 or A100 (recommended for speed)
   - **Min Workers:** 0 (scale to zero)
   - **Max Workers:** 5 (adjust as needed)
   - **GPU Count:** 1
   - **Volume:** 20GB (for model cache if not baked in)

4. **Environment Variables** (optional):
   ```
   CIVITAI_API_KEY=your_civitai_api_key  # For downloading LoRAs from CivitAI
   HF_TOKEN=your_huggingface_token        # For private HF models
   ```

5. Click **"Create Endpoint"**

---

## Option 2: Deploy via GitHub Integration

### Step 1: Push to GitHub

```bash
cd ~/workspace/auraglow/runpod
git init
git add .
git commit -m "Initial RunPod worker"
git remote add origin https://github.com/yourusername/auraglow-runpod.git
git push -u origin main
```

### Step 2: Connect GitHub to RunPod

1. Go to RunPod Console → Serverless → **"+ New Endpoint"**
2. Select **"GitHub"** as source
3. Connect your GitHub account
4. Select the repository
5. Configure endpoint settings (same as Option 1)

---

## API Usage

### Endpoint URL
```
https://api.runpod.ai/v2/{endpoint_id}/runsync
```

### Authentication
```
Authorization: Bearer YOUR_RUNPOD_API_KEY
```

### Basic Request (Text-to-Image)

```bash
curl -X POST "https://api.runpod.ai/v2/{endpoint_id}/runsync" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "beautiful woman, professional photo, studio lighting, 8k",
      "negative_prompt": "ugly, deformed, blurry, low quality",
      "width": 1024,
      "height": 1024,
      "seed": 42
    }
  }'
```

### Request with LoRA

```bash
curl -X POST "https://api.runpod.ai/v2/{endpoint_id}/runsync" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "ohwx woman, beautiful portrait, professional photo",
      "negative_prompt": "ugly, deformed, blurry",
      "lora_url": "https://huggingface.co/user/lora-name/resolve/main/model.safetensors",
      "lora_scale": 0.8,
      "width": 1024,
      "height": 1024,
      "num_inference_steps": 30,
      "guidance_scale": 7.0,
      "seed": 12345
    }
  }'
```

### Async Request (for longer jobs)

```bash
# Start job
curl -X POST "https://api.runpod.ai/v2/{endpoint_id}/run" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"prompt": "your prompt here"}}'

# Response: {"id": "job-id-here", "status": "IN_QUEUE"}

# Check status
curl "https://api.runpod.ai/v2/{endpoint_id}/status/{job-id}" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY"
```

---

## Input Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | *required* | Image generation prompt |
| `negative_prompt` | string | (default) | What to avoid in the image |
| `lora_url` | string | null | URL to LoRA weights (.safetensors) |
| `lora_scale` | float | 0.8 | LoRA strength (0.0-1.0) |
| `width` | int | 1024 | Image width (512-2048, divisible by 8) |
| `height` | int | 1024 | Image height (512-2048, divisible by 8) |
| `num_inference_steps` | int | 30 | Denoising steps (more = better quality) |
| `guidance_scale` | float | 7.0 | CFG scale (higher = more prompt adherence) |
| `seed` | int | random | Seed for reproducibility |

---

## Output Format

```json
{
  "delayTime": 1234,
  "executionTime": 5678,
  "id": "job-id",
  "output": {
    "image": "data:image/png;base64,iVBORw0KGgo...",
    "seed": 42,
    "width": 1024,
    "height": 1024
  },
  "status": "COMPLETED"
}
```

---

## LoRA Sources

### From HuggingFace
```
https://huggingface.co/{user}/{repo}/resolve/main/{filename}.safetensors
```

### From CivitAI (requires API key)
```
https://civitai.com/api/download/models/{model-version-id}
```

Set `CIVITAI_API_KEY` environment variable in RunPod console.

### Self-hosted (recommended for privacy)
Upload LoRAs to your own storage (S3, DigitalOcean Spaces, etc.):
```
https://your-bucket.s3.amazonaws.com/loras/custom-lora.safetensors
```

---

## Recommended GPU Types

| GPU | VRAM | Speed | Cost | Notes |
|-----|------|-------|------|-------|
| RTX 4090 | 24GB | Fast | ~$0.44/hr | Best value |
| RTX A6000 | 48GB | Fast | ~$0.79/hr | Multiple LoRAs |
| A100 80GB | 80GB | Fastest | ~$1.99/hr | Production |
| RTX 3090 | 24GB | Medium | ~$0.31/hr | Budget option |

---

## Troubleshooting

### Cold Start Slow
- Bake model into Docker image (`DOWNLOAD_MODEL=true`)
- Use network volume for model cache
- Keep min workers > 0

### Out of Memory
- Reduce image dimensions
- Use RTX 4090 or higher
- The handler uses `enable_model_cpu_offload()` automatically

### LoRA Not Loading
- Check URL is publicly accessible
- For CivitAI, ensure `CIVITAI_API_KEY` is set
- Check file extension (.safetensors or .bin)

### Generation Errors
- Check prompt isn't empty
- Ensure dimensions are divisible by 8
- Check RunPod worker logs for details

---

## Local Testing

```bash
# Create test input
echo '{"input": {"prompt": "test image"}}' > test_input.json

# Run locally (requires GPU)
python handler.py
```

---

## Cost Estimation

Assuming RTX 4090 at $0.44/hr:
- ~15 seconds per image = ~$0.002 per image
- 1000 images ≈ $2.00

With scale-to-zero, you only pay for actual generation time.

---

## Integration with AuraGlow

Update your AuraGlow backend to call RunPod instead of Fal.ai:

```typescript
const RUNPOD_API_KEY = 'YOUR_RUNPOD_API_KEY';
const ENDPOINT_ID = 'your-endpoint-id';

async function generateImage(params: {
  prompt: string;
  loraUrl?: string;
  loraScale?: number;
  width?: number;
  height?: number;
  seed?: number;
}) {
  const response = await fetch(
    `https://api.runpod.ai/v2/${ENDPOINT_ID}/runsync`,
    {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${RUNPOD_API_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        input: {
          prompt: params.prompt,
          lora_url: params.loraUrl,
          lora_scale: params.loraScale ?? 0.8,
          width: params.width ?? 1024,
          height: params.height ?? 1024,
          seed: params.seed,
        },
      }),
    }
  );

  const result = await response.json();
  
  if (result.status === 'COMPLETED') {
    return {
      imageUrl: result.output.image, // base64 data URI
      seed: result.output.seed,
    };
  } else {
    throw new Error(result.output?.error || 'Generation failed');
  }
}
```
