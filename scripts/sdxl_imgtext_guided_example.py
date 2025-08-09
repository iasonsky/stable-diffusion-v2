import torch
from PIL import Image

# Patch diffusers imports for older huggingface_hub versions
import huggingface_hub
if not hasattr(huggingface_hub, "cached_download"):
    from huggingface_hub import hf_hub_download
    huggingface_hub.cached_download = hf_hub_download

from diffusers import StableDiffusionXLPipeline


def encode_reference_image(image: Image.Image, encoder) -> torch.Tensor:
    """Encode an input PIL image into a sequence of tokens.

    The encoder should return a tensor of shape [batch, n_tokens, 768].
    Replace this stub with your aligner/MLP encoder.
    """
    # Example placeholder implementation
    with torch.no_grad():
        image = image.convert("RGB").resize((224, 224))
        img_tensor = torch.tensor(torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
                                ).view(image.size[1], image.size[0], 3).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)
        tokens = encoder(img_tensor)
    return tokens


def generate_with_image_guidance(
    model_id: str,
    prompt: str,
    reference_image: Image.Image,
    image_encoder,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
):
    """Run SDXL generation injecting image tokens into the CLIP stream."""

    pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    pipe.enable_model_cpu_offload()

    # 1. Encode prompts normally (both text encoders)
    prompt_embeds, neg_prompt_embeds, pooled, neg_pooled = pipe.encode_prompt(prompt)

    # 2. Encode reference image using custom MLP aligner
    image_tokens = encode_reference_image(reference_image, image_encoder)

    # 3. Fuse CLIP embeddings with image tokens
    #    Only text_encoder tokens are fused; T5 embeddings (in pooled) are kept as is
    fused_prompt_embeds = torch.cat([prompt_embeds, image_tokens], dim=1)

    # Negative prompt embeddings are padded with zeros for new tokens
    if neg_prompt_embeds is not None:
        pad_shape = list(image_tokens.shape)
        pad_shape[0] = neg_prompt_embeds.shape[0]
        pad = torch.zeros(pad_shape, dtype=neg_prompt_embeds.dtype, device=neg_prompt_embeds.device)
        neg_prompt_embeds = torch.cat([neg_prompt_embeds, pad], dim=1)

    # 4. Run inference with overridden prompt embeddings
    output = pipe(
        prompt_embeds=fused_prompt_embeds,
        negative_prompt_embeds=neg_prompt_embeds,
        pooled_prompt_embeds=pooled,
        negative_pooled_prompt_embeds=neg_pooled,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )
    return output.images[0]


__all__ = ["generate_with_image_guidance"]
