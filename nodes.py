import torch
import numpy as np
from typing import Optional, Tuple
import sys
import os

# Import bokeh adapter
from .bokeh_diffusion.adapter_flux import BokehFluxControlAdapter
from .bokeh_diffusion.utils import color_transfer_lab

MIN_BOKEH, MAX_BOKEH = 0, 30


def find_flux_transformer(model):
    """Find the FLUX transformer in a ComfyUI model"""
    # Try different possible locations for the transformer
    if hasattr(model, 'model'):
        if hasattr(model.model, 'transformer'):
            return model.model.transformer
        elif hasattr(model.model, 'diffusion_model'):
            # Check if it's a FluxTransformer2DModel
            transformer = model.model.diffusion_model
            if 'FluxTransformer2DModel' in str(type(transformer)) or hasattr(transformer, 'attn_processors'):
                return transformer
    elif hasattr(model, 'diffusion_model'):
        transformer = model.diffusion_model
        if 'FluxTransformer2DModel' in str(type(transformer)) or hasattr(transformer, 'attn_processors'):
            return transformer
    elif hasattr(model, 'transformer'):
        return model.transformer

    # Last resort: check if model itself is the transformer
    if hasattr(model, 'attn_processors'):
        return model

    return None


class BokehAdapterLoader:
    """Load the Bokeh Diffusion adapter from HuggingFace"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "adapter_name": ("STRING", {
                    "default": "atfortes/BokehDiffusion",
                    "multiline": False
                }),
            },
            "optional": {
                "bokeh_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1
                }),
                "lora_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1
                }),
            }
        }

    RETURN_TYPES = ("BOKEH_ADAPTER",)
    RETURN_NAMES = ("adapter",)
    FUNCTION = "load_adapter"
    CATEGORY = "bokeh_diffusion"

    def load_adapter(self, adapter_name: str, bokeh_scale: float = 1.0, lora_scale: float = 1.0):
        """Load the bokeh adapter configuration"""
        adapter_info = {
            "adapter_name": adapter_name,
            "bokeh_scale": bokeh_scale,
            "lora_scale": lora_scale,
            "adapter": None  # Will be loaded when model is available
        }
        return (adapter_info,)


class ApplyBokehAdapter:
    """Apply the Bokeh adapter to a FLUX model"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "adapter": ("BOKEH_ADAPTER",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_adapter"
    CATEGORY = "bokeh_diffusion"

    def apply_adapter(self, model, adapter):
        """Apply the bokeh adapter to the model's transformer"""
        try:
            # Find the transformer
            transformer = find_flux_transformer(model)

            if transformer is None:
                raise Exception("Could not find FLUX transformer in model. Make sure you're using a FLUX model.")

            # Check if transformer has attn_processors (required for adapter)
            if not hasattr(transformer, 'attn_processors'):
                raise Exception("Model transformer does not have attn_processors. This may not be a FLUX model.")

            # Load the adapter if not already loaded
            adapter_name = adapter["adapter_name"]
            if adapter["adapter"] is None:
                print(f"Loading Bokeh adapter from {adapter_name}...")
                adapter_obj = BokehFluxControlAdapter.from_pretrained(
                    adapter_name,
                    base_model=transformer,
                )
                # Move to same device as transformer
                device = next(transformer.parameters()).device
                dtype = next(transformer.parameters()).dtype
                adapter_obj = adapter_obj.to(device=device, dtype=dtype)
                adapter["adapter"] = adapter_obj
                print("Bokeh adapter loaded successfully.")

            adapter_obj = adapter["adapter"]

            # Set scales
            if adapter["bokeh_scale"] != 1.0:
                adapter_obj.set_bokeh_scale(adapter["bokeh_scale"])
            if adapter["lora_scale"] != 1.0:
                adapter_obj.set_lora_scale(adapter["lora_scale"])

            # Store adapter in model for use during sampling
            model.bokeh_adapter = adapter_obj
            model.bokeh_adapter_config = adapter
            model._bokeh_transformer = transformer

            return (model,)
        except Exception as e:
            raise Exception(f"Failed to apply bokeh adapter: {e}")


class BokehLevel:
    """Set bokeh level for generation"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bokeh_level": ("FLOAT", {
                    "default": 15.0,
                    "min": MIN_BOKEH,
                    "max": MAX_BOKEH,
                    "step": 0.1,
                    "tooltip": "Bokeh level from 0 (sharp) to 30 (maximum blur)"
                }),
            },
            "optional": {
                "bokeh_pivot": ("FLOAT", {
                    "default": 15.0,
                    "min": MIN_BOKEH,
                    "max": MAX_BOKEH,
                    "step": 0.1,
                    "tooltip": "Pivot bokeh level for grounded generation (scene consistency)"
                }),
                "num_grounding_steps": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 50,
                    "tooltip": "Number of grounding steps for scene consistency (0 = disabled, 18-24 recommended)"
                }),
            }
        }

    RETURN_TYPES = ("BOKEH_CONTROL",)
    RETURN_NAMES = ("bokeh_control",)
    FUNCTION = "create_control"
    CATEGORY = "bokeh_diffusion"

    def create_control(self, bokeh_level: float, bokeh_pivot: Optional[float] = None, num_grounding_steps: int = 0):
        """Create bokeh control parameters"""
        # Normalize bokeh level (0-30 -> 0-1)
        bokeh_target = bokeh_level / MAX_BOKEH
        bokeh_pivot_norm = bokeh_pivot / MAX_BOKEH if bokeh_pivot is not None else None

        control = {
            "bokeh_target": bokeh_target,
            "bokeh_pivot": bokeh_pivot_norm,
            "num_grounding_steps": num_grounding_steps,
            "is_grounded": num_grounding_steps > 0 and bokeh_pivot is not None
        }
        return (control,)


class BokehKSampler:
    """KSampler with bokeh control support for FLUX models"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": ("STRING", {"default": "euler"}),
                "scheduler": ("STRING", {"default": "normal"}),
                "bokeh_control": ("BOKEH_CONTROL",),
            },
            "optional": {
                "true_cfg": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "True CFG scale for negative prompt"
                }),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sample"
    CATEGORY = "bokeh_diffusion/sampling"

    def sample(self, model, positive, negative, latent_image, seed, steps, cfg, sampler_name, scheduler,
               bokeh_control, true_cfg=1.0):
        """Sample with bokeh control"""

        # Check if model has bokeh adapter
        if not hasattr(model, 'bokeh_adapter'):
            raise Exception("Model does not have bokeh adapter applied. Use ApplyBokehAdapter node first.")

        bokeh_adapter = model.bokeh_adapter
        transformer = model._bokeh_transformer

        # Get bokeh control parameters
        bokeh_target = bokeh_control["bokeh_target"]
        bokeh_pivot = bokeh_control.get("bokeh_pivot")
        num_grounding_steps = bokeh_control.get("num_grounding_steps", 0)
        is_grounded = bokeh_control.get("is_grounded", False)

        # Get latent
        samples = latent_image["samples"]
        device = samples.device
        dtype = samples.dtype

        # Prepare bokeh annotations
        if is_grounded:
            bokeh_ann = torch.tensor([bokeh_target, bokeh_pivot], dtype=dtype, device=device)
        else:
            bokeh_ann = torch.tensor([bokeh_target], dtype=dtype, device=device)
        bokeh_ann = bokeh_ann.unsqueeze(1)
        neg_bokeh_ann = torch.full_like(bokeh_ann, -1)

        # Store bokeh info in transformer for use during forward pass
        # The attention processors will access this via joint_attention_kwargs
        transformer._bokeh_ann = bokeh_ann
        transformer._neg_bokeh_ann = neg_bokeh_ann
        transformer._is_grounded = is_grounded
        transformer._num_grounding_steps = num_grounding_steps
        transformer._true_cfg = true_cfg
        transformer._current_step = 0

        # Patch the transformer's forward method to inject bokeh annotations
        original_forward = transformer.forward

        def patched_forward(
            hidden_states,
            timestep=None,
            encoder_hidden_states=None,
            pooled_projections=None,
            added_cond_kwargs=None,
            return_dict=True,
            **kwargs
        ):
            # Get current step info
            current_step = getattr(transformer, '_current_step', 0)
            is_grounded = getattr(transformer, '_is_grounded', False)
            num_grounding_steps = getattr(transformer, '_num_grounding_steps', 0)
            bokeh_ann = getattr(transformer, '_bokeh_ann', None)

            # Prepare joint_attention_kwargs with bokeh info
            joint_attention_kwargs = kwargs.get('joint_attention_kwargs', {})
            if bokeh_ann is not None:
                perform_swap = is_grounded and (current_step < num_grounding_steps)
                batch_swap_ids = [1, 1] if is_grounded else None

                # Get bokeh embeddings from adapter
                bokeh_embeds = bokeh_adapter.embedding_layer(bokeh_ann)
                joint_attention_kwargs['bokeh_embeds'] = bokeh_embeds
                joint_attention_kwargs['perform_swap'] = perform_swap
                joint_attention_kwargs['batch_swap_ids'] = batch_swap_ids

            kwargs['joint_attention_kwargs'] = joint_attention_kwargs
            return original_forward(
                hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled_projections,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=return_dict,
                **kwargs
            )

        transformer.forward = patched_forward

        try:
            # Use standard ComfyUI sampling
            import comfy.samplers
            import comfy.sample

            # Get the sampler function
            sampler = comfy.samplers.sampler_object(sampler_name)

            # Create a callback to track steps for grounded generation
            def step_callback(step, x0, x, total_steps):
                transformer._current_step = step

            # Sample using ComfyUI's sample function
            # Note: This is a simplified version. Full implementation may need
            # to handle negative prompts with bokeh control separately
            disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

            sampled = comfy.sample.sample(
                model,
                seed,
                steps,
                cfg,
                sampler_name,
                scheduler,
                positive,
                negative,
                samples,
                denoise=1.0,
                disable_noise=False,
                start_step=None,
                last_step=None,
                force_full_denoise=False,
                callback=step_callback,
                disable_pbar=disable_pbar,
                seed_mode="scale_alike"
            )

            # Restore original forward
            transformer.forward = original_forward

            return (sampled,)

        except ImportError:
            # Fallback: if ComfyUI imports fail, return original samples
            # This allows the node to be defined even if not in ComfyUI
            print("Warning: ComfyUI not found. Bokeh control requires ComfyUI environment.")
            transformer.forward = original_forward
            return ({"samples": samples},)
        except Exception as e:
            # Restore original forward on error
            transformer.forward = original_forward
            raise Exception(f"Sampling failed: {e}")


# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "BokehAdapterLoader": BokehAdapterLoader,
    "ApplyBokehAdapter": ApplyBokehAdapter,
    "BokehLevel": BokehLevel,
    "BokehKSampler": BokehKSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BokehAdapterLoader": "Load Bokeh Adapter",
    "ApplyBokehAdapter": "Apply Bokeh Adapter",
    "BokehLevel": "Bokeh Level",
    "BokehKSampler": "Bokeh KSampler",
}
