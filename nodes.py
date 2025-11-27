import torch
import numpy as np
from typing import Optional, Tuple
import sys
import os
import logging

# Import bokeh adapter
from .bokeh_diffusion.adapter_flux import BokehFluxControlAdapter
from .bokeh_diffusion.utils import color_transfer_lab

MIN_BOKEH, MAX_BOKEH = 0, 30

# Configure logging - can be controlled via BOKEH_DEBUG environment variable
# Set BOKEH_DEBUG=1 for DEBUG level, BOKEH_DEBUG=2 for even more verbose
_log_level = os.environ.get('BOKEH_DEBUG', '1')
if _log_level == '2':
    logging.basicConfig(level=logging.DEBUG, format='[BokehDiffusion] %(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
elif _log_level == '1':
    logging.basicConfig(level=logging.INFO, format='[BokehDiffusion] %(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
else:
    logging.basicConfig(level=logging.WARNING, format='[BokehDiffusion] %(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.WARNING)


def find_flux_transformer(model):
    """Find the FLUX transformer in a ComfyUI model"""
    logger.info("=" * 80)
    logger.info("Starting FLUX transformer search")
    logger.info(f"Model type: {type(model)}")
    logger.info(f"Model ID: {id(model)}")

    # Helper function to check if an object is a FLUX transformer
    def is_flux_transformer(obj, obj_name="unknown"):
        if obj is None:
            logger.debug(f"  [is_flux_transformer] {obj_name}: Object is None")
            return False
        obj_type = str(type(obj))
        logger.debug(f"  [is_flux_transformer] {obj_name}: Type = {obj_type}")

        # Check for FluxTransformer2DModel class name
        if 'FluxTransformer2DModel' in obj_type:
            logger.info(f"  [is_flux_transformer] {obj_name}: ✓ MATCH - FluxTransformer2DModel detected!")
            return True

        # Check for attn_processors attribute (key indicator of FLUX transformer)
        has_attn_procs = hasattr(obj, 'attn_processors')
        has_set_attn = hasattr(obj, 'set_attn_processor')
        logger.debug(f"  [is_flux_transformer] {obj_name}: has attn_processors={has_attn_procs}, has set_attn_processor={has_set_attn}")

        if has_attn_procs:
            # Additional check: FLUX transformers have set_attn_processor method
            if has_set_attn:
                logger.info(f"  [is_flux_transformer] {obj_name}: ✓ MATCH - Has attn_processors and set_attn_processor!")
                return True
            else:
                logger.debug(f"  [is_flux_transformer] {obj_name}: Has attn_processors but missing set_attn_processor")

        logger.debug(f"  [is_flux_transformer] {obj_name}: ✗ Not a transformer")
        return False

    # Try different possible locations for the transformer
    # Most common: model.model.diffusion_model (ComfyUI ModelPatcher structure)
    logger.info("Checking model.model structure...")
    if hasattr(model, 'model'):
        logger.info(f"  model.model exists: {type(model.model)}")
        logger.debug(f"  model.model attributes: {[a for a in dir(model.model) if not a.startswith('__')][:20]}")

        if hasattr(model.model, 'diffusion_model'):
            logger.info("  Found model.model.diffusion_model")
            flux_model = model.model.diffusion_model
            flux_type = str(type(flux_model))
            logger.info(f"  flux_model type: {flux_type}")

            # Check if it's a ComfyUI Flux wrapper (comfy.ldm.flux.model.Flux)
            # In this case, the actual transformer is usually in flux_model.transformer
            if 'comfy.ldm.flux' in flux_type or 'comfy.model_base.Flux' in flux_type:
                logger.info("  ✓ Detected ComfyUI Flux wrapper - searching for transformer inside...")

                # The actual FluxTransformer2DModel is typically in the 'transformer' attribute
                logger.info("  Checking flux_model.transformer...")
                if hasattr(flux_model, 'transformer'):
                    logger.info("    flux_model.transformer exists!")
                    transformer = flux_model.transformer
                    logger.info(f"    transformer type: {type(transformer)}")
                    if is_flux_transformer(transformer, "flux_model.transformer"):
                        logger.info("    ✓ FOUND TRANSFORMER at flux_model.transformer!")
                        return transformer
                    else:
                        logger.warning("    flux_model.transformer exists but is_flux_transformer returned False")
                else:
                    logger.warning("    flux_model.transformer attribute NOT FOUND")

                # Check all direct attributes (including private ones)
                logger.info("  Checking direct attributes: model, _model, transformer, _transformer...")
                for attr_name in ['model', '_model', 'transformer', '_transformer']:
                    logger.debug(f"    Checking attribute: {attr_name}")
                    if hasattr(flux_model, attr_name):
                        logger.info(f"      {attr_name} exists!")
                        try:
                            inner_obj = getattr(flux_model, attr_name)
                            logger.debug(f"      {attr_name} type: {type(inner_obj)}")
                            if inner_obj is not None:
                                if is_flux_transformer(inner_obj, f"flux_model.{attr_name}"):
                                    logger.info(f"      ✓ FOUND TRANSFORMER at flux_model.{attr_name}!")
                                    return inner_obj
                                # Recursively check if it contains the transformer
                                if hasattr(inner_obj, 'attn_processors') or 'FluxTransformer' in str(type(inner_obj)):
                                    logger.debug(f"      {attr_name} has attn_processors or FluxTransformer in type")
                                    # Check nested
                                    if hasattr(inner_obj, 'transformer'):
                                        logger.info(f"        {attr_name}.transformer exists!")
                                        nested = getattr(inner_obj, 'transformer')
                                        if is_flux_transformer(nested, f"flux_model.{attr_name}.transformer"):
                                            logger.info(f"        ✓ FOUND TRANSFORMER at flux_model.{attr_name}.transformer!")
                                            return nested
                        except (AttributeError, RuntimeError, TypeError) as e:
                            logger.debug(f"      Error accessing {attr_name}: {e}")
                            continue
                    else:
                        logger.debug(f"      {attr_name} does not exist")

                # Search through all named modules/children to find the transformer
                # ComfyUI might store it as a submodule
                logger.info("  Searching through named_modules...")
                transformer_candidates = []
                attn_processor_candidates = []
                try:
                    module_count = 0
                    for name, module in flux_model.named_modules():
                        module_count += 1
                        module_type = str(type(module))
                        # Check if it's a FluxTransformer2DModel
                        if 'FluxTransformer2DModel' in module_type:
                            logger.debug(f"    Found FluxTransformer2DModel candidate: {name}")
                            transformer_candidates.append((name, module_type))
                            if is_flux_transformer(module, f"named_modules[{name}]"):
                                logger.info(f"    ✓ FOUND TRANSFORMER in named_modules: {name} ({type(module)})!")
                                return module
                        # Check if it has attn_processors
                        elif hasattr(module, 'attn_processors') and hasattr(module, 'set_attn_processor'):
                            logger.debug(f"    Found attn_processor candidate: {name}")
                            attn_processor_candidates.append((name, module_type))
                            if is_flux_transformer(module, f"named_modules[{name}]"):
                                logger.info(f"    ✓ FOUND TRANSFORMER-LIKE in named_modules: {name} ({type(module)})!")
                                return module

                    logger.info(f"  Searched {module_count} modules total")
                    logger.info(f"  Found {len(transformer_candidates)} FluxTransformer2DModel candidates")
                    logger.info(f"  Found {len(attn_processor_candidates)} attn_processor candidates")

                    # Print candidate modules for debugging
                    if transformer_candidates:
                        logger.info("  FluxTransformer2DModel candidates:")
                        for name, mod_type in transformer_candidates[:20]:  # Show first 20
                            logger.info(f"    {name}: {mod_type}")

                    if attn_processor_candidates:
                        logger.info("  attn_processor candidates:")
                        for name, mod_type in attn_processor_candidates[:20]:  # Show first 20
                            logger.info(f"    {name}: {mod_type}")

                except (AttributeError, RuntimeError, TypeError) as e:
                    logger.error(f"  Error searching named_modules: {e}", exc_info=True)
                    pass

                # Also check direct children
                logger.info("  Checking direct children...")
                try:
                    child_count = 0
                    for child in flux_model.children():
                        child_count += 1
                        logger.debug(f"    Child {child_count}: {type(child)}")
                        if is_flux_transformer(child, f"children[{child_count}]"):
                            logger.info(f"    ✓ FOUND TRANSFORMER in children: {type(child)}!")
                            return child
                    logger.info(f"  Checked {child_count} direct children")
                except (AttributeError, RuntimeError, TypeError) as e:
                    logger.error(f"  Error searching children: {e}", exc_info=True)
                    pass

                # Try searching through all attributes more thoroughly
                # Check if any attribute contains FluxTransformer2DModel
                logger.info("  Searching through all private attributes...")
                try:
                    all_attrs = [a for a in dir(flux_model) if a.startswith('_') and not a.startswith('__')]
                    logger.debug(f"  Found {len(all_attrs)} private attributes to check")
                    checked_count = 0
                    for attr_name in all_attrs:
                        checked_count += 1
                        if checked_count % 10 == 0:
                            logger.debug(f"    Checked {checked_count}/{len(all_attrs)} attributes...")
                        try:
                            attr_obj = getattr(flux_model, attr_name, None)
                            if attr_obj is not None:
                                attr_type = str(type(attr_obj))
                                # Check if it's a FluxTransformer2DModel
                                if 'FluxTransformer2DModel' in attr_type:
                                    logger.info(f"    Found FluxTransformer2DModel in attribute: {attr_name} ({attr_type})")
                                    if is_flux_transformer(attr_obj, f"attr[{attr_name}]"):
                                        logger.info(f"    ✓ FOUND TRANSFORMER in attribute: {attr_name}!")
                                        return attr_obj
                                # Check if it has attn_processors
                                if hasattr(attr_obj, 'attn_processors') and hasattr(attr_obj, 'set_attn_processor'):
                                    logger.info(f"    Found transformer-like object in attribute: {attr_name} ({attr_type})")
                                    if is_flux_transformer(attr_obj, f"attr[{attr_name}]"):
                                        logger.info(f"    ✓ FOUND TRANSFORMER in attribute: {attr_name}!")
                                        return attr_obj
                        except (AttributeError, RuntimeError, TypeError) as e:
                            logger.debug(f"    Error accessing {attr_name}: {e}")
                            continue
                    logger.info(f"  Checked {checked_count} private attributes")
                except Exception as e:
                    logger.error(f"  Error searching attributes: {e}", exc_info=True)
                    pass
            else:
                # Not a ComfyUI wrapper, check if it's the transformer directly
                logger.info("  Not a ComfyUI wrapper, checking if flux_model is transformer directly...")
                if is_flux_transformer(flux_model, "flux_model (direct)"):
                    logger.info("  ✓ FOUND TRANSFORMER - flux_model is transformer directly!")
                    return flux_model
        else:
            logger.warning("  model.model.diffusion_model does not exist")

        logger.info("  Checking model.model.transformer...")
        if hasattr(model.model, 'transformer'):
            logger.info("    model.model.transformer exists!")
            transformer = model.model.transformer
            if is_flux_transformer(transformer, "model.model.transformer"):
                logger.info("    ✓ FOUND TRANSFORMER at model.model.transformer!")
                return transformer
        else:
            logger.debug("    model.model.transformer does not exist")

        # Sometimes the model.model itself might be the transformer
        logger.info("  Checking if model.model is transformer directly...")
        if is_flux_transformer(model.model, "model.model (direct)"):
            logger.info("  ✓ FOUND TRANSFORMER - model.model is transformer directly!")
            return model.model
    else:
        logger.warning("  model.model does not exist")

    # Check for private _model attribute (some wrappers use this)
    logger.info("Checking model._model...")
    if hasattr(model, '_model'):
        logger.info("  model._model exists!")
        try:
            inner_model = model._model
            if inner_model is not None:
                logger.info(f"    _model type: {type(inner_model)}")
                if hasattr(inner_model, 'diffusion_model'):
                    logger.info("      _model.diffusion_model exists!")
                    transformer = inner_model.diffusion_model
                    if is_flux_transformer(transformer, "model._model.diffusion_model"):
                        logger.info("      ✓ FOUND TRANSFORMER at model._model.diffusion_model!")
                        return transformer
                if hasattr(inner_model, 'transformer'):
                    logger.info("      _model.transformer exists!")
                    transformer = inner_model.transformer
                    if is_flux_transformer(transformer, "model._model.transformer"):
                        logger.info("      ✓ FOUND TRANSFORMER at model._model.transformer!")
                        return transformer
                if is_flux_transformer(inner_model, "model._model (direct)"):
                    logger.info("      ✓ FOUND TRANSFORMER - model._model is transformer directly!")
                    return inner_model
        except (AttributeError, RuntimeError, TypeError) as e:
            logger.debug(f"  Error accessing _model: {e}")
            pass
    else:
        logger.debug("  model._model does not exist")

    # Check if model has a get_model() method (some ComfyUI wrappers)
    logger.info("Checking model.get_model()...")
    if hasattr(model, 'get_model'):
        logger.info("  model.get_model() exists!")
        try:
            inner_model = model.get_model()
            logger.info(f"    get_model() returned: {type(inner_model)}")
            if inner_model is not None:
                if hasattr(inner_model, 'diffusion_model'):
                    logger.info("      inner_model.diffusion_model exists!")
                    transformer = inner_model.diffusion_model
                    if is_flux_transformer(transformer, "get_model().diffusion_model"):
                        logger.info("      ✓ FOUND TRANSFORMER at get_model().diffusion_model!")
                        return transformer
                if hasattr(inner_model, 'transformer'):
                    logger.info("      inner_model.transformer exists!")
                    transformer = inner_model.transformer
                    if is_flux_transformer(transformer, "get_model().transformer"):
                        logger.info("      ✓ FOUND TRANSFORMER at get_model().transformer!")
                        return transformer
                if is_flux_transformer(inner_model, "get_model() (direct)"):
                    logger.info("      ✓ FOUND TRANSFORMER - get_model() result is transformer directly!")
                    return inner_model
        except (AttributeError, RuntimeError, TypeError) as e:
            logger.debug(f"  Error calling get_model(): {e}")
            pass
    else:
        logger.debug("  model.get_model() does not exist")

    # Direct access: model.diffusion_model
    logger.info("Checking model.diffusion_model (direct)...")
    if hasattr(model, 'diffusion_model'):
        logger.info("  model.diffusion_model exists!")
        transformer = model.diffusion_model
        if is_flux_transformer(transformer, "model.diffusion_model (direct)"):
            logger.info("  ✓ FOUND TRANSFORMER at model.diffusion_model!")
            return transformer
    else:
        logger.debug("  model.diffusion_model does not exist")

    # Direct access: model.transformer
    logger.info("Checking model.transformer (direct)...")
    if hasattr(model, 'transformer'):
        logger.info("  model.transformer exists!")
        transformer = model.transformer
        if is_flux_transformer(transformer, "model.transformer (direct)"):
            logger.info("  ✓ FOUND TRANSFORMER at model.transformer!")
            return transformer
    else:
        logger.debug("  model.transformer does not exist")

    # Last resort: check if model itself is the transformer
    logger.info("Checking if model itself is transformer...")
    if is_flux_transformer(model, "model (self)"):
        logger.info("  ✓ FOUND TRANSFORMER - model itself is transformer!")
        return model

    # Recursive search as final fallback
    logger.info("Starting recursive search as final fallback...")
    def search_recursive(obj, depth=0, max_depth=3, visited=None, path=""):
        if visited is None:
            visited = set()
        if depth > max_depth or id(obj) in visited:
            return None
        visited.add(id(obj))
        logger.debug(f"  [recursive] depth={depth}, path={path}, type={type(obj)}")

        # Check if current object is the transformer
        if is_flux_transformer(obj, f"recursive[{path}]"):
            logger.info(f"  ✓ FOUND TRANSFORMER in recursive search at: {path}!")
            return obj

        # Search in common attribute names
        for attr_name in ['diffusion_model', 'transformer', 'model', '_model']:
            if hasattr(obj, attr_name):
                try:
                    attr_obj = getattr(obj, attr_name)
                    if attr_obj is not None and id(attr_obj) not in visited:
                        new_path = f"{path}.{attr_name}" if path else attr_name
                        result = search_recursive(attr_obj, depth + 1, max_depth, visited, new_path)
                        if result is not None:
                            return result
                except (AttributeError, RuntimeError, TypeError):
                    continue

        # Check for get_model() method
        if hasattr(obj, 'get_model'):
            try:
                inner_model = obj.get_model()
                if inner_model is not None and id(inner_model) not in visited:
                    new_path = f"{path}.get_model()" if path else "get_model()"
                    result = search_recursive(inner_model, depth + 1, max_depth, visited, new_path)
                    if result is not None:
                        return result
            except (AttributeError, RuntimeError, TypeError):
                pass

        return None

    transformer = search_recursive(model, path="model")
    if transformer is not None:
        logger.info("  ✓ FOUND TRANSFORMER via recursive search!")
        return transformer

    # Final debug summary
    logger.error("=" * 80)
    logger.error("FAILED TO FIND FLUX TRANSFORMER")
    logger.error("=" * 80)
    logger.error("Model structure summary:")
    logger.error(f"  Model type: {type(model)}")
    logger.error(f"  Model attributes: {[a for a in dir(model) if not a.startswith('__')][:30]}")
    if hasattr(model, 'model'):
        logger.error(f"  model.model type: {type(model.model)}")
        logger.error(f"  model.model attributes: {[a for a in dir(model.model) if not a.startswith('__')][:30]}")
        if hasattr(model.model, 'diffusion_model'):
            flux_model = model.model.diffusion_model
            flux_type = str(type(flux_model))
            logger.error(f"  model.model.diffusion_model type: {flux_type}")
            logger.error(f"  Has attn_processors: {hasattr(flux_model, 'attn_processors')}")
            logger.error(f"  Has set_attn_processor: {hasattr(flux_model, 'set_attn_processor')}")

            # Check if it's a ComfyUI Flux wrapper
            if 'comfy.ldm.flux' in flux_type or 'comfy.model_base.Flux' in flux_type:
                logger.error("  Detected ComfyUI Flux wrapper")
                # Check for transformer attribute
                if hasattr(flux_model, 'transformer'):
                    transformer = flux_model.transformer
                    logger.error(f"  flux_model.transformer exists: {type(transformer)}")
                    logger.error(f"    Has attn_processors: {hasattr(transformer, 'attn_processors')}")
                    logger.error(f"    Has set_attn_processor: {hasattr(transformer, 'set_attn_processor')}")
                else:
                    logger.error("  flux_model.transformer attribute NOT FOUND")
                    # List all attributes that might be relevant
                    all_attrs = [attr for attr in dir(flux_model) if not attr.startswith('__')]
                    logger.error(f"  All non-private attributes ({len(all_attrs)} total):")
                    for attr in all_attrs[:50]:  # Show first 50
                        try:
                            attr_obj = getattr(flux_model, attr, None)
                            if attr_obj is not None and not isinstance(attr_obj, (str, int, float, bool, list, dict, tuple)):
                                logger.error(f"    {attr}: {type(attr_obj)}")
                        except:
                            pass
    logger.error("=" * 80)

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
        logger.info("=" * 80)
        logger.info("ApplyBokehAdapter.apply_adapter() called")
        logger.info(f"Adapter config: {adapter}")
        try:
            # Find the transformer
            logger.info("Step 1: Finding FLUX transformer...")
            transformer = find_flux_transformer(model)

            if transformer is None:
                logger.error("Transformer is None - search failed!")
                raise Exception("Could not find FLUX transformer in model. Make sure you're using a FLUX model.")

            logger.info(f"✓ Transformer found: {type(transformer)}")
            logger.info(f"  Transformer ID: {id(transformer)}")

            # Check if transformer has attn_processors (required for adapter)
            logger.info("Step 2: Verifying transformer has attn_processors...")
            if not hasattr(transformer, 'attn_processors'):
                logger.error("Transformer does not have attn_processors attribute!")
                raise Exception("Model transformer does not have attn_processors. This may not be a FLUX model.")

            logger.info("✓ Transformer has attn_processors")
            logger.debug(f"  attn_processors type: {type(transformer.attn_processors)}")
            if hasattr(transformer.attn_processors, '__len__'):
                logger.debug(f"  Number of processors: {len(transformer.attn_processors)}")

            # Check for set_attn_processor method
            if not hasattr(transformer, 'set_attn_processor'):
                logger.error("Transformer does not have set_attn_processor method!")
                raise Exception("Model transformer does not have set_attn_processor method. This may not be a FLUX model.")
            logger.info("✓ Transformer has set_attn_processor method")

            # Load the adapter if not already loaded
            adapter_name = adapter["adapter_name"]
            logger.info(f"Step 3: Loading adapter from {adapter_name}...")
            if adapter["adapter"] is None:
                logger.info(f"  Adapter not yet loaded, loading now...")
                try:
                    adapter_obj = BokehFluxControlAdapter.from_pretrained(
                        adapter_name,
                        base_model=transformer,
                    )
                    logger.info("  ✓ Adapter loaded from HuggingFace")

                    # Move to same device as transformer
                    logger.info("Step 4: Moving adapter to same device/dtype as transformer...")
                    try:
                        device = next(transformer.parameters()).device
                        dtype = next(transformer.parameters()).dtype
                        logger.info(f"  Transformer device: {device}, dtype: {dtype}")
                        adapter_obj = adapter_obj.to(device=device, dtype=dtype)
                        logger.info("  ✓ Adapter moved to transformer device/dtype")
                    except Exception as e:
                        logger.warning(f"  Error moving adapter to device: {e}")
                        logger.info("  Continuing anyway...")

                    adapter["adapter"] = adapter_obj
                    logger.info("✓ Bokeh adapter loaded and configured successfully")
                except Exception as e:
                    logger.error(f"  Failed to load adapter: {e}", exc_info=True)
                    raise
            else:
                logger.info("  Adapter already loaded, reusing existing instance")

            adapter_obj = adapter["adapter"]
            logger.info(f"  Adapter object: {type(adapter_obj)}")

            # Set scales
            logger.info("Step 5: Setting adapter scales...")
            bokeh_scale = adapter.get("bokeh_scale", 1.0)
            lora_scale = adapter.get("lora_scale", 1.0)
            logger.info(f"  bokeh_scale: {bokeh_scale}, lora_scale: {lora_scale}")

            if bokeh_scale != 1.0:
                logger.info(f"  Setting bokeh_scale to {bokeh_scale}")
                adapter_obj.set_bokeh_scale(bokeh_scale)
            if lora_scale != 1.0:
                logger.info(f"  Setting lora_scale to {lora_scale}")
                adapter_obj.set_lora_scale(lora_scale)

            # Store adapter in model for use during sampling
            logger.info("Step 6: Storing adapter in model...")
            model.bokeh_adapter = adapter_obj
            model.bokeh_adapter_config = adapter
            model._bokeh_transformer = transformer
            logger.info("✓ Adapter stored in model")
            logger.info("=" * 80)
            logger.info("✓ Bokeh adapter applied successfully!")
            logger.info("=" * 80)

            return (model,)
        except Exception as e:
            logger.error("=" * 80)
            logger.error("FAILED TO APPLY BOKEH ADAPTER")
            logger.error("=" * 80)
            logger.error(f"Error: {e}", exc_info=True)
            logger.error("=" * 80)
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
