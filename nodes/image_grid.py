import numpy as np
import torch

from ..tilemap_core.scene_render import label_grid


class ImageGridLabeled:
    """Tile an IMAGE batch into one grid image, optionally labelling each cell with its 1-based index."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "columns": ("INT", {"default": 2, "min": 1, "max": 16, "step": 1}),
                "padding": ("INT", {"default": 8, "min": 0, "max": 256, "step": 1}),
                "background": (["black", "white", "gray"], {"default": "black"}),
                "show_index": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("grid",)
    FUNCTION = "make_grid"
    CATEGORY = "custom_comfy_nodes/image"

    def make_grid(self, images, columns, padding, background, show_index):
        arr = (images[..., :3].clamp(0, 1) * 255).round().byte().cpu().numpy()
        grid = label_grid(arr, columns, padding, background, show_index)
        return (torch.from_numpy(grid.astype(np.float32) / 255.0).unsqueeze(0),)


NODE_CLASS_MAPPINGS = {"CCN_ImageGridLabeled": ImageGridLabeled}
NODE_DISPLAY_NAME_MAPPINGS = {"CCN_ImageGridLabeled": "Image Grid (Labeled)"}
