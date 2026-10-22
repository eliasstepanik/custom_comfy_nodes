import math

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

_BG = {"black": (0, 0, 0), "white": (255, 255, 255), "gray": (128, 128, 128)}


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
        b, h, w = images.shape[:3]
        rows = math.ceil(b / columns)
        grid = Image.new(
            "RGB",
            (columns * w + (columns + 1) * padding, rows * h + (rows + 1) * padding),
            _BG[background],
        )
        draw = ImageDraw.Draw(grid)
        font = ImageFont.load_default()
        for i in range(b):
            x = padding + (i % columns) * (w + padding)
            y = padding + (i // columns) * (h + padding)
            arr = (images[i, :, :, :3].clamp(0, 1) * 255).round().byte().cpu().numpy()
            grid.paste(Image.fromarray(arr), (x, y))
            if show_index:
                label = str(i + 1)
                right, bottom = draw.textbbox((x + 2, y + 2), label, font=font)[2:]
                draw.rectangle((x, y, right + 2, bottom + 2), fill=(0, 0, 0))
                draw.text((x + 2, y + 2), label, fill=(255, 255, 255), font=font)
        out = torch.from_numpy(np.asarray(grid).astype(np.float32) / 255.0).unsqueeze(0)
        return (out,)


NODE_CLASS_MAPPINGS = {"CCN_ImageGridLabeled": ImageGridLabeled}
NODE_DISPLAY_NAME_MAPPINGS = {"CCN_ImageGridLabeled": "Image Grid (Labeled)"}
