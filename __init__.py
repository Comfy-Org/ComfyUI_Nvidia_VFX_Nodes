from enum import Enum
import math
from typing import TypedDict

import nvvfx
import torch
import torch.nn.functional as F
from typing_extensions import override

import comfy.model_management
from comfy_api.latest import ComfyExtension, io


MAX_BATCH_PIXELS = 1024 * 1024 * 16
VSR_QUALITY_OPTIONS = list(nvvfx.QualityLevel.__members__)

PQ_M1 = 2610 / 16384
PQ_M2 = 2523 / 4096 * 128
PQ_C1 = 3424 / 4096
PQ_C2 = 2413 / 4096 * 32
PQ_C3 = 2392 / 4096 * 32

HLG_A = 0.17883277
HLG_B = 1 - 4 * HLG_A
HLG_C = 0.5 - HLG_A * math.log(4 * HLG_A)


class UpscaleType(str, Enum):
    SCALE_BY = "scale by multiplier"
    TARGET_DIMENSIONS = "target dimensions"


class FrameGenerationType(str, Enum):
    MULTIPLIER = "frame rate multiplier"
    TIMESTEP = "specific timestep"


class ImageEncodingType(str, Enum):
    RGB8 = "8-bit RGB"
    RGB10A2 = "10-bit RGB (RGB10A2)"


class HDRColorSpace(str, Enum):
    HLG = "HDR"
    PQ = "HDR PQ"


def get_effect_device() -> tuple[torch.device, int]:
    device = comfy.model_management.get_torch_device()
    if device.type != "cuda":
        raise RuntimeError("NVIDIA RTX Video Effects require a CUDA device.")
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    return device, device_index


def pack_rgb10a2(image: torch.Tensor) -> torch.Tensor:
    rgb = image.clamp(0.0, 1.0).mul(1023.0).round().to(torch.int32)
    packed = rgb[..., 0] | (rgb[..., 1] << 10) | (rgb[..., 2] << 20) | -1073741824
    return packed.view(torch.uint32).contiguous()


def unpack_rgb10a2(packed: torch.Tensor) -> torch.Tensor:
    packed = packed.view(torch.int32)
    red = packed.bitwise_and(0x3FF)
    green = packed.bitwise_right_shift(10).bitwise_and(0x3FF)
    blue = packed.bitwise_right_shift(20).bitwise_and(0x3FF)
    return torch.stack((red, green, blue), dim=-1).to(torch.float32).div_(1023.0)


def pq_to_hlg(image: torch.Tensor, peak_luminance: float) -> torch.Tensor:
    pq = image.clamp(0.0, 1.0).pow(1.0 / PQ_M2)
    display_light = ((pq.sub(PQ_C1).clamp_min(0.0) / (PQ_C2 - PQ_C3 * pq)) ** (1.0 / PQ_M1)) * 10000.0
    display_light.clamp_(max=peak_luminance)

    display_luminance = 0.2627 * display_light[..., 0] + 0.6780 * display_light[..., 1] + 0.0593 * display_light[..., 2]
    system_gamma = 1.2 + 0.42 * math.log10(peak_luminance / 1000.0)
    luminance_ratio = (display_luminance / peak_luminance).clamp_min(torch.finfo(display_light.dtype).eps)
    scene_light = (display_light / peak_luminance) * luminance_ratio.unsqueeze(-1).pow((1.0 - system_gamma) / system_gamma)
    scene_light.clamp_(0.0, 1.0)

    low = torch.sqrt(3.0 * scene_light)
    high = HLG_A * torch.log((12.0 * scene_light - HLG_B).clamp_min(torch.finfo(scene_light.dtype).eps)) + HLG_C
    return torch.where(scene_light <= 1.0 / 12.0, low, high)


def prepare_frame(image: torch.Tensor, device: torch.device, image_encoding: ImageEncodingType) -> torch.Tensor:
    image = image.to(device=device, dtype=torch.float32)
    if image_encoding == ImageEncodingType.RGB10A2:
        return pack_rgb10a2(image)
    return image.movedim(-1, 0).contiguous()


def copy_effect_output(dlpack_output, destination: torch.Tensor, image_encoding: ImageEncodingType) -> None:
    output = torch.from_dlpack(dlpack_output).clone()
    if image_encoding == ImageEncodingType.RGB10A2:
        output = unpack_rgb10a2(output)
    else:
        output = output.movedim(0, -1)
    destination.copy_(output)


class RTXVideoSuperResolution(io.ComfyNode):
    class UpscaleTypedDict(TypedDict):
        resize_type: UpscaleType
        scale: float
        width: int
        height: int

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="RTXVideoSuperResolution",
            display_name="RTX Video Super Resolution",
            category="image/upscaling",
            description="Upscale, denoise, or deblur images with NVIDIA Video Effects.",
            search_aliases=["rtx", "nvidia", "upscale", "super resolution", "vsr", "denoise", "deblur"],
            inputs=[
                io.Image.Input("images"),
                io.DynamicCombo.Input(
                    "resize_type",
                    tooltip="Choose to scale by a multiplier or to exact target dimensions. Denoise and deblur modes keep the input dimensions.",
                    options=[
                        io.DynamicCombo.Option(UpscaleType.SCALE_BY, [
                            io.Float.Input("scale", default=2.0, min=1.0, max=4.0, step=0.01, tooltip="Scale factor (e.g., 2.0 doubles the size)."),
                        ]),
                        io.DynamicCombo.Option(UpscaleType.TARGET_DIMENSIONS, [
                            io.Int.Input("width", default=1920, min=64, max=8192, step=8, tooltip="Target width in pixels."),
                            io.Int.Input("height", default=1080, min=64, max=8192, step=8, tooltip="Target height in pixels."),
                        ]),
                    ],
                ),
                io.Combo.Input(
                    "quality",
                    options=VSR_QUALITY_OPTIONS,
                    default="ULTRA",
                    tooltip="Select the SDK model. DENOISE and DEBLUR process at the original resolution; HIGHBITRATE modes preserve clean sources.",
                ),
                io.Float.Input("strength", default=1.0, min=0.0, max=1.0, step=0.01, tooltip="Effect strength."),
                io.Combo.Input(
                    "image_encoding",
                    options=[encoding.value for encoding in ImageEncodingType],
                    default=ImageEncodingType.RGB8,
                    tooltip="SDK pixel encoding. 10-bit preserves more precision but does not by itself convert SDR content to HDR.",
                ),
            ],
            outputs=[
                io.Image.Output("upscaled_images"),
            ],
        )

    @classmethod
    def execute(
        cls,
        images: torch.Tensor,
        resize_type: UpscaleTypedDict,
        quality: str,
        strength: float = 1.0,
        image_encoding: str = ImageEncodingType.RGB8,
    ) -> io.NodeOutput:
        alpha = images[..., 3:4] if images.shape[-1] > 3 else None
        images = images[..., :3]
        b, h, w, c = images.shape

        selected_type = resize_type["resize_type"]
        if selected_type == UpscaleType.SCALE_BY:
            scale = resize_type["scale"]
            output_width = int(w * scale)
            output_height = int(h * scale)
        elif selected_type == UpscaleType.TARGET_DIMENSIONS:
            output_width = resize_type["width"]
            output_height = resize_type["height"]
        else:
            raise ValueError(f"Unsupported resize type: {selected_type}")

        if quality.startswith(("DENOISE_", "DEBLUR_")):
            output_width = w
            output_height = h
        else:
            output_width = max(8, round(output_width / 8) * 8)
            output_height = max(8, round(output_height / 8) * 8)

        output = torch.empty((b, output_height, output_width, c), device=images.device, dtype=images.dtype)
        device, device_index = get_effect_device()
        batch_size = max(1, MAX_BATCH_PIXELS // (output_width * output_height))
        selected_encoding = ImageEncodingType(image_encoding)

        with torch.cuda.device(device), nvvfx.VideoSuperRes(
            quality=nvvfx.QualityLevel[quality],
            strength=strength,
            device=device_index,
            image_encoding=nvvfx.ImageEncoding.RGB10A2 if selected_encoding == ImageEncodingType.RGB10A2 else nvvfx.ImageEncoding.RGB8,
        ) as effect:
            effect.output_width = output_width
            effect.output_height = output_height
            effect.load()

            for i in range(0, images.shape[0], batch_size):
                batch = images[i:i + batch_size]
                batch_cuda = batch.to(device=device, dtype=torch.float32)
                if selected_encoding == ImageEncodingType.RGB10A2:
                    batch_cuda = pack_rgb10a2(batch_cuda)
                else:
                    batch_cuda = batch_cuda.movedim(-1, 1).contiguous()
                for j, frame in enumerate(batch_cuda):
                    copy_effect_output(effect.run(frame).image, output[i + j], selected_encoding)

        if alpha is not None:
            upscaled_alpha = F.interpolate(
                alpha.movedim(-1, 1),
                size=(output_height, output_width),
                mode="bilinear",
                align_corners=False,
            ).movedim(1, -1)
            output = torch.cat((output, upscaled_alpha), dim=-1)

        return io.NodeOutput(output)


class RTXVideoHDR(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="RTXTrueHDR",
            display_name="RTX Video HDR",
            category="image/enhancement",
            description="Convert SDR images to HLG or PQ HDR image data with NVIDIA RTX Video HDR.",
            search_aliases=["rtx", "nvidia", "video hdr", "true hdr", "truehdr", "hdr10", "hlg", "pq", "sdr to hdr"],
            inputs=[
                io.Image.Input("images"),
                io.Int.Input("contrast", default=100, min=0, max=200),
                io.Int.Input("saturation", default=100, min=0, max=200),
                io.Int.Input("middle_gray", default=50, min=10, max=100, tooltip="Middle-grey reference level."),
                io.Int.Input("luminance", default=650, min=400, max=2000, tooltip="Target HDR display peak luminance in nits."),
                io.Boolean.Input("debanding", default=True, tooltip="Apply the SDK's DL-Debander pass."),
                io.Combo.Input(
                    "output_colorspace",
                    options=[colorspace.value for colorspace in HDRColorSpace],
                    default=HDRColorSpace.HLG,
                    tooltip="HDR outputs BT.2020/HLG using the selected peak luminance. HDR PQ returns the SDK's BT.2020/PQ signal unchanged.",
                ),
            ],
            outputs=[
                io.Image.Output("hdr_images", tooltip="Normalized RGB values using the selected HDR colorspace."),
            ],
        )

    @classmethod
    def execute(
        cls,
        images: torch.Tensor,
        contrast: int,
        saturation: int,
        middle_gray: int,
        luminance: int,
        debanding: bool,
        output_colorspace: str = HDRColorSpace.HLG,
    ) -> io.NodeOutput:
        alpha = images[..., 3:4] if images.shape[-1] > 3 else None
        images = images[..., :3]
        device, device_index = get_effect_device()
        output = torch.empty(images.shape, device=images.device, dtype=torch.float32)
        selected_colorspace = HDRColorSpace(output_colorspace)

        with torch.cuda.device(device), nvvfx.TrueHDR(
            contrast=contrast,
            saturation=saturation,
            middle_gray=middle_gray,
            luminance=luminance,
            debanding_off=int(not debanding),
            device=device_index,
        ) as effect:
            effect.load()
            for i, image in enumerate(images):
                result = effect.run(prepare_frame(image, device, ImageEncodingType.RGB8))
                hdr_image = unpack_rgb10a2(torch.from_dlpack(result.image).clone())
                if selected_colorspace == HDRColorSpace.HLG:
                    hdr_image = pq_to_hlg(hdr_image, luminance)
                output[i].copy_(hdr_image)

        if alpha is not None:
            output = torch.cat((output, alpha.to(dtype=output.dtype)), dim=-1)

        return io.NodeOutput(output)


class RTXVideoFrameGeneration(io.ComfyNode):
    class GenerationTypedDict(TypedDict):
        generation_type: FrameGenerationType
        multiplier: int
        timestep: float

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="RTXVideoFrameGeneration",
            display_name="RTX Video Frame Generation",
            category="video",
            description="Generate intermediate video frames with NVIDIA Video Effects.",
            search_aliases=["rtx", "nvidia", "frame generation", "interpolation", "slow motion", "vfi"],
            inputs=[
                io.Image.Input("images"),
                io.DynamicCombo.Input(
                    "generation_type",
                    options=[
                        io.DynamicCombo.Option(FrameGenerationType.MULTIPLIER, [
                            io.Int.Input("multiplier", default=2, min=2, max=16, tooltip="Output frame-rate multiplier."),
                        ]),
                        io.DynamicCombo.Option(FrameGenerationType.TIMESTEP, [
                            io.Float.Input("timestep", default=0.5, min=0.01, max=0.99, step=0.01, tooltip="Position between each adjacent input pair."),
                        ]),
                    ],
                ),
                io.Combo.Input("mode", options=["LOW", "MEDIUM", "HIGH"], default="MEDIUM", tooltip="Frame-generation quality mode."),
                io.Boolean.Input("automatic_shot_change_detection", default=True),
                io.Boolean.Input("shot_change", default=False, tooltip="Mark every submitted pair as a shot change. Useful when processing a single pair."),
                io.Combo.Input(
                    "image_encoding",
                    options=[encoding.value for encoding in ImageEncodingType],
                    default=ImageEncodingType.RGB8,
                    tooltip="SDK pixel encoding. 10-bit preserves more precision but does not by itself convert SDR content to HDR.",
                ),
            ],
            outputs=[
                io.Image.Output("interpolated_images"),
            ],
        )

    @classmethod
    def execute(
        cls,
        images: torch.Tensor,
        generation_type: GenerationTypedDict,
        mode: str,
        automatic_shot_change_detection: bool,
        shot_change: bool,
        image_encoding: str = ImageEncodingType.RGB8,
    ) -> io.NodeOutput:
        images = images[..., :3]
        if images.shape[0] < 2:
            return io.NodeOutput(images)

        selected_type = generation_type["generation_type"]
        if selected_type == FrameGenerationType.MULTIPLIER:
            multiplier = generation_type["multiplier"]
            generated_per_pair = multiplier - 1
        elif selected_type == FrameGenerationType.TIMESTEP:
            multiplier = 2
            generated_per_pair = 1
        else:
            raise ValueError(f"Unsupported frame generation type: {selected_type}")

        output_count = (images.shape[0] - 1) * (generated_per_pair + 1) + 1
        output = torch.empty((output_count, *images.shape[1:]), device=images.device, dtype=images.dtype)
        output[0].copy_(images[0])

        _, height, width, _ = images.shape
        device, device_index = get_effect_device()
        selected_encoding = ImageEncodingType(image_encoding)
        with torch.cuda.device(device), nvvfx.VideoFrameGeneration(
            mode=nvvfx.VideoFrameGeneration.Mode[mode],
            automatic_shot_change_detection_enabled=automatic_shot_change_detection,
            device=device_index,
            image_encoding=nvvfx.ImageEncoding.RGB10A2 if selected_encoding == ImageEncodingType.RGB10A2 else nvvfx.ImageEncoding.RGB8,
        ) as effect:
            effect.input_width = width
            effect.input_height = height
            effect.frame_multiplier = multiplier
            effect.load()

            output_index = 1
            previous_frame = prepare_frame(images[0], device, selected_encoding)
            for current_image in images[1:]:
                current_frame = prepare_frame(current_image, device, selected_encoding)
                if selected_type == FrameGenerationType.MULTIPLIER:
                    for frame_index in range(1, multiplier):
                        result = effect.run(
                            previous_frame,
                            current_frame,
                            frame_index=frame_index,
                            shot_change=shot_change,
                        )
                        copy_effect_output(result.image, output[output_index], selected_encoding)
                        output_index += 1
                else:
                    result = effect.run_at_timestep(
                        previous_frame,
                        current_frame,
                        generation_type["timestep"],
                        shot_change=shot_change,
                    )
                    copy_effect_output(result.image, output[output_index], selected_encoding)
                    output_index += 1

                output[output_index].copy_(current_image)
                output_index += 1
                previous_frame = current_frame

        return io.NodeOutput(output)


class NVVFXVideoExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            RTXVideoSuperResolution,
            RTXVideoHDR,
            RTXVideoFrameGeneration,
        ]


async def comfy_entrypoint() -> NVVFXVideoExtension:
    return NVVFXVideoExtension()

# Registry scanner compatibility.
if False:
    NODE_CLASS_MAPPINGS = {
        "RTXVideoSuperResolution": RTXVideoSuperResolution,
        "RTXTrueHDR": RTXVideoHDR,
        "RTXVideoFrameGeneration": RTXVideoFrameGeneration,
    }
