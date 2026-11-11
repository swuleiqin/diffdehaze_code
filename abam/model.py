import math

import torch
from torch import nn
from torch.nn import functional as F


ILLUMINATION_METHOD = "pixel_rgb_max_v1"


class IlluminationDecomposition(nn.Module):

    def __init__(self, epsilon=1e-6):
        super().__init__()
        if not math.isfinite(epsilon) or not 0 < epsilon < 1:
            raise ValueError("epsilon must be finite and in (0,1)")
        self.epsilon = epsilon

    def forward(self, image):

        illumination = image.amax(dim=1, keepdim=True).clamp(self.epsilon, 1.0)
        reflectance = image / illumination
        return reflectance, illumination


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.weights = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1),
            nn.SiLU(), nn.Conv2d(hidden, channels, 1), nn.Sigmoid(),
        )

    def forward(self, features):
        return features * self.weights(features)


class ResidualAttentionBlock(nn.Module):

    def __init__(self, channels, reduction, residual_scale):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            ChannelAttention(channels, reduction),
        )
        self.scale = residual_scale

    def forward(self, features):
        return features + self.scale * self.body(features)


class DilatedContext(nn.Module):

    def __init__(self, channels, dilations, reduction, residual_scale):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=d, dilation=d, groups=channels),
                nn.SiLU(), nn.Conv2d(channels, channels, 1),
            ) for d in dilations
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(len(dilations) * channels, channels, 1), nn.SiLU(),
            ChannelAttention(channels, reduction),
        )
        self.scale = residual_scale

    def forward(self, features):
        context = torch.cat([branch(features) for branch in self.branches], dim=1)
        return features + self.scale * self.fuse(context)


class MultiScaleBrightnessBackbone(nn.Module):

    def __init__(self, channels, multipliers, blocks, dilations, reduction,
                 global_hidden, residual_scale):
        super().__init__()
        widths = [channels * multiplier for multiplier in multipliers]

        def stack(width, count):
            return nn.Sequential(*[
                ResidualAttentionBlock(width, reduction, residual_scale) for _ in range(count)
            ])

        self.structure_stem = nn.Sequential(nn.Conv2d(4, widths[0], 3, padding=1), nn.SiLU())
        self.illumination_stem = nn.Sequential(nn.Conv2d(1, widths[0], 3, padding=1), nn.SiLU())
        self.stem_fusion = nn.Conv2d(2 * widths[0], widths[0], 1)
        self.encoders = nn.ModuleList([stack(w, n) for w, n in zip(widths, blocks)])
        self.downsamples = nn.ModuleList([
            nn.Sequential(nn.Conv2d(a, b, 3, stride=2, padding=1), nn.SiLU())
            for a, b in zip(widths[:-1], widths[1:])
        ])
        self.context = DilatedContext(widths[-1], dilations, reduction, residual_scale)

        self.global_features = nn.Sequential(
            nn.Linear(18, global_hidden), nn.SiLU(),
            nn.Linear(global_hidden, global_hidden), nn.SiLU(),
        )
        self.global_affine = nn.Linear(global_hidden, 2 * widths[-1])
        self.global_gamma = nn.Linear(global_hidden, 1)
        nn.init.zeros_(self.global_affine.weight)
        nn.init.zeros_(self.global_affine.bias)
        nn.init.zeros_(self.global_gamma.weight)
        nn.init.zeros_(self.global_gamma.bias)

        self.up_projections = nn.ModuleList([
            nn.Conv2d(widths[i + 1], widths[i], 1) for i in reversed(range(len(widths) - 1))
        ])
        self.skip_fusions = nn.ModuleList([
            nn.Conv2d(2 * widths[i], widths[i], 1) for i in reversed(range(len(widths) - 1))
        ])
        self.decoders = nn.ModuleList([
            stack(widths[i], blocks[i]) for i in reversed(range(len(widths) - 1))
        ])

    def forward(self, reflectance, illumination):
        structure = self.structure_stem(torch.cat([reflectance, illumination], dim=1))
        lighting = self.illumination_stem(illumination)
        features = self.stem_fusion(torch.cat([structure, lighting], dim=1))
        skips = []
        for level, encoder in enumerate(self.encoders):
            features = encoder(features)
            if level < len(self.downsamples):
                skips.append(features)
                features = self.downsamples[level](features)
        features = self.context(features)
        flat = illumination.flatten(1)
        statistics = torch.cat([
            F.adaptive_avg_pool2d(illumination, (4, 4)).flatten(1),
            flat.mean(dim=1, keepdim=True),
            (flat.var(dim=1, keepdim=True, unbiased=False) + 1e-8).sqrt(),
        ], dim=1)
        global_features = self.global_features(statistics)
        gain, bias = self.global_affine(global_features).chunk(2, dim=1)
        features = (features * (1 + 0.1 * gain.tanh()[:, :, None, None])
                    + 0.1 * bias.tanh()[:, :, None, None])
        for skip, projection, fusion, decoder in zip(
            reversed(skips), self.up_projections, self.skip_fusions, self.decoders,
        ):

            features = F.interpolate(features, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            features = projection(features)
            features = decoder(fusion(torch.cat([features, skip], dim=1)))
        return features, self.global_gamma(global_features)[:, :, None, None]


class AdaptiveBrightnessAdjustmentModule(nn.Module):
    def __init__(self, channels=32, depth=3, attention_reduction=4,
                 illumination_method=ILLUMINATION_METHOD, epsilon=1e-6,
                 gamma_min=0.05, gamma_init=0.95, architecture="lightweight",
                 channel_multipliers=(1, 2, 4), blocks_per_level=(2, 2, 3),
                 context_dilations=(1, 2, 4), global_hidden=64, residual_scale=0.2):
        super().__init__()
        if channels < 1 or depth < 1 or attention_reduction < 1:
            raise ValueError("channels, depth and attention_reduction must be positive")
        if not 0 < gamma_min < gamma_init < 1:
            raise ValueError("Require 0 < gamma_min < gamma_init < 1")
        if illumination_method != ILLUMINATION_METHOD:
            raise ValueError(f"Only {ILLUMINATION_METHOD} decomposition is supported; retrain old ABAM weights")
        self.illumination_method = illumination_method
        self.decomposition = IlluminationDecomposition(epsilon=epsilon)
        self.gamma_min = float(gamma_min)
        self.architecture = architecture
        if architecture == "lightweight":

            structure = [nn.Conv2d(4, channels, 3, padding=1), nn.SiLU()]
            for _ in range(depth - 1):
                structure.extend([nn.Conv2d(channels, channels, 3, padding=1), nn.SiLU()])
            self.structure = nn.Sequential(*structure)
            self.illumination_features = nn.Sequential(
                nn.Conv2d(1, channels, 3, padding=1), nn.SiLU(),
            )
            self.fusion = nn.Sequential(
                nn.Conv2d(2 * channels, channels, 1), nn.SiLU(),
                ChannelAttention(channels, attention_reduction),
                nn.Conv2d(channels, channels, 3, padding=1), nn.SiLU(),
            )
        elif architecture == "multiscale":
            if (len(channel_multipliers) < 2 or len(channel_multipliers) != len(blocks_per_level)
                    or channel_multipliers[0] != 1):
                raise ValueError("Use >=2 levels, matching blocks, and first channel multiplier=1")
            for values in (channel_multipliers, blocks_per_level, context_dilations):
                if not values or any(type(value) is not int or value < 1 for value in values):
                    raise ValueError("Multipliers, block counts and dilations must be positive integers")
            if type(global_hidden) is not int or global_hidden < 1:
                raise ValueError("global_hidden must be a positive integer")
            if not math.isfinite(residual_scale) or not 0 < residual_scale <= 1:
                raise ValueError("residual_scale must be in (0,1]")
            self.backbone = MultiScaleBrightnessBackbone(
                channels, channel_multipliers, blocks_per_level, context_dilations,
                attention_reduction, global_hidden, residual_scale,
            )
        else:
            raise ValueError("architecture must be lightweight or multiscale")
        self.gamma_head = nn.Conv2d(channels, 1, 3, padding=1)

        nn.init.zeros_(self.gamma_head.weight)
        probability = (gamma_init - gamma_min) / (1.0 - gamma_min)
        nn.init.constant_(self.gamma_head.bias, math.log(probability / (1 - probability)))

    def forward(self, image):
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("ABAM expects BCHW RGB in [0,1], not SD latents")
        reflectance, illumination = self.decomposition(image)
        if self.architecture == "lightweight":
            structure = self.structure(torch.cat([reflectance, illumination], dim=1))
            exposure = self.illumination_features(illumination)
            features = self.fusion(torch.cat([structure, exposure], dim=1))
            logits = self.gamma_head(features)
        else:
            features, global_gamma = self.backbone(reflectance, illumination)
            logits = self.gamma_head(features) + global_gamma
        gamma = self.gamma_min + (1 - self.gamma_min) * torch.sigmoid(logits)
        corrected_illumination = illumination.pow(gamma)
        output = reflectance * corrected_illumination

        return {
            "image": output, "reflectance": reflectance,
            "illumination": illumination, "corrected_illumination": corrected_illumination,
            "gamma": gamma,
        }
