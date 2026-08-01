"""Adaptive frame preprocessing.

Keeps detection stable across day, night, rain, light fog and backlight by
normalising exposure and local contrast before inference. The pipeline is
content-adaptive: it measures each frame and only applies what that frame
needs, so daylight images are left essentially untouched.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from config import Config

logger = logging.getLogger(__name__)


class FramePreprocessor:
    """CLAHE + gamma + sharpen + auto brightness/contrast, applied adaptively."""

    def __init__(
        self,
        enabled: bool = Config.PREPROCESS_ENABLED,
        clip_limit: float = Config.CLAHE_CLIP_LIMIT,
        tile_grid: int = Config.CLAHE_TILE_GRID,
    ):
        self.enabled = enabled
        self.clip_limit = clip_limit
        self.tile_grid = tile_grid
        self._clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit, tileGridSize=(self.tile_grid, self.tile_grid)
        )
        # Smoothed scene brightness; prevents flicker between frames.
        self._brightness_ema: Optional[float] = None
        self.last_profile = 'day'
        self.last_brightness = 0.0
        self.last_gamma = 1.0

    # ------------------------------------------------------------------
    def configure(self, enabled: Optional[bool] = None, clip_limit: Optional[float] = None) -> None:
        if enabled is not None:
            self.enabled = bool(enabled)
        if clip_limit is not None:
            self.clip_limit = float(np.clip(clip_limit, 0.5, 8.0))
            self._clahe = cv2.createCLAHE(
                clipLimit=self.clip_limit, tileGridSize=(self.tile_grid, self.tile_grid)
            )

    # ------------------------------------------------------------------
    def process(self, frame: np.ndarray) -> np.ndarray:
        """Return an enhanced copy of ``frame``; never raises."""
        if frame is None or not self.enabled:
            return frame
        try:
            return self._process(frame)
        except Exception as error:
            logger.debug('Preprocessing skipped: %s', error)
            return frame

    def _process(self, frame: np.ndarray) -> np.ndarray:
        """Enhance only what the frame actually needs.

        Each step below was A/B tested against the detector on this camera;
        anything that reduced detections or confidence (percentile stretching
        a well-exposed frame, unsharp masking, denoising) is applied only in
        the regime where it measurably helps, or not at all.
        """
        # Work in LAB so only luminance is touched and colours stay natural.
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lightness, a_channel, b_channel = cv2.split(lab)

        brightness = float(lightness.mean())
        self._brightness_ema = (
            brightness if self._brightness_ema is None
            else 0.85 * self._brightness_ema + 0.15 * brightness
        )
        smoothed = self._brightness_ema
        self.last_brightness = round(smoothed, 1)
        contrast = float(lightness.std())

        is_night = smoothed < Config.NIGHT_BRIGHTNESS_THRESHOLD
        is_backlit = contrast > 70.0 and smoothed > 110.0
        # Genuinely washed out: haze, light fog, rain. A normal traffic scene
        # sits well above this, so daylight is never treated as fog.
        is_flat = contrast < 22.0 and not is_night
        self.last_profile = (
            'night' if is_night else 'backlight' if is_backlit else 'fog' if is_flat else 'day'
        )

        # Night + oncoming-headlight glare: measured directly against real
        # night footage from this deployment (glare-heavy urban intersection)
        # that CLAHE and gamma-brightening — individually and combined — cut
        # raw detections by roughly a quarter to a third per frame (e.g.
        # 16-17 -> 11-14 boxes on identical frames), almost certainly because
        # boosting local contrast/brightness on an already-blown-out glare
        # region pushes marginal, partially-visible vehicle silhouettes the
        # rest of the way into pure white, destroying the little shape
        # information the detector had to work with. Unlike the other steps
        # below (already A/B tested and kept because they measurably help),
        # this one measurably hurt, so night frames are passed through
        # unmodified rather than enhanced.
        if is_night:
            self.last_gamma = 1.0
            return frame

        # 1) Local contrast (CLAHE). Helps in the remaining (non-night)
        #    regimes; slightly stronger when washed out. (Night already
        #    returned above.)
        clip = self.clip_limit
        if is_flat or is_backlit:
            clip = min(8.0, self.clip_limit * 1.3)
        clahe = (
            self._clahe if abs(clip - self.clip_limit) < 1e-6
            else cv2.createCLAHE(clipLimit=clip, tileGridSize=(self.tile_grid, self.tile_grid))
        )
        lightness = clahe.apply(lightness)

        # 2) Gamma correction (auto brightness) for genuinely over-exposed
        #    scenes. (Dark/night scenes already returned above.)
        gamma = 1.0
        if smoothed > 190.0:
            gamma = self._auto_gamma(float(lightness.mean()))
            if abs(gamma - 1.0) > 0.03:
                lightness = self._apply_gamma(lightness, gamma)
        self.last_gamma = round(gamma, 3)

        # 3) Auto contrast: percentile stretch, only for washed-out frames.
        #    On a well-exposed frame this costs confidence, so it stays off.
        if is_flat or is_backlit:
            lightness = self._auto_levels(lightness)

        merged = cv2.merge((lightness, a_channel, b_channel))
        result = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

        # 4) Mild unsharp mask to recover edges of small/distant vehicles in
        #    hazy conditions. Never at night, where it amplifies sensor noise.
        if is_flat:
            result = self._unsharp(result, 0.3)
        return result

    # ------------------------------------------------------------------
    @staticmethod
    def _auto_gamma(mean_value: float, target: float = 128.0) -> float:
        """Exponent that moves the mean luminance towards ``target``.

        Solves ``(mean/255) ** gamma == target/255``, so gamma < 1 brightens
        a dark frame and gamma > 1 tames an over-exposed one.
        """
        mean_value = max(1.0, min(254.0, mean_value))
        gamma = float(np.log(target / 255.0) / np.log(mean_value / 255.0))
        return float(np.clip(gamma, 0.35, 2.2))

    @staticmethod
    def _apply_gamma(channel: np.ndarray, gamma: float) -> np.ndarray:
        """Apply ``out = (in/255) ** gamma * 255`` through a lookup table."""
        exponent = max(gamma, 1e-6)
        table = np.array(
            [((i / 255.0) ** exponent) * 255 for i in range(256)], dtype=np.uint8
        )
        return cv2.LUT(channel, table)

    @staticmethod
    def _auto_levels(
        channel: np.ndarray, low_percentile: float = 1.0, high_percentile: float = 99.0
    ) -> np.ndarray:
        """Percentile-based contrast stretch, robust to specular highlights."""
        low, high = np.percentile(channel, (low_percentile, high_percentile))
        if high - low < 12:  # already well spread, or nearly flat: leave it
            return channel
        scale = 255.0 / (high - low)
        stretched = (channel.astype(np.float32) - low) * scale
        return np.clip(stretched, 0, 255).astype(np.uint8)

    @staticmethod
    def _unsharp(image: np.ndarray, amount: float) -> np.ndarray:
        blurred = cv2.GaussianBlur(image, (0, 0), 1.2)
        return cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)

    # ------------------------------------------------------------------
    def stats(self) -> dict:
        return {
            'enabled': self.enabled,
            'profile': self.last_profile,
            'brightness': self.last_brightness,
            'gamma': self.last_gamma,
            'clip_limit': round(self.clip_limit, 2),
        }
