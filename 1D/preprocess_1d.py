import random
import numpy as np
import librosa


class PitchShifting:
    def __init__(self, n_steps_range=(-2, 2), sr=22050):
        self.n_steps_range = n_steps_range
        self.sr = sr

    def __call__(self, y: np.ndarray) -> np.ndarray:
        n_steps = random.randint(self.n_steps_range[0], self.n_steps_range[1])
        return librosa.effects.pitch_shift(y, sr=self.sr, n_steps=n_steps)


class AddNoise:
    def __init__(self, noise_factor=0.005):
        self.noise_factor = noise_factor

    def __call__(self, y: np.ndarray) -> np.ndarray:
        noise = np.random.randn(len(y))
        return y + self.noise_factor * noise


class Gain:
    def __init__(self, gain_range=(0.8, 1.2)):
        self.gain_range = gain_range

    def __call__(self, y: np.ndarray) -> np.ndarray:
        g = random.uniform(*self.gain_range)
        return y * g


class ComposeAudio:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, y: np.ndarray) -> np.ndarray:
        for t in self.transforms:
            y = t(y)
        return y


def build_augmentations(enable: bool):
    if not enable:
        return None
    return ComposeAudio([
        PitchShifting(n_steps_range=(-2, 2), sr=22050),
        AddNoise(noise_factor=0.005),
        Gain(gain_range=(0.8, 1.2)),
    ])
