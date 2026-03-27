import torch
import torch.nn.functional as F
from torchaudio import transforms as T


def set_audio_channels(audio, target_channels):
    """Convert audio tensor to target number of channels.

    Args:
        audio: Audio tensor of shape [B, C, T]
        target_channels: Desired number of channels (1 for mono, 2 for stereo)

    Returns:
        Audio tensor with the target number of channels.
    """
    if target_channels == 1:
        # Convert to mono
        audio = audio.mean(1, keepdim=True)
    elif target_channels == 2:
        # Convert to stereo
        if audio.shape[1] == 1:
            audio = audio.repeat(1, 2, 1)
        elif audio.shape[1] > 2:
            audio = audio[:, :2, :]
    return audio


def prepare_audio(audio, in_sr, target_sr, target_length, target_channels, device):
    """Resample, pad/trim, and convert channels of an audio tensor.

    Args:
        audio: Audio tensor (1D, 2D [C, T], or 3D [B, C, T])
        in_sr: Input sample rate
        target_sr: Target sample rate
        target_length: Target length in samples (padded or cropped)
        target_channels: Target number of channels
        device: Torch device to place the audio on

    Returns:
        Audio tensor of shape [B, target_channels, target_length] on device.
    """
    audio = audio.to(device)

    if in_sr != target_sr:
        resample_tf = T.Resample(in_sr, target_sr).to(device)
        audio = resample_tf(audio)

    # Add batch dimension
    if audio.dim() == 1:
        audio = audio.unsqueeze(0).unsqueeze(0)
    elif audio.dim() == 2:
        audio = audio.unsqueeze(0)

    # Pad or crop to target_length
    if audio.shape[-1] < target_length:
        audio = F.pad(audio, (0, target_length - audio.shape[-1]))
    elif audio.shape[-1] > target_length:
        audio = audio[:, :, :target_length]

    audio = set_audio_channels(audio, target_channels)

    return audio
