"""
Model factory functions for PrismAudio inference.

Extracted from:
  - PrismAudio/models/factory.py
  - PrismAudio/models/autoencoders.py  (create_autoencoder_from_config)
  - PrismAudio/models/diffusion.py     (create_diffusion_cond_from_config)
  - PrismAudio/models/conditioners.py  (create_multi_conditioner_from_conditioning_config)

Source: https://github.com/FunAudioLLM/ThinkSound (prismaudio branch)
Only inference-critical factory functions are retained.
"""

import json
import typing as tp
from typing import Dict, Any

import numpy as np


def create_model_from_config(model_config):
    model_type = model_config.get('model_type', None)

    assert model_type is not None, 'model_type must be specified in model config'

    if model_type == 'autoencoder':
        return create_autoencoder_from_config(model_config)
    elif model_type == 'diffusion_cond' or model_type == 'diffusion_cond_inpaint' or model_type == "diffusion_prior" or model_type == "diffusion_infill" or model_type == "mm_diffusion_cond":
        return create_diffusion_cond_from_config(model_config)
    else:
        raise NotImplementedError(f'Unknown model type: {model_type}')


def create_pretransform_from_config(pretransform_config, sample_rate):
    pretransform_type = pretransform_config.get('type', None)

    assert pretransform_type is not None, 'type must be specified in pretransform config'

    if pretransform_type == 'autoencoder':
        from prismaudio_core.models.pretransforms import AutoencoderPretransform

        # Create fake top-level config to pass sample rate to autoencoder constructor
        # This is a bit of a hack but it keeps us from re-defining the sample rate in the config
        autoencoder_config = {"sample_rate": sample_rate, "model": pretransform_config["config"]}
        autoencoder = create_autoencoder_from_config(autoencoder_config)

        scale = pretransform_config.get("scale", 1.0)
        model_half = pretransform_config.get("model_half", False)
        iterate_batch = pretransform_config.get("iterate_batch", False)
        chunked = pretransform_config.get("chunked", False)

        pretransform = AutoencoderPretransform(autoencoder, scale=scale, model_half=model_half, iterate_batch=iterate_batch, chunked=chunked)
    elif pretransform_type == 'wavelet':
        from prismaudio_core.models.pretransforms import WaveletPretransform

        wavelet_config = pretransform_config["config"]
        channels = wavelet_config["channels"]
        levels = wavelet_config["levels"]
        wavelet = wavelet_config["wavelet"]

        pretransform = WaveletPretransform(channels, levels, wavelet)
    elif pretransform_type == 'pqmf':
        from prismaudio_core.models.pretransforms import PQMFPretransform
        pqmf_config = pretransform_config["config"]
        pretransform = PQMFPretransform(**pqmf_config)
    elif pretransform_type == 'dac_pretrained':
        from prismaudio_core.models.pretransforms import PretrainedDACPretransform
        pretrained_dac_config = pretransform_config["config"]
        pretransform = PretrainedDACPretransform(**pretrained_dac_config)
    elif pretransform_type == "audiocraft_pretrained":
        from prismaudio_core.models.pretransforms import AudiocraftCompressionPretransform

        audiocraft_config = pretransform_config["config"]
        pretransform = AudiocraftCompressionPretransform(**audiocraft_config)
    else:
        raise NotImplementedError(f'Unknown pretransform type: {pretransform_type}')

    enable_grad = pretransform_config.get('enable_grad', False)
    pretransform.enable_grad = enable_grad

    pretransform.eval().requires_grad_(pretransform.enable_grad)

    return pretransform


def create_bottleneck_from_config(bottleneck_config):
    bottleneck_type = bottleneck_config.get('type', None)

    assert bottleneck_type is not None, 'type must be specified in bottleneck config'

    if bottleneck_type == 'tanh':
        from prismaudio_core.models.bottleneck import TanhBottleneck
        bottleneck = TanhBottleneck()
    elif bottleneck_type == 'vae':
        from prismaudio_core.models.bottleneck import VAEBottleneck
        bottleneck = VAEBottleneck()
    elif bottleneck_type == 'rvq':
        from prismaudio_core.models.bottleneck import RVQBottleneck

        quantizer_params = {
            "dim": 128,
            "codebook_size": 1024,
            "num_quantizers": 8,
            "decay": 0.99,
            "kmeans_init": True,
            "kmeans_iters": 50,
            "threshold_ema_dead_code": 2,
        }

        quantizer_params.update(bottleneck_config["config"])

        bottleneck = RVQBottleneck(**quantizer_params)
    elif bottleneck_type == "dac_rvq":
        from prismaudio_core.models.bottleneck import DACRVQBottleneck

        bottleneck = DACRVQBottleneck(**bottleneck_config["config"])

    elif bottleneck_type == 'rvq_vae':
        from prismaudio_core.models.bottleneck import RVQVAEBottleneck

        quantizer_params = {
            "dim": 128,
            "codebook_size": 1024,
            "num_quantizers": 8,
            "decay": 0.99,
            "kmeans_init": True,
            "kmeans_iters": 50,
            "threshold_ema_dead_code": 2,
        }

        quantizer_params.update(bottleneck_config["config"])

        bottleneck = RVQVAEBottleneck(**quantizer_params)

    elif bottleneck_type == 'dac_rvq_vae':
        from prismaudio_core.models.bottleneck import DACRVQVAEBottleneck
        bottleneck = DACRVQVAEBottleneck(**bottleneck_config["config"])
    elif bottleneck_type == 'l2_norm':
        from prismaudio_core.models.bottleneck import L2Bottleneck
        bottleneck = L2Bottleneck()
    elif bottleneck_type == "wasserstein":
        from prismaudio_core.models.bottleneck import WassersteinBottleneck
        bottleneck = WassersteinBottleneck(**bottleneck_config.get("config", {}))
    elif bottleneck_type == "fsq":
        from prismaudio_core.models.bottleneck import FSQBottleneck
        bottleneck = FSQBottleneck(**bottleneck_config["config"])
    else:
        raise NotImplementedError(f'Unknown bottleneck type: {bottleneck_type}')

    requires_grad = bottleneck_config.get('requires_grad', True)
    if not requires_grad:
        for param in bottleneck.parameters():
            param.requires_grad = False

    return bottleneck


def create_autoencoder_from_config(config: Dict[str, Any]):
    """Create an AudioAutoencoder from a config dictionary.

    Originally in PrismAudio/models/autoencoders.py.
    """
    from prismaudio_core.models.autoencoders import (
        AudioAutoencoder,
        create_encoder_from_config,
        create_decoder_from_config,
    )

    ae_config = config["model"]

    encoder = create_encoder_from_config(ae_config["encoder"])
    decoder = create_decoder_from_config(ae_config["decoder"])

    bottleneck = ae_config.get("bottleneck", None)

    latent_dim = ae_config.get("latent_dim", None)
    assert latent_dim is not None, "latent_dim must be specified in model config"
    downsampling_ratio = ae_config.get("downsampling_ratio", None)
    assert downsampling_ratio is not None, "downsampling_ratio must be specified in model config"
    io_channels = ae_config.get("io_channels", None)
    assert io_channels is not None, "io_channels must be specified in model config"
    sample_rate = config.get("sample_rate", None)
    assert sample_rate is not None, "sample_rate must be specified in model config"

    in_channels = ae_config.get("in_channels", None)
    out_channels = ae_config.get("out_channels", None)

    pretransform = ae_config.get("pretransform", None)

    if pretransform is not None:
        pretransform = create_pretransform_from_config(pretransform, sample_rate)

    if bottleneck is not None:
        bottleneck = create_bottleneck_from_config(bottleneck)

    soft_clip = ae_config["decoder"].get("soft_clip", False)

    return AudioAutoencoder(
        encoder,
        decoder,
        io_channels=io_channels,
        latent_dim=latent_dim,
        downsampling_ratio=downsampling_ratio,
        sample_rate=sample_rate,
        bottleneck=bottleneck,
        pretransform=pretransform,
        in_channels=in_channels,
        out_channels=out_channels,
        soft_clip=soft_clip
    )


def create_multi_conditioner_from_conditioning_config(config: tp.Dict[str, tp.Any]):
    """Create a MultiConditioner from a conditioning config dictionary.

    Originally in PrismAudio/models/conditioners.py.
    """
    from prismaudio_core.models.conditioners import (
        MultiConditioner,
        T5Conditioner,
        CLAPTextConditioner,
        CLIPTextConditioner,
        MetaCLIPTextConditioner,
        CLAPAudioConditioner,
        Cond_MLP,
        Global_MLP,
        Sync_MLP,
        Cond_MLP_1,
        Cond_ConvMLP,
        Cond_MLP_Global,
        Cond_MLP_Global_1,
        Cond_MLP_Global_2,
        Video_Global,
        Video_Sync,
        Text_Linear,
        CLIPConditioner,
        IntConditioner,
        NumberConditioner,
        PhonemeConditioner,
        TokenizerLUTConditioner,
        PretransformConditioner,
        mm_unchang,
    )
    from prismaudio_core.models.utils import load_ckpt_state_dict

    conditioners = {}
    cond_dim = config["cond_dim"]

    default_keys = config.get("default_keys", {})

    for conditioner_info in config["configs"]:
        id = conditioner_info["id"]

        conditioner_type = conditioner_info["type"]

        conditioner_config = {"output_dim": cond_dim}

        conditioner_config.update(conditioner_info["config"])
        if conditioner_type == "t5":
            conditioners[id] = T5Conditioner(**conditioner_config)
        elif conditioner_type == "clap_text":
            conditioners[id] = CLAPTextConditioner(**conditioner_config)
        elif conditioner_type == "clip_text":
            conditioners[id] = CLIPTextConditioner(**conditioner_config)
        elif conditioner_type == "metaclip_text":
            conditioners[id] = MetaCLIPTextConditioner(**conditioner_config)
        elif conditioner_type == "clap_audio":
            conditioners[id] = CLAPAudioConditioner(**conditioner_config)
        elif conditioner_type == "cond_mlp":
            conditioners[id] = Cond_MLP(**conditioner_config)
        elif conditioner_type == "global_mlp":
            conditioners[id] = Global_MLP(**conditioner_config)
        elif conditioner_type == "sync_mlp":
            conditioners[id] = Sync_MLP(**conditioner_config)
        elif conditioner_type == "cond_mlp_1":
            conditioners[id] = Cond_MLP_1(**conditioner_config)
        elif conditioner_type == "cond_convmlp":
            conditioners[id] = Cond_ConvMLP(**conditioner_config)
        elif conditioner_type == "cond_mlp_global":
            conditioners[id] = Cond_MLP_Global(**conditioner_config)
        elif conditioner_type == "cond_mlp_global_1":
            conditioners[id] = Cond_MLP_Global_1(**conditioner_config)
        elif conditioner_type == "cond_mlp_global_2":
            conditioners[id] = Cond_MLP_Global_2(**conditioner_config)
        elif conditioner_type == "video_global":
            conditioners[id] = Video_Global(**conditioner_config)
        elif conditioner_type == "video_sync":
            conditioners[id] = Video_Sync(**conditioner_config)
        elif conditioner_type == "text_linear":
            conditioners[id] = Text_Linear(**conditioner_config)
        elif conditioner_type == "video_clip":
            conditioners[id] = CLIPConditioner(**conditioner_config)
        elif conditioner_type == "int":
            conditioners[id] = IntConditioner(**conditioner_config)
        elif conditioner_type == "number":
            conditioners[id] = NumberConditioner(**conditioner_config)
        elif conditioner_type == "phoneme":
            conditioners[id] = PhonemeConditioner(**conditioner_config)
        elif conditioner_type == "lut":
            conditioners[id] = TokenizerLUTConditioner(**conditioner_config)
        elif conditioner_type == "pretransform":
            sample_rate = conditioner_config.pop("sample_rate", None)
            assert sample_rate is not None, "Sample rate must be specified for pretransform conditioners"

            pretransform = create_pretransform_from_config(conditioner_config.pop("pretransform_config"), sample_rate=sample_rate)

            if conditioner_config.get("pretransform_ckpt_path", None) is not None:
                pretransform.load_state_dict(load_ckpt_state_dict(conditioner_config.pop("pretransform_ckpt_path")))

            conditioners[id] = PretransformConditioner(pretransform, **conditioner_config)
        elif conditioner_type == "mm_unchang":
            conditioners[id] = mm_unchang(**conditioner_config)
        else:
            raise ValueError(f"Unknown conditioner type: {conditioner_type}")

    return MultiConditioner(conditioners, default_keys=default_keys)


def create_diffusion_cond_from_config(config: tp.Dict[str, tp.Any]):
    """Create a ConditionedDiffusionModelWrapper from a config dictionary.

    Originally in PrismAudio/models/diffusion.py.
    """
    from prismaudio_core.models.diffusion import (
        ConditionedDiffusionModelWrapper,
        MMConditionedDiffusionModelWrapper,
        UNetCFG1DWrapper,
        UNet1DCondWrapper,
        DiTWrapper,
        MMDiTWrapper,
    )

    model_config = config["model"]

    model_type = config["model_type"]

    diffusion_config = model_config.get('diffusion', None)
    assert diffusion_config is not None, "Must specify diffusion config"

    diffusion_model_type = diffusion_config.get('type', None)
    assert diffusion_model_type is not None, "Must specify diffusion model type"

    diffusion_model_config = diffusion_config.get('config', None)
    assert diffusion_model_config is not None, "Must specify diffusion model config"

    if diffusion_model_type == 'adp_cfg_1d':
        diffusion_model = UNetCFG1DWrapper(**diffusion_model_config)
    elif diffusion_model_type == 'adp_1d':
        diffusion_model = UNet1DCondWrapper(**diffusion_model_config)
    elif diffusion_model_type == 'dit':
        diffusion_model = DiTWrapper(**diffusion_model_config)
    elif diffusion_model_type == 'mmdit':
        diffusion_model = MMDiTWrapper(**diffusion_model_config)

    io_channels = model_config.get('io_channels', None)
    assert io_channels is not None, "Must specify io_channels in model config"

    sample_rate = config.get('sample_rate', None)
    assert sample_rate is not None, "Must specify sample_rate in config"

    diffusion_objective = diffusion_config.get('diffusion_objective', 'v')

    conditioning_config = model_config.get('conditioning', None)

    conditioner = None
    if conditioning_config is not None:
        conditioner = create_multi_conditioner_from_conditioning_config(conditioning_config)

    cross_attention_ids = diffusion_config.get('cross_attention_cond_ids', [])
    add_cond_ids = diffusion_config.get('add_cond_ids', [])
    sync_cond_ids = diffusion_config.get('sync_cond_ids', [])
    global_cond_ids = diffusion_config.get('global_cond_ids', [])
    input_concat_ids = diffusion_config.get('input_concat_ids', [])
    prepend_cond_ids = diffusion_config.get('prepend_cond_ids', [])
    mm_cond_ids = diffusion_config.get('mm_cond_ids', [])
    zero_init = diffusion_config.get('zero_init', False)
    pretransform = model_config.get("pretransform", None)

    if pretransform is not None:
        pretransform = create_pretransform_from_config(pretransform, sample_rate)
        min_input_length = pretransform.downsampling_ratio
    else:
        min_input_length = 1

    if diffusion_model_type == "adp_cfg_1d" or diffusion_model_type == "adp_1d":
        min_input_length *= np.prod(diffusion_model_config["factors"])
    elif diffusion_model_type == "dit":
        min_input_length *= diffusion_model.model.patch_size

    # Get the proper wrapper class

    extra_kwargs = {}

    if model_type == "mm_diffusion_cond":
        wrapper_fn = MMConditionedDiffusionModelWrapper
        extra_kwargs["diffusion_objective"] = diffusion_objective
        extra_kwargs["mm_cond_ids"] = mm_cond_ids

    if model_type == "diffusion_cond" or model_type == "diffusion_cond_inpaint" or model_type == 'diffusion_infill':
        wrapper_fn = ConditionedDiffusionModelWrapper
        extra_kwargs["diffusion_objective"] = diffusion_objective

    elif model_type == "diffusion_prior":
        prior_type = model_config.get("prior_type", None)
        assert prior_type is not None, "Must specify prior_type in diffusion prior model config"

        if prior_type == "mono_stereo":
            from prismaudio_core.models.diffusion_prior import MonoToStereoDiffusionPrior
            wrapper_fn = MonoToStereoDiffusionPrior

    return wrapper_fn(
        diffusion_model,
        conditioner,
        min_input_length=min_input_length,
        sample_rate=sample_rate,
        cross_attn_cond_ids=cross_attention_ids,
        global_cond_ids=global_cond_ids,
        input_concat_ids=input_concat_ids,
        prepend_cond_ids=prepend_cond_ids,
        add_cond_ids=add_cond_ids,
        sync_cond_ids=sync_cond_ids,
        pretransform=pretransform,
        io_channels=io_channels,
        zero_init=zero_init,
        **extra_kwargs
    )
