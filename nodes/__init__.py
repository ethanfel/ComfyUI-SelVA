NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

_NODES = {
    "SelvaModelLoader":      (".selva_model_loader",      "SelvaModelLoader",      "SelVA Model Loader"),
    "SelvaFeatureExtractor": (".selva_feature_extractor", "SelvaFeatureExtractor", "SelVA Feature Extractor"),
    "SelvaSampler":          (".selva_sampler",           "SelvaSampler",          "SelVA Sampler"),
    "SelvaLoraLoader":       (".selva_lora_loader",       "SelvaLoraLoader",       "SelVA LoRA Loader"),
    "SelvaLoraTrainer":      (".selva_lora_trainer",      "SelvaLoraTrainer",      "SelVA LoRA Trainer"),
    "SelvaLoraScheduler":    (".selva_lora_scheduler",    "SelvaLoraScheduler",    "SelVA LoRA Scheduler"),
    "SelvaDatasetBrowser":   (".selva_dataset_browser",   "SelvaDatasetBrowser",   "SelVA Dataset Browser"),
    "SelvaSkipExperiment":   (".selva_skip_experiment",   "SelvaSkipExperiment",   "SelVA Skip Experiment"),
    "SelvaLoraEvaluator":    (".selva_lora_evaluator",    "SelvaLoraEvaluator",    "SelVA LoRA Evaluator"),
    "SelvaVaeRoundtrip":     (".selva_vae_roundtrip",     "SelvaVaeRoundtrip",     "SelVA VAE Roundtrip"),
    "SelvaHfSmoother":       (".selva_audio_preprocessors", "SelvaHfSmoother",       "SelVA HF Smoother"),
    "SelvaSpectralMatcher":  (".selva_audio_preprocessors", "SelvaSpectralMatcher",  "SelVA Spectral Matcher"),
    "SelvaTextualInversionTrainer": (".selva_textual_inversion_trainer", "SelvaTextualInversionTrainer", "SelVA Textual Inversion Trainer"),
    "SelvaTextualInversionLoader":  (".selva_textual_inversion_loader",  "SelvaTextualInversionLoader",  "SelVA Textual Inversion Loader"),
    "SelvaTiScheduler":                      (".selva_ti_scheduler",                      "SelvaTiScheduler",                      "SelVA TI Scheduler"),
    "SelvaActivationSteeringExtractor":      (".selva_activation_steering_extractor",      "SelvaActivationSteeringExtractor",      "SelVA Activation Steering Extractor"),
    "SelvaActivationSteeringLoader":         (".selva_activation_steering_loader",         "SelvaActivationSteeringLoader",         "SelVA Activation Steering Loader"),
    "SelvaBigvganTrainer":                   (".selva_bigvgan_trainer",                    "SelvaBigvganTrainer",                   "SelVA BigVGAN Trainer"),
    "SelvaBigvganLoader":                    (".selva_bigvgan_loader",                     "SelvaBigvganLoader",                    "SelVA BigVGAN Loader"),
    "SelvaBigvganScheduler":                 (".selva_bigvgan_scheduler",                  "SelvaBigvganScheduler",                 "SelVA BigVGAN Scheduler"),
    "SelvaDittoOptimizer":                   (".selva_ditto_optimizer",                    "SelvaDittoOptimizer",                   "SelVA DITTO Optimizer"),
    "SelvaDatasetLoader":          (".selva_dataset_pipeline", "SelvaDatasetLoader",          "SelVA Dataset Loader"),
    "SelvaDatasetResampler":       (".selva_dataset_pipeline", "SelvaDatasetResampler",       "SelVA Dataset Resampler"),
    "SelvaDatasetLUFSNormalizer":  (".selva_dataset_pipeline", "SelvaDatasetLUFSNormalizer",  "SelVA Dataset LUFS Normalizer"),
    "SelvaDatasetCompressor":      (".selva_dataset_pipeline", "SelvaDatasetCompressor",      "SelVA Dataset Compressor"),
    "SelvaDatasetInspector":       (".selva_dataset_pipeline", "SelvaDatasetInspector",       "SelVA Dataset Inspector"),
    "SelvaDatasetItemExtractor":   (".selva_dataset_pipeline", "SelvaDatasetItemExtractor",   "SelVA Dataset Item Extractor"),
    "SelvaDatasetSaver":           (".selva_dataset_pipeline", "SelvaDatasetSaver",           "SelVA Dataset Saver"),
    "SelvaHarmonicExciter":        (".selva_audio_postprocess", "SelvaHarmonicExciter",        "SelVA Harmonic Exciter"),
    "SelvaOutputNormalizer":       (".selva_audio_postprocess", "SelvaOutputNormalizer",       "SelVA Output Normalizer"),
    "SelvaDatasetSpectralMatcher": (".selva_dataset_pipeline",  "SelvaDatasetSpectralMatcher", "SelVA Dataset Spectral Matcher"),
    "SelvaDatasetHfSmoother":      (".selva_dataset_pipeline",  "SelvaDatasetHfSmoother",      "SelVA Dataset HF Smoother"),
    "SelvaDatasetAugmenter":       (".selva_dataset_pipeline",  "SelvaDatasetAugmenter",       "SelVA Dataset Augmenter"),
}

for key, (module_path, class_name, display_name) in _NODES.items():
    try:
        import importlib
        mod = importlib.import_module(module_path, package=__name__)
        NODE_CLASS_MAPPINGS[key] = getattr(mod, class_name)
        NODE_DISPLAY_NAME_MAPPINGS[key] = display_name
    except (ImportError, AttributeError) as e:
        print(f"[SelVA] Skipping {key}: {e}")
