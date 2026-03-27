NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

_NODES = {
    "PrismAudioModelLoader": (".model_loader", "PrismAudioModelLoader", "PrismAudio Model Loader"),
    "PrismAudioFeatureLoader": (".feature_loader", "PrismAudioFeatureLoader", "PrismAudio Feature Loader"),
    "PrismAudioFeatureExtractor": (".feature_extractor", "PrismAudioFeatureExtractor", "PrismAudio Feature Extractor"),
    "PrismAudioSampler": (".sampler", "PrismAudioSampler", "PrismAudio Sampler"),
    "PrismAudioTextOnly": (".text_only", "PrismAudioTextOnly", "PrismAudio Text Only"),
}

for key, (module_path, class_name, display_name) in _NODES.items():
    try:
        import importlib
        mod = importlib.import_module(module_path, package=__name__)
        NODE_CLASS_MAPPINGS[key] = getattr(mod, class_name)
        NODE_DISPLAY_NAME_MAPPINGS[key] = display_name
    except (ImportError, AttributeError) as e:
        print(f"[PrismAudio] Skipping {key}: {e}")
