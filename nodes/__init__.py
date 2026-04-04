NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

_NODES = {
    "SelvaModelLoader":      (".selva_model_loader",      "SelvaModelLoader",      "SelVA Model Loader"),
    "SelvaFeatureExtractor": (".selva_feature_extractor", "SelvaFeatureExtractor", "SelVA Feature Extractor"),
    "SelvaSampler":          (".selva_sampler",           "SelvaSampler",          "SelVA Sampler"),
}

for key, (module_path, class_name, display_name) in _NODES.items():
    try:
        import importlib
        mod = importlib.import_module(module_path, package=__name__)
        NODE_CLASS_MAPPINGS[key] = getattr(mod, class_name)
        NODE_DISPLAY_NAME_MAPPINGS[key] = display_name
    except (ImportError, AttributeError) as e:
        print(f"[SelVA] Skipping {key}: {e}")
