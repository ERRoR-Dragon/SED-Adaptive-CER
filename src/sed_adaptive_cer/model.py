"""Build the original SED model and run individual or shared-prefix inference."""

import sys
from time import perf_counter
from unittest.mock import patch

import numpy as np
import torch

from .policy import BUDGETS, BUDGET_NAMES, scene_features


def build_sed(args):
    sys.path.insert(0, str(args.sed_root))
    import open_clip
    from detectron2.config import get_cfg
    from detectron2.data import transforms as T
    from detectron2.modeling import build_model
    from detectron2.projects.deeplab import add_deeplab_config
    from sed import add_sed_config

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_sed_config(cfg)
    cfg.merge_from_file(str(args.sed_root / "configs" / "convnextB_768.yaml"))
    cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON = str(args.sed_root / "datasets" / "coco.json")
    cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON = str(args.sed_root / "datasets" / "ade150.json")
    cfg.MODEL.SEM_SEG_HEAD.POOLING_SIZES = [1, 1]
    cfg.MODEL.DEVICE = "cuda"
    cfg.TEST.SLIDING_WINDOW = False
    cfg.freeze()

    # Build the original text features before loading the segmentation checkpoint.
    factory = open_clip.create_model_and_transforms

    def local_factory(name, *, pretrained=None, **kwargs):
        if name != "convnext_base_w_320":
            raise ValueError(f"Unexpected OpenCLIP model {name!r} for ConvNeXt-B.")
        return factory(name, pretrained=str(args.openclip_weights), **kwargs)

    with patch.object(open_clip, "create_model_and_transforms", local_factory):
        model = build_model(cfg)
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.requires_grad_(False).eval()
    resize = T.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST] * 2, cfg.INPUT.MAX_SIZE_TEST)
    return model, resize, cfg.dump()


class SedInference:
    def __init__(self, model, amp):
        from sed import sed_model

        self.model = model
        self.predictor = model.sem_seg_head.predictor
        self.aggregator = self.predictor.transformer
        if not hasattr(self.aggregator, "decode_prepared"):
            raise RuntimeError("Apply patches/adaptive_cer.patch to the pinned SED source.")
        self.vocabularies = {"coco": self.predictor.text_features, "ade150": self.predictor.text_features_test}
        self.num_classes = 150
        self.amp = amp
        self.sed_module = sed_model
        self.postprocess = sed_model.sem_seg_postprocess

    def set_vocabulary(self, name):
        self.predictor.text_features_test = self.vocabularies[name]
        self.num_classes = self.vocabularies[name].shape[0]

    def label_map(self, scores, image_size, height, width):
        # Chunk very large outputs without changing the interpolation or argmax tie rule.
        output_bytes = scores.shape[0] * height * width * scores.element_size()
        if output_bytes <= 256 * 1024**2:
            return self.postprocess(scores, image_size, height, width).argmax(dim=0)

        best_score = best_index = None
        for offset in range(0, scores.shape[0], 16):
            tile = self.postprocess(scores[offset:offset + 16], image_size, height, width)
            score, index = tile.max(dim=0)
            if best_score is None:
                best_score, best_index = score, index
            else:
                best_index = torch.where(score > best_score, index + offset, best_index)
                best_score = torch.maximum(best_score, score)
        return best_index

    def _mode(self, budget, policy=None):
        fast = budget is not None or policy is not None
        self.model.fast_inference = self.aggregator.fast_infer = fast
        self.aggregator.cer_policy = policy
        if budget is not None:
            self.aggregator.topK = budget

    def _kept(self):
        kept = self.aggregator.cer_kept
        return np.ones(self.num_classes, dtype=bool) if kept is None else kept.cpu().numpy()

    @torch.inference_mode()
    def predict(self, item, budget=None, policy=None, timed=False):
        self._mode(budget, policy)
        with patch.object(self.sed_module, "sem_seg_postprocess", self.label_map):
            if timed:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = perf_counter()
            try:
                with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
                    labels = self.model([item])[0]["sem_seg"]
                timing = None
                if timed:
                    torch.cuda.synchronize()
                    timing = {"ms": 1000 * (perf_counter() - start),
                              "peak_mib": torch.cuda.max_memory_allocated() / 1024**2}
            finally:
                self.aggregator.cer_policy = None
        return labels.cpu().numpy(), self._kept(), timing

    @torch.inference_mode()
    def sweep(self, item):
        """Share the prefix across five fixed budgets. Return CPU masks and coarse features."""
        prepare = self.aggregator.prepare_decoder
        prepared = features = None

        def capture_prefix(*args):
            nonlocal prepared
            prepared = prepare(*args)
            return prepared

        def capture_features(logits):
            nonlocal features
            features = scene_features(logits)
            return BUDGETS[0]

        with patch.object(self.aggregator, "prepare_decoder", capture_prefix):
            first, kept, _ = self.predict(item, budget=BUDGETS[0], policy=capture_features)
        masks = {BUDGET_NAMES[0]: (first, kept)}
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
            for name, budget in zip(BUDGET_NAMES[1:], BUDGETS[1:]):
                self._mode(budget)
                scores = self.aggregator.decode_prepared(*prepared)[0][0]
                if budget is None:
                    scores = scores.sigmoid()
                labels = self.label_map(scores, item["image"].shape[-2:], item["height"], item["width"])
                masks[name] = (labels.cpu().numpy(), self._kept())
        return features, masks
