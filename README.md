# DinoFOrge : 

A unified pipeline for robust object discovery and segmentation that combines:
- YOLO for fast, high-precision known-class detection.
- An open-vocabulary ensemble (Grounding DINO, Florence-2, GLIP, Detectron2-OVD, LP-OVOD) for novel-category discovery from text prompts.
- SAM 2 for mask-level refinement and box tightening.[10][11][12][13][14][15]

## Highlights
- Known classes first: YOLO provides trusted detections for the in-domain label set with strong latency/throughput.[13]
- Open-world recall: Grounding DINO, GLIP, Florence-2, Detectron2 open-vocab head, and LP-OVOD expand coverage to unseen classes via prompts and language-aligned features.[11][12][10][13]
- Pixel-accurate masks: SAM 2 converts boxes to high-quality instance masks without task-specific finetuning.[14][15]

## Architecture

1) YOLO pass (Known set)
- Input: Raw RGB image.
- Output: Boxes, labels, and confidences for the predefined/known categories.
- Action: Keep detections above conf_known and apply class-aware NMS; these go directly to the final pool as.[16][13]

2) Open-vocabulary ensemble pass (Novel discovery)
- Models on the same raw image:
  - Grounding DINO: Text-prompted detection with phrase thresholds.[10]
  - Florence-2: Vision-language model; run object detection/grounding task for phrase boxes.[11]
  - GLIP: Open-vocabulary detector aligning region and text; strong zero-/few-shot behavior.[12]
  - Detectron2 OVD: Open-vocabulary inference via language-aligned heads within Detectron2.[13]
  - LP-OVOD: Linear probing on pre-trained vision-language features to obtain open-vocabulary boxes and scores.[12]
- Pooling: Apply cross-model box fusion per class/phrase using WBF or soft-NMS, selecting high-score boxes as.[13]

3) Merge and refine
- Concatenate  +, run a global de-duplication (NMS/WBF).[16][13]
- Send all merged boxes to SAM 2 to obtain instance masks and refined/tightened boxes.[15][14]

4) Final outputs
- Per instance: {box, mask, label, score, provenance}, where provenance records the winning model or the fusion mix.[14][15]

## Rationale

- Separation of concerns: Known-set precision from YOLO; open-set recall from a diverse ensemble; mask precision from SAM 2.[10][13][14]
- Complementary strengths: Grounding DINO handles prompt-phrase localization; GLIP leverages language-region alignment; Florence-2 supplies grounding via a modern VLM; Detectron2 OVD offers a familiar framework; LP-OVOD adds a strong, efficient linear-probe baseline.[11][12][13][10]
- Reliable refinement: SAM 2 improves boundaries and tightens boxes consistently across categories.[15][14]

## Setup

Prerequisites
- Python 3.10+, CUDA-ready PyTorch recommended.
- GPUs with sufficient VRAM for concurrent model inference.

Install references
- Grounding DINO and Grounded-SAM ecosystem for text-prompted detection scaffolding.[10]
- Florence-2 checkpoints/APIs for detection/grounding tasks.[11]
- GLIP open-vocabulary detection environment.[12]
- Detectron2 with an OVD-capable configuration.[13]
- LP-OVOD repository and linear-probe setup.[12]
- SAM 2 for image inference and masks.[14]

Note: Follow each linked project’s official installation instructions and licenses when pulling code/weights.[14][10][11]

## Configuration

Prompts
- A YAML/JSON file lists classes, synonyms, and prompt templates used by Grounding DINO, GLIP, and Florence-2 (“a photo of a [class]”, “there is a [class]”).[10][11][12]

Thresholds
- conf_known (YOLO).
- τ_gdino, τ_florence, τ_glip, τ_d2, τ_lp (per open-vocab model).
- Ensemble fusion IoU for WBF/soft-NMS; global NMS IoU after concatenation.
- SAM 2 mask quality cutoff to drop weak masks.[13][14]

Backbones and variants
- Grounding DINO variant and text score threshold.[10]
- Florence-2 model size (e.g., base/large) and task spec.[11]
- GLIP checkpoint (Swin/ViT variants).[12]
- Detectron2 OVD head configuration.[13]
- LP-OVOD linear probe weights and feature extractor choice.[12]
- SAM 2 model size (image mode).[15][14]

## Inference

1) YOLO known-classes
- Run inference; filter by conf_known; class-aware NMS; save as set.[16][13]

2) Open-vocab ensemble
- Grounding DINO: prompts → boxes; keep score ≥ τ_gdino.[10]
- Florence-2: detection/grounding; keep score ≥ τ_florence.[11]
- GLIP: prompted detection; keep score ≥ τ_glip.[12]
- Detectron2 OVD: open-vocab logits; keep score ≥ τ_d2.[13]
- LP-OVOD: linear-probe scores; keep score ≥ τ_lp.[12]
- Fuse per phrase/class with WBF/soft-NMS to form set.[13]

3) Concatenate and deduplicate
- Combine  +; run global NMS/WBF; record provenance.[16][13]

4) SAM 2 refinement
- For each box, run SAM 2 to get a mask and a tightened box; compute final score by fusing detector score and mask confidence.[15][14]

Outputs
- JSONL or COCO-style: image_id, bbox, segmentation (RLE/polygons), category, score, provenance.[14]

## Training and Tuning Notes

- Known-set detector: Train or fine-tune YOLO on your canonical labels for best precision.[13]
- Prompt engineering: Curate synonyms/templates to improve recall in Grounding DINO, GLIP, and Florence-2.[11][10][12]
- LP-OVOD setup: Choose a robust vision-language backbone for features; validate probe thresholds against a held-out set.[12]
- Ensemble calibration: Start τ values in 0.25–0.35 range and adjust; tune WBF IoU around 0.5–0.6; verify class merging logic for overlapping phrases.[13]
- SAM 2 mask gating: Drop low-quality masks to reduce false positives and refine final AP.[15][14]

## Evaluation

- Closed-set: mAP/mAR on known categories from YOLO outputs.[13]
- Open-vocab: Phrase-level detection metrics using curated prompt lists; report per-model and fused performance.[13]
- Segmentation: Evaluate mask AP after SAM 2 refinement; confirm box tightening improves IoU over raw proposals.[14][15]

## Use Cases

- Auto-annotation with prompts for rapid dataset bootstrapping; export masks and labels to labeling tools.[14][10]
- Robotics/AV scenarios requiring novel object discovery with reliable masks for downstream planners.[15][14]
- Continuous taxonomy expansion where classes evolve without retraining everything.[10][11]

## References and Pointers

- Grounding DINO and Grounded-SAM ecosystem (text-prompted detection and pipelines).[10]
- Florence-2 tasks and checkpoints for object detection/grounding.[11]
- GLIP open-vocabulary detector and usage.[12]
- Detectron2 with open-vocabulary configurations.[13]
- LP-OVOD: Linear probing for open-vocabulary object detection.[12]
- SAM 2: Image/video segmentation; installation and inference guides.[15][14]

Legal and licensing
- Respect each upstream project’s license; distribute weights and code per their terms. Avoid reproducing proprietary content verbatim.[14][11][10]

[1](https://gist.github.com/raytroop/abbfb31772a5c8797dade81193da16d5)
[2](https://huggingface.co/google/owlv2-base-patch16-ensemble/blob/27d9485a27e88c5f0435d0ed8382dcb3b911ab46/README.md)
[3](https://pmc.ncbi.nlm.nih.gov/articles/PMC8516532/)
[4](https://stackoverflow.com/questions/45385661/object-detection-api-how-to-create-an-ensemble-of-trainings)
[5](https://huggingface.co/google/owlv2-base-patch16-ensemble/blame/27d9485a27e88c5f0435d0ed8382dcb3b911ab46/README.md)
[6](https://www.ultralytics.com/glossary/ensemble)
[7](https://gitlab.physik.uni-muenchen.de/Christoph.Fischer/enstools-feature/-/blob/master/README.md)
[8](https://www.sciencedirect.com/science/article/pii/S0168169925013195)
[9](https://handong1587.github.io/deep_learning/2015/10/09/object-detection.html)
[10](https://github.com/IDEA-Research/Grounded-Segment-Anything)
[11](https://huggingface.co/microsoft/Florence-2-base)
[12](https://github.com/VinAIResearch/LP-OVOD)
[13](https://github.com/topics/open-vocabulary-detection)
[14](https://github.com/facebookresearch/sam2)
[15](https://ai.meta.com/sam2/)
[16](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/images/60973254/7520d583-6e7b-4ab4-92f6-e0cada17db83/image.jpg)