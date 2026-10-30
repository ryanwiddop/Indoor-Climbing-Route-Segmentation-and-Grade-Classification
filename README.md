# Indoor Climbing Route Segmentation and Grade Classification

## Objective

This project is an honors thesis project at The University of Massachusetts Lowell. The goal of this project is to create a dataset consisting of images of indoor rock climbing (bouldering) walls, and train a machine learning model to segment and classify routes by V grade.

## Pipeline
1. Segment mask for every hold on the wall and classify type
2. Pairwise score each hold against every other hold to classify routes
3. Classify grade given a route (group of holds)

## Dataset

This dataset was created specifically for this project. Every image is a photograph of a bouldering wall taken at an indoor climbing gym. Images were annotated using [Computer Vision Annotation Tool (CVAT)](https://app.cvat.ai/) by drawing polygons around individual holds visible in the foreground of each image. Every polygons class is the hold type, and its attributes are route id and route grade.

### Dataset Statistics

> **Dataset will be published to Kaggle:** `[Kaggle dataset link - TBD upon upload]`

| Stat | Value |
|---|---|
| Total wall images | 35 |
| Total annotated holds | 3858 |
| Total routes | 365 |
| Grade range | V0-V11 |

### Tiling

Full-resolution wall photographs are too large to feed directly into a detection model. To work around this, `prepare_tiles.py` cuts each image into 1600x1600 pixel crops using a sliding window with 75% overlap between adjacent tiles (configurable). A hold annotation is included in a tile only if at least 66% of the hold's bounding box overlaps the tile window, which avoids training on holds that are mostly cropped out. Polygon coordinates are translated into tile-local space and saved alongside each tile in a COCO JSON.

Negative tiles (those containing no qualifying holds) are preserved and passed through the dataset so the model sees background-only examples during training.

## Model

### Model 1: Hold Detection & Type Classification

Phase 1 is a Mask R-CNN instance segmentation model that takes an 800px wall tile as input and outputs per-hold bounding boxes, segmentation masks, and a hold type classification. The backbone is a ResNet-50 FPN (Feature Pyramid Network), initialized with pretrained V2 weights from torchvision. Both the box predictor and mask predictor heads are replaced with new heads sized to the number of hold type classes in the dataset.

Training uses AdamW (lr=2e-4, weight_decay=1e-5) with a MultiStepLR scheduler that drops the learning rate by 10x at epochs 30 and 40. The model is trained for 45 epochs with a batch size of 16 and FP16 mixed precision via PyTorch's `GradScaler`. Data is split 80/20 for train/validation with a fixed seed for reproducibility. The training set receives random augmentation (flips, color jitter, etc.) while the validation set does not.

Validation runs at the end of training against the held-out 20%. Predictions are matched greedily to ground truth boxes by box IoU, and threshold sweeping over selects the score threshold with the best F1. The final checkpoint and training loss curve are saved to `models/checkpoints/` and `models/figures/`.

#### Evaluation (Whole-Wall Inference)

Because the model was trained on 800px tiles, evaluating it on a full-resolution wall image requires a sliding-window approach. `phase_1_eval.py` tiles each wall image using the same stride/overlap logic as preprocessing, runs the model on each tile independently, translates the tile-local detections back into wall coordinates, then applies a global greedy NMS pass to merge duplicates from overlapping tiles. The model's built-in detection cap is raised to 500 per tile to avoid hard-capping recall on dense walls.

Metrics computed at evaluation include precision, recall, and F1 at a configurable operating point, mean matched box IoU, mean matched mask IoU, and COCO-style mAP. All metrics are written to a JSON file alongside per-image visualization overlays, P-R curves, score histograms, and box/mask IoU histograms.
