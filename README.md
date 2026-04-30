# Indoor Climbing Route Segmentation and Classification

## Objective

This project is an honors thesis project at The University of Massachusetts Lowell. The goal of this project is to create a dataset consisting of images of indoor rock climbing (bouldering) walls, then train a machine learning model to segment and classify routes by V grade.

The pipeline is split into four phases. Phase 1 trains a Mask R-CNN model to detect and classify individual holds on a wall image. Phase 2 assigns each detected hold a color using CIELab-based binning. Phase 3 groups same-color holds into candidate routes via DBSCAN spatial clustering and DFS tracing. Phase 4 classifies each route by V-grade.


## Dataset

This dataset was created specifically for this project. Every image is a photograph of a bouldering wall taken at an indoor climbing gym. Images were annotated using the [VGG Image Annotator (VIA)](https://www.robots.ox.ac.uk/~vgg/software/via/) and later [Computer Vision Annotation Tool (CVAT)](https://app.cvat.ai/) by drawing polygons around individual holds visible in the foreground of each image. Every polygon has a label with the hold type, as well as route id and route grade.

### Dataset Statistics

> **Dataset will be published to Kaggle:** `[Kaggle dataset link - TBD upon upload]`

| Stat | Value |
|---|---|
| Total wall images | `[N]` |
| Total annotated holds | `[N]` |
| Total routes | `[N]` |
| Grade range | V0 – V`[N]` |
| Total tiles (800px, 25% overlap) | `[N]` |
| Positive tiles (contain ≥1 hold) | `[N]` |
| Negative tiles (background only) | `[N]` |

### Tiling

Full-resolution wall photographs are too large to feed directly into a detection model. To work around this, `prepare_tiles.py` cuts each image into 800×800 pixel crops using a sliding window with 25% overlap between adjacent tiles (configurable). A hold annotation is included in a tile only if at least 35% of the hold's bounding box overlaps the tile window, which avoids training on holds that are mostly cropped out. Polygon coordinates are translated into tile-local space and saved alongside each tile in a CSV. The resulting `tiles/` directory and `tiles.csv` are what the training script actually reads.

Negative tiles (those containing no qualifying holds) are preserved in the CSV and passed through the dataset so the model sees background-only examples during training.

## Model

### Model 1: Hold Detection & Type Classification

Phase 1 is a Mask R-CNN instance segmentation model that takes an 800px wall tile as input and outputs per-hold bounding boxes, segmentation masks, and a hold type classification. The backbone is a ResNet-50 FPN (Feature Pyramid Network), initialized with pretrained V2 weights from torchvision. Both the box predictor and mask predictor heads are replaced with new heads sized to the number of hold type classes in the dataset.

Training uses AdamW (lr=2e-4, weight_decay=1e-5) with a MultiStepLR scheduler that drops the learning rate by 10x at epochs 30 and 40. The model is trained for 45 epochs with a batch size of 16 and FP16 mixed precision via PyTorch's `GradScaler`. Data is split 80/20 for train/validation with a fixed seed for reproducibility. The training set receives random augmentation (flips, color jitter, etc.) while the validation set does not.

Validation runs at the end of training against the held-out 20%. Predictions are matched greedily to ground truth boxes by box IoU, and threshold sweeping over selects the score threshold with the best F1. The final checkpoint and training loss curve are saved to `model/checkpoints/` and `model/figures/`.

#### Evaluation (Whole-Wall Inference)

Because the model was trained on 800px tiles, evaluating it on a full-resolution wall image requires a sliding-window approach. `phase_1_eval.py` tiles each wall image using the same stride/overlap logic as preprocessing, runs the model on each tile independently, translates the tile-local detections back into wall coordinates, then applies a global greedy NMS pass to merge duplicates from overlapping tiles. The model's built-in detection cap is raised to 500 per tile to avoid hard-capping recall on dense walls.

Metrics computed at evaluation include precision, recall, and F1 at a configurable operating point, mean matched box IoU, mean matched mask IoU, and COCO-style mAP. All metrics are written to a JSON file alongside per-image visualization overlays, P-R curves, score histograms, and box/mask IoU histograms.

### Algorithm 1: Hold Color Binning

Once Phase 1 has detected and masked individual holds, the next problem is figuring out which route each hold belongs to. At most gyms, all holds on the same route share a color, so color is the natural signal to cluster on. The color binning step takes each predicted hold mask and assigns it a color name from a fixed vocabulary: red, orange, yellow, green, blue, purple, pink, white, and black.

**CIELab** (`color_bin_lab`, currently the default): Converts the masked region to CIELab color space, which separates luminance from chroma and is perceptually more uniform than HSV. A hold is considered chromatic if its 75th-percentile chroma is >= 12 or its 95th-percentile chroma is >= 25, this two-condition gate means holds that are mostly chalked white but have some exposed color still get classified correctly. For chromatic holds, the top-quartile pixels by chroma are used to compute a median Lab value, which is then matched to the nearest reference centroid using Euclidean distance in a*b* space (ignoring L, since luminance varies too much with gym lighting to be a reliable discriminator). Achromatic holds fall back to white/black based on median L.

### Algorithm 2: Route Segmentation (DBSCAN)

Once holds have been color-binned, the plan is to use DBSCAN spatial clustering to group holds of the same color into candidate routes. Holds that are spatially close together and share a color are likely part of the same route. Volumes can always be added to a route, but cannot be scanned from. This step is not yet implemented beyond a stub.

### Model 2: Route Grade Classification

The final step would take a segmented route and classify its difficulty. Architecture possibly Random forest or gradient boost (TBD).
