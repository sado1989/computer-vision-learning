# Computer Vision From Scratch — Accelerated Plan

## How we’ll work
- Tooling: Python 3.11+, PyTorch, OpenCV, NumPy, Matplotlib, scikit-learn, Jupyter.
- Cadence: ~12–15 hrs/week. Weekly lab + mini-project checkpoints.
- Deliverables: Labs each week; mini-projects; 1 deployable demo; 1 capstone.

## Phase 0 — Prereqs (fast-track, 2–3 days)
- Python/NumPy refresh; vectorization; plotting.
- Math: linear algebra basics, gradients, probability intuition.
**Lab:** implement 2D convolution in NumPy; compare to OpenCV.

## Phase 1 — Classical CV (Week 1)
- Color spaces; filtering (box/gaussian/median), edges (Sobel, Canny), corners (Harris/Shi-Tomasi), descriptors (HOG/SIFT/ORB), histogram equalization/CLAHE.
**Lab:** full Canny pipeline (gradients → NMS → hysteresis).
**Mini-Project A:** count/segment simple objects (coins/screws). Report failure cases.

## Phase 2 — Deep Learning Foundations (Week 2)
- Tensors/autograd; optimizers (SGD/Adam); regularization; LR schedules.
- CNN building blocks; over/underfitting diagnosis; transfer learning.
**Lab:** small CNN on CIFAR-10; confusion matrix + curves.
**Mini-Project B:** fine-tune ResNet-18 on your tiny custom dataset.

## Phase 3 — Detection & Segmentation (Weeks 3–4)
- Detection: Faster R-CNN/RetinaNet/YOLOv8; focal loss; IoU/GIoU; NMS.
- Segmentation: U-Net/DeepLab; Dice vs BCE; class imbalance.
- Metrics: mAP@[.5:.95], PR, IoU/Dice; dataset formats (COCO/VOC).
**Labs:** train YOLOv8n on 20–50 labeled images; train U-Net for binary masks.
**Mini-Project C:** choose detection OR segmentation; deliver dataset card + eval.

## Phase 4 — Geometry & 3D (Week 5)
- Pinhole camera; intrinsics/extrinsics; distortion; homography; epipolar geometry; RANSAC; basic stereo.
**Labs:** camera calibration; panorama via homography.
**Mini-Project D:** AR overlay or panorama stitcher.

## Phase 5 — Robustness & Deployment (Week 6)
- Data curation; hard negative mining; augmentations; calibration; mixed precision.
- Export: ONNX/TorchScript; basic benchmarking; quantization overview.
**Lab:** AMP training speedup; export + run ONNX inference.
**Mini-Project E:** CLI or lightweight web demo with reproducible env.

## Capstone (Week 7)
- One of: industrial/medical segmentation, multi-object video tracking, or 3D/AR project.
**Deliverables:** 6–10 page write-up; clean repo; trained weights; demo video/gif.

## Weekly checklist (accelerated)
- W0: environment + conv2d NumPy; Canny start
- W1: classical CV complete; Mini-Project A
- W2: CNNs + transfer learning; Mini-Project B
- W3: detection training + eval
- W4: segmentation training + eval; Mini-Project C
- W5: camera models + homography; Mini-Project D
- W6: robustness + deployment; Mini-Project E
- W7: capstone build & write-up

## Exit criteria (you’re ready to advance if…)
- You can implement IoU/Dice, explain mAP, and choose losses rationally.
- You can calibrate a camera and compute/verify a homography with RANSAC.
- You can export a model and measure end-to-end latency.

## Reading path (optional)
Szeliski (CVA&A), Goodfellow (DL), Dive into Deep Learning; skim AlexNet→ResNet→U-Net→YOLO/ViT/DETR for ideas.

## Repo structure we’ll use
- `notebooks/` labs & experiments
- `src/cv_guide/` reusable code
- `data/{raw,processed}/` (not committed except .gitkeep)
- `models/` trained weights (use Git LFS if large)
- `reports/figures/` visuals

