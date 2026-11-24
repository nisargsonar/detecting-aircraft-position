# Morning report — Mon Nov 16 22:22

All numbers measured on IN_ODD runways; crops from the released YOLO detector boxes for every model. Paired comparisons are on identical runways; 95% CIs are paired bootstraps of the median difference (an interval containing 0 means no significant difference).

## Headline (measured; see sections for detail)
- Run A finished 10 of 10 epochs.
- Corners, Bing: every model of ours beats YOLOv8-Pose; Run A by -0.33 (95% CI -0.37 to -0.28, n = 1086) px.
- Position, Bing: no model differs significantly from YOLO (all CIs contain 0); Run A vs YOLO -19.73 (95% CI -62.37 to 12.47, n = 1086) m.
- Run A vs the Bing-subset ResNet-50 on Bing: position -36.81 (95% CI -62.92 to -7.8, n = 1086) m (significant).
- 16-point vs 4-corner PnP, same model and runways: 2.16 m (95% CI -14.53 to 25.46, n = 1080): no measured gain.
- CNN vs ViT: tied on Bing (-0.02 (95% CI -0.05 to 0.02, n = 1086) px); on unseen Flight Simulator DINOv2 is better by -62.63 m (95% CI -95.22 to -33.95, n = 1088).
- Joint 2-runway PnP cuts median position error 8-12x for every model, n = 55-92 images (section 4).

## 1. CNN vs ViT (Bing Maps, the 662 Nov-6 test runways)
Both models trained on the same 4,599 Bing crops (723 val). DINOv2 epochs: 10 of 10.

| Model | n | Corner px (median) | ≤1 px | Position m (median) | Near / mid / far m | Lateral / glide err |
|---|---|---|---|---|---|---|
| YOLOv8-Pose (released) | 662 | 1.39 | 25.8% | 302.5 | 119.6 (n=189) / 377.1 (n=239) / 495.5 (n=234) | 0.155° / 0.209° |
| ResNet-50 (ours) | 662 | 1.12 | 39.3% | 320.1 | 100.9 (n=189) / 375.6 (n=239) / 528.0 (n=234) | 0.149° / 0.239° |
| DINOv2 ViT-S/14 (ours) | 662 | 1.13 | 39.7% | 349.6 | 88.6 (n=189) / 404.4 (n=239) / 632.1 (n=234) | 0.124° / 0.243° |

- dinov2 vs resnet50 corner px: 1.13 vs 1.12 px, difference 0.0 px (95% CI -0.04 to 0.05; n = 662)
- dinov2 vs resnet50 pos m: 349.56 vs 320.08 m, difference 29.48 m (95% CI -15.37 to 63.53; n = 662)
- resnet50 vs yolo corner px: 1.12 vs 1.39 px, difference -0.27 px (95% CI -0.32 to -0.22; n = 662)
- resnet50 vs yolo pos m: 320.08 vs 302.47 m, difference 17.61 m (95% CI -34.66 to 65.95; n = 662)
- dinov2 vs yolo corner px: 1.13 vs 1.39 px, difference -0.26 px (95% CI -0.32 to -0.21; n = 662)
- dinov2 vs yolo pos m: 349.56 vs 302.47 m, difference 47.09 m (95% CI -10.15 to 88.71; n = 662)

## 2. Run A training (ResNet-50, 16 keypoints, Bing + X-Plane): 10 of 10 epochs, n_train = 15925, n_val = 1723

| Epoch | Train loss | Val loss | Val corner px median | Val corner px mean |
|---|---|---|---|---|
| 1 | 4.237 | 3.089 | 1.625 | 4.699 |
| 2 | 2.514 | 2.856 | 1.452 | 4.465 |
| 3 | 2.288 | 2.726 | 1.378 | 4.294 |
| 4 | 2.182 | 2.65 | 1.339 | 4.222 |
| 5 | 2.09 | 2.635 | 1.281 | 4.221 |
| 6 | 1.996 | 2.62 | 1.27 | 4.155 |
| 7 | 1.911 | 2.534 | 1.21 | 4.091 |
| 8 | 1.813 | 2.519 | 1.173 | 4.07 |
| 9 | 1.763 | 2.504 | 1.162 | 4.045 |
| 10 | 1.714 | 2.499 | 1.143 | 4.039 |

Validation split by renderer (true box with fixed small jitter, 4 corners):

| Renderer | n | Median px | Mean px | ≤1 px | >5 px | >20 px | Error share of worst 5% |
|---|---|---|---|---|---|---|---|
| bingmaps | 723 | 1.07 | 1.31 | 43.0% | 1.0% | 0.0% | 15.5% |
| xplane | 1000 | 1.25 | 6.02 | 33.1% | 2.1% | 1.3% | 78.5% |

## 2a. Run A (ResNet-50, 16 keypoints, Bing + X-Plane): bing_all

| Model | n | Corner px (median) | ≤1 px | Position m (median) | Near / mid / far m | Lateral / glide err |
|---|---|---|---|---|---|---|
| Run A, 4-corner PnP | 1086 | 1.06 | 44.4% | 288.8 | 73.1 (n=320) / 351.8 (n=385) / 564.5 (n=381) | 0.133° / 0.212° |
| Run A, 16-point PnP | 1086 | 1.06 | 44.4% | 290.5 | 84.7 (n=320) / 369.9 (n=385) / 588.9 (n=381) | 0.128° / 0.205° |
| YOLOv8-Pose, same runways | 1086 | 1.39 | 25.1% | 308.5 | 107.3 (n=320) / 394.2 (n=385) / 525.3 (n=381) | 0.152° / 0.22° |

- 16-point vs 4-corner PnP, same model and runways (position): 290.47 vs 288.31 m, difference 2.16 m (95% CI -14.53 to 25.46; n = 1080)
- runA16 vs yolo pos m: 290.47 vs 306.09 m, difference -15.63 m (95% CI -54.48 to 21.65; n = 1080)
- runA vs yolo corner px: 1.06 vs 1.39 px, difference -0.33 px (95% CI -0.37 to -0.28; n = 1086)

## 2b. Run A (ResNet-50, 16 keypoints, Bing + X-Plane): flsim

| Model | n | Corner px (median) | ≤1 px | Position m (median) | Near / mid / far m | Lateral / glide err |
|---|---|---|---|---|---|---|
| Run A, 4-corner PnP | 1088 | 1.51 | 16.1% | 335.2 | 80.5 (n=335) / 410.5 (n=403) / 753.7 (n=350) | 0.133° / 0.216° |
| Run A, 16-point PnP | 1088 | 1.51 | 16.1% | 343.0 | 80.1 (n=335) / 403.8 (n=403) / 812.3 (n=350) | 0.154° / 0.215° |

- 16-point vs 4-corner PnP, same model and runways (position): 342.97 vs 335.42 m, difference 7.55 m (95% CI -11.74 to 39.16; n = 1076)

## 2c. All our models vs YOLOv8-Pose on the same Bing runways (larger test set)

| Model | n | Corner px (median) | ≤1 px | Position m (median) | Near / mid / far m | Lateral / glide err |
|---|---|---|---|---|---|---|
| YOLOv8-Pose (released) | 1086 | 1.39 | 25.1% | 308.5 | 107.3 (n=320) / 394.2 (n=385) / 525.3 (n=381) | 0.152° / 0.22° |
| ResNet-50, Bing subset | 1086 | 1.13 | 38.1% | 325.6 | 97.1 (n=320) / 364.8 (n=385) / 530.9 (n=381) | 0.146° / 0.234° |
| DINOv2, Bing subset | 1086 | 1.12 | 40.4% | 333.9 | 93.9 (n=320) / 383.2 (n=385) / 636.0 (n=381) | 0.133° / 0.239° |
| Run A, 4-corner PnP | 1086 | 1.06 | 44.4% | 288.8 | 73.1 (n=320) / 351.8 (n=385) / 564.5 (n=381) | 0.133° / 0.212° |
| Run A, 16-point PnP | 1086 | 1.06 | 44.4% | 290.5 | 84.7 (n=320) / 369.9 (n=385) / 588.9 (n=381) | 0.128° / 0.205° |

- resnet50 bing vs yolo corner px: 1.13 vs 1.39 px, difference -0.26 px (95% CI -0.3 to -0.21; n = 1086)
- resnet50 bing vs yolo pos m: 325.57 vs 308.5 m, difference 17.07 m (95% CI -27.31 to 46.98; n = 1086)
- dinov2 bing vs yolo corner px: 1.12 vs 1.39 px, difference -0.27 px (95% CI -0.32 to -0.23; n = 1086)
- dinov2 bing vs yolo pos m: 333.86 vs 308.5 m, difference 25.36 m (95% CI -18.48 to 57.2; n = 1086)
- runA vs yolo corner px: 1.06 vs 1.39 px, difference -0.33 px (95% CI -0.37 to -0.28; n = 1086)
- runA vs yolo pos m: 288.76 vs 308.5 m, difference -19.73 m (95% CI -62.37 to 12.47; n = 1086)
- dinov2 vs resnet50 corner px: 1.12 vs 1.13 px, difference -0.02 px (95% CI -0.05 to 0.02; n = 1086)
- dinov2 vs resnet50 pos m: 333.86 vs 325.57 m, difference 8.28 m (95% CI -19.66 to 37.09; n = 1086)
- runA vs resnet50 corner px: 1.06 vs 1.13 px, difference -0.07 px (95% CI -0.1 to -0.04; n = 1086)
- runA vs resnet50 pos m: 288.76 vs 325.57 m, difference -36.81 m (95% CI -62.92 to -7.8; n = 1086)

## 3. Unseen renderer: Flight Simulator (our models only; released YOLO may have seen it)

| Model | n | Corner px (median) | ≤1 px | Position m (median) | Near / mid / far m | Lateral / glide err |
|---|---|---|---|---|---|---|
| resnet50_bing | 1088 | 1.43 | 19.9% | 318.4 | 107.8 (n=335) / 374.3 (n=403) / 662.4 (n=350) | 0.171° / 0.23° |
| dinov2_bing | 1088 | 1.28 | 27.7% | 255.7 | 77.1 (n=335) / 321.8 (n=403) / 513.0 (n=350) | 0.116° / 0.199° |
| runA | 1088 | 1.51 | 16.1% | 335.2 | 80.5 (n=335) / 410.5 (n=403) / 753.7 (n=350) | 0.133° / 0.216° |

- DINOv2 vs ResNet-50 corner px: 1.28 vs 1.43 px, difference -0.15 px (95% CI -0.19 to -0.11; n = 1088)
- DINOv2 vs ResNet-50 position: 255.75 vs 318.38 m, difference -62.63 m (95% CI -95.22 to -33.95; n = 1088)
- Run A vs ResNet-50 (Bing only) corner px: 1.51 vs 1.43 px, difference 0.07 px (95% CI 0.02 to 0.11; n = 1088)
- Run A vs ResNet-50 (Bing only) position: 335.25 vs 318.38 m, difference 16.87 m (95% CI -26.0 to 52.32; n = 1088)

## 4. Multi-runway joint PnP and wrong-runway check (images with 2 runways)

| Run | Test | n images | One runway m | Both runways m | 95% CI of difference | Joint better | Wrong-runway correct |
|---|---|---|---|---|---|---|---|
| dinov2_bing | bing662 | 55 | 488.1 | 43.1 | -508.7 to -256.3 | 92.7% | 96.4% |
| dinov2_bing | bing_all | 92 | 493.2 | 41.6 | -506.7 to -321.3 | 92.4% | 96.7% |
| resnet50_bing | bing662 | 55 | 386.6 | 48.6 | -472.4 to -246.9 | 90.9% | 96.4% |
| resnet50_bing | bing_all | 92 | 407.7 | 44.5 | -481.3 to -281.0 | 92.4% | 96.7% |
| runA | bing_all | 92 | 421.4 | 45.9 | -467.9 to -269.5 | 92.4% | 95.7% |

## Notes
- All numbers measured; IN_ODD runways only; n stated per entry.
- Crops use the released YOLO detector boxes (IoU>=0.3, score>=0.25) for every model.
- YOLO keypoint comparisons use Bing only; Flight Simulator compares our own models (released YOLO may have seen it).
- Released YOLO training config uses the LARD test split as validation (possible optimistic bias in YOLO numbers).
- 16-point PnP returned no physically valid pose on 6 of 1,086 Bing and 12 of 1,088 Flight Simulator runways; paired 16-vs-4 comparisons drop those (n = 1,080 / 1,076), the per-model medians do not.
- Run A is the only model that saw X-Plane in training; ResNet-50 and DINOv2 rows trained on the 4,599-crop Bing subset.
- Validation corner errors use the true box with a fixed small jitter; test errors use detector-box crops.
