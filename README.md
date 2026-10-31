# ColorLab Studio models

Core ML models that ColorLab Studio downloads on first use. Nothing here is needed by projects that never use the feature.

## MoGe2ViTB560P8.mlpackage

Metric depth for **Depth of Field Advanced**. A Core ML conversion of **MoGe-2 ViT-B**
([microsoft/MoGe](https://github.com/microsoft/MoGe), weights
[Ruicheng/moge-2-vitb-normal](https://huggingface.co/Ruicheng/moge-2-vitb-normal) at revision
`ca5f0e07ff01d3e5a364c1d954ed12ee1814b368`, `model.pt` SHA-256
`16b8110e86d5dc5a849db120ca96ef3a223fd30b0c9146d1d81db504073da5f6`).

- **Trunk only:** encoder, neck and the points / mask / scale heads. The normal head is dropped. The input is fixed at
  560 x 420 (40 x 30 tokens), RGB 0...255.
- **Outputs:** `points` (1 x 3 x 420 x 560), `mask` (1 x 1 x 420 x 560, 1 = not sky), `scale` (1 x 1). The remaining
  post-processing (depth shift and focal length solve) runs in the app.
- **Precision:** FP16 compute, weights 8-bit k-means palettized per tensor, macOS 14+. It is within about 1.5 % of the
  PyTorch model.
- **Made with:** coremltools 9.0 and torch 2.15.0, by `tools/convert_moge.py moge2-vitb 40 30 f16.mlpackage` then
  `tools/palettize.py f16.mlpackage MoGe2ViTB560P8.mlpackage 8`.

| File | Bytes | SHA-256 |
|---|---|---|
| `MoGe2ViTB560P8.mlpackage/Manifest.json` | 617 | `35ae98e5bb92d247ea9f24b25018882cbd8ace0424837ca6be13ce47c07b83d4` |
| `MoGe2ViTB560P8.mlpackage/Data/com.apple.CoreML/model.mlmodel` | 258,750 | `8dd0614eb470a953808e149b31e390e71a4d3cfd2a55ca84416a52fac4463069` |
| `MoGe2ViTB560P8.mlpackage/Data/com.apple.CoreML/weights/weight.bin` | 101,698,560 | `6ed5901086e84bf60b06bd8d0aa531b7f6e519072769cde08fcf27fc48d7a503` |

## VideoDepthAnythingS518EncoderP8.mlpackage, VideoDepthAnythingS518Head32.mlpackage

Temporally stable relative depth for **Depth of Field Extra Advanced** (video clips). A Core ML conversion of **Video
Depth Anything Small** ([DepthAnything/Video-Depth-Anything](https://github.com/DepthAnything/Video-Depth-Anything),
weights `video_depth_anything_vits.pth`, Apache-2.0), split in two:

- **Encoder** (`VideoDepthAnythingS518EncoderP8`): DINOv2 ViT-S over one 518 x 392 RGB picture (0...255, 37 x 28 tokens)
  -> `f0` ... `f3`, the 1 x 1036 x 384 token maps of layers 3, 6, 9, 12. FP16 compute, weights 8-bit k-means palettized per
  tensor. Through the app its maps are within 0.1 % of the FP16 encoder's. Runs on the Neural Engine.
- **Head** (`VideoDepthAnythingS518Head32`): the temporal DPT head over a 32-frame window, `f0` ... `f3` as 32 x 1036 x 384
  -> `depth`, 32 x 392 x 518 relative disparity (ReLU: the sky 0). FP16. Run it on CPU and GPU only: its Neural Engine
  compile never finished.
- The split matches the repository's own forward to 7e-6. The app reproduces the repository's windowing
  (`infer_video_depth`: 32 frames, 10 overlap with keyframes, scale / shift alignment, 8-frame blend).
- **Precision:** macOS 14+.
- **Made with:** coremltools 9.0 and torch 2.15.0, by `tools/convert_vda.py <repo> video_depth_anything_vits.pth 518 392
  out 32`, then `tools/palettize.py out/vda_s_enc_518x392.mlpackage VideoDepthAnythingS518EncoderP8.mlpackage 8`.

| File | Bytes | SHA-256 |
|---|---|---|
| `VideoDepthAnythingS518EncoderP8.mlpackage/Manifest.json` | 617 | `5704062fe1ed2e5beced1808411b9f6671272f4357a354dc12d3e6130ee6320c` |
| `VideoDepthAnythingS518EncoderP8.mlpackage/Data/com.apple.CoreML/model.mlmodel` | 136,003 | `e83667fb59e32f3e4120b1035cbcbcb842708a81e53d6b281c2955e32775d39a` |
| `VideoDepthAnythingS518EncoderP8.mlpackage/Data/com.apple.CoreML/weights/weight.bin` | 22,038,976 | `c2113e960aeb34700309fd6d2e36c55e5418dc470569bbebc5e1b6f401464017` |
| `VideoDepthAnythingS518Head32.mlpackage/Manifest.json` | 617 | `38f27f62c3aca0e58fa92753483c05790b6fc7535e4d8acb93e19135ccd534d9` |
| `VideoDepthAnythingS518Head32.mlpackage/Data/com.apple.CoreML/model.mlmodel` | 183,159 | `c77ff4bc6939306c3b18657fa6fcb887c6271b55efcca213d3a8868c81dbc1dc` |
| `VideoDepthAnythingS518Head32.mlpackage/Data/com.apple.CoreML/weights/weight.bin` | 13,866,752 | `4fb91217a32a8613c2039d64cebb19c3791f129fff258411bad4dea42dfa769d` |

Video Depth Anything and its Small weights are Apache-2.0, Copyright (c) ByteDance; the DINOv2 backbone is Apache-2.0.

## ViTMatteS512P8.mlpackage

Soft subject alpha (hair, fine edges) for **Depth of Field Extra Advanced**. A Core ML conversion of **ViTMatte-S**
([hustvl/ViTMatte](https://github.com/hustvl/ViTMatte), MIT code; weights
[hustvl/vitmatte-small-composition-1k](https://huggingface.co/hustvl/vitmatte-small-composition-1k), Apache-2.0) at a
fixed 512 x 512 tile:

- **Inputs:** `image`, 512 x 512 RGB (0 ... 255; a BGRA pixel buffer works), and `trimap`, 512 x 512 one channel (0
  background, 128 unknown, 255 foreground). **Output:** `alpha`, 1 x 1 x 512 x 512 (FP16, 0 ... 1).
- The bicubic position-embedding resize is done once for the 32 x 32 token grid, and the relative-position lookup is plain
  indexing, so the graph is static.
- FP16 compute, weights 8-bit k-means palettized per tensor. Against the FP32 PyTorch model, alpha is within 0.0025 on
  average over a photo's unknown band.
- About 42 ms a tile on an M3's Neural Engine (60 on the GPU), first load about 6 s.
- The app runs it only on the tiles that hold the unknown band of a trimap made from Apple Vision's subject matte, with
  tiles at a 384 px stride, tent-blended.
- **Precision:** macOS 14+.
- **Made with:** coremltools 9.0, torch 2.15.0 and transformers, by `tools/convert_vitmatte.py 512
  vitmatte_s_512_f16.mlpackage`, then `tools/palettize.py vitmatte_s_512_f16.mlpackage ViTMatteS512P8.mlpackage 8`.

| File | Bytes | SHA-256 |
|---|---|---|
| `ViTMatteS512P8.mlpackage/Manifest.json` | 617 | `8d0f5355a11a4ba73ddb4581630f4444c05d32af208ea22a720a7698b323a118` |
| `ViTMatteS512P8.mlpackage/Data/com.apple.CoreML/model.mlmodel` | 296,199 | `0869ed973ad02062cb8d2c29a1863fb5f54e7b1bc2e5c4a72e70d36de3489f7c` |
| `ViTMatteS512P8.mlpackage/Data/com.apple.CoreML/weights/weight.bin` | 26,936,160 | `7ba3aad122e2b9e4bf4fa4e6044d813c3ec167cecb7f6274315f7603b53d7093` |

ViTMatte is MIT (code) and its Small composition-1k weights are Apache-2.0, Copyright (c) hustvl (Huazhong University of
Science and Technology). The weights were trained on Adobe's Composition-1k, a dataset whose own terms are
research-only; the weights' Apache-2.0 tag is what this repository relies on.

## License

MoGe and its weights are MIT, Copyright (c) Microsoft Corporation. Video Depth Anything and its Small weights are
Apache-2.0, Copyright (c) ByteDance. ViTMatte's Small composition-1k weights are Apache-2.0, Copyright (c) hustvl. The ViT backbones come from DINOv2 (Apache-2.0). `LICENSE` is MoGe's license file,
which carries both the MIT and the Apache-2.0 texts. The conversion scripts in `tools/` are MIT as well.
