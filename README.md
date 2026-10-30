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

## License

MoGe and its weights are MIT, Copyright (c) Microsoft Corporation. The ViT backbone comes from DINOv2 (Apache-2.0).
`LICENSE` is MoGe's license file, which carries both texts. The conversion scripts in `tools/` are MIT as well.
