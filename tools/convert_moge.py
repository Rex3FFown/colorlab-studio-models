# convert_moge.py <moge2-vits|moge2-vitb> <cols> <rows> <out.mlpackage> [nbits]
# MoGe-2 trunk (encoder, neck, points / mask / scale heads; no normal head) at a fixed token grid, image 0...1 in.
import sys, os, numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import models
from moge.utils.geometry_torch import normalized_view_plane_uv
import coremltools as ct

class Trunk(nn.Module):
    def __init__(self, m, rows, cols):
        super().__init__()
        self.m, self.rows, self.cols = m, rows, cols
        a = cols / rows
        for l in range(5):
            uv = normalized_view_plane_uv(width=cols * 2 ** l, height=rows * 2 ** l, aspect_ratio=a).permute(2, 0, 1)[None]
            self.register_buffer(f'uv{l}', uv)
        bb = m.encoder.backbone
        with torch.no_grad():
            pe = bb.interpolate_pos_encoding(torch.zeros(1, rows * cols + 1, bb.embed_dim), rows * 14, cols * 14)
        self.register_buffer('pe', pe.detach().clone())
        bb.interpolate_pos_encoding = lambda x, h, w: self.pe  # fixed grid: bicubic pos-embed resampling done once here

    def forward(self, image):
        enc, rows, cols = self.m.encoder, self.rows, self.cols
        x = (image - enc.image_mean) / enc.image_std
        feats = enc.backbone.get_intermediate_layers(x, n=enc.intermediate_layers, return_class_token=True)
        f0 = torch.stack([p(f.permute(0, 2, 1).unflatten(2, (rows, cols)).contiguous()) for p, (f, c) in zip(enc.output_projections, feats)], dim=1).sum(dim=1)
        fs = [torch.cat([f0, self.uv0], 1), self.uv1, self.uv2, self.uv3, self.uv4]
        fs = self.m.neck(fs)
        pts = self.m.points_head(fs)[-1]                     # 1, 3, h, w (16 x the token grid)
        xy, z = pts[:, :2], torch.exp(pts[:, 2:3])            # remap 'exp'
        pts = torch.cat([xy * z, z], 1)
        mask = torch.sigmoid(self.m.mask_head(fs)[-1])        # 1, 1, h, w: 1 = valid (not sky)
        scale = torch.exp(self.m.scale_head(feats[-1][1]))    # 1, 1: metres per point-map unit
        return pts, mask, scale

if __name__ == '__main__':
    name, cols, rows, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    m = models.load_moge(name)
    m.onnx_compatible_mode = True
    for h in ['normal_head']:
        if hasattr(m, h): delattr(m, h)
    t = Trunk(m, rows, cols).eval()
    ex = torch.rand(1, 3, rows * 14, cols * 14)
    with torch.no_grad():
        ref = t(ex)
        tr = torch.jit.trace(t, ex, check_trace=False)
    ml = ct.convert(tr, inputs=[ct.ImageType(name='image', shape=ex.shape, scale=1 / 255, color_layout=ct.colorlayout.RGB)],
                    outputs=[ct.TensorType(name='points'), ct.TensorType(name='mask'), ct.TensorType(name='scale')],
                    convert_to='mlprogram', compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.macOS14)
    nbits = int(sys.argv[5]) if len(sys.argv) > 5 else 16
    if nbits < 16:
        import coremltools.optimize.coreml as cto
        cfg = cto.OptimizationConfig(global_config=cto.OpPalettizerConfig(mode='kmeans', nbits=nbits, granularity='per_tensor'))
        ml = cto.palettize_weights(ml, cfg)
    ml.short_description = f'MoGe-2 {name} trunk, {cols}x{rows} tokens ({cols*14}x{rows*14}), MIT (github.com/microsoft/MoGe)'
    ml.save(out)
    print('saved', out, [r.shape for r in ref])
