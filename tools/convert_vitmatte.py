# convert_vitmatte.py <size> <out.mlpackage> [nbits]: ViTMatte-S (hustvl/vitmatte-small-composition-1k, Apache-2.0)
# at a fixed size x size tile; inputs: image (RGB, 0..255) and trimap (gray: 0 bg, 128 unknown, 255 fg), output alpha.
import sys, os, time, numpy as np, torch, torch.nn as nn, coremltools as ct
from transformers import VitMatteForImageMatting

class Wrap(nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m
    def forward(self, image, trimap):
        return self.m(pixel_values=torch.cat([(image - 0.5) / 0.5, trimap], 1)).alphas

if __name__ == '__main__':
    size, out = int(sys.argv[1]), sys.argv[2]
    nbits = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    m = VitMatteForImageMatting.from_pretrained('hustvl/vitmatte-small-composition-1k').eval()
    import transformers.models.vitdet.modeling_vitdet as vd
    emb = m.backbone.embeddings
    g = size // 16
    with torch.no_grad():  # fixed grid: the bicubic pos-embed resize done once here (Core ML has no bicubic upsample)
        pe = emb.get_absolute_positions(emb.position_embeddings, True, g, g).detach().clone()
    emb.get_absolute_positions = lambda *a: pe
    def rel(q, k, rp):  # the scripted lookup as plain indexing (q, k are ints at trace time)
        n = int(2 * max(q, k) - 1)
        if rp.shape[0] != n:
            rp = nn.functional.interpolate(rp.reshape(1, rp.shape[0], -1).permute(0, 2, 1), size=n, mode='linear').reshape(-1, n).permute(1, 0)
        c = (torch.arange(q)[:, None] * max(k / q, 1.0) - torch.arange(k)[None, :] * max(q / k, 1.0) + (k - 1) * max(q / k, 1.0)).long()
        return rp.detach()[c]
    vd.get_rel_pos = rel
    w = Wrap(m).eval()
    im, tr = torch.rand(1, 3, size, size), torch.rand(1, 1, size, size)
    with torch.no_grad():
        tt = torch.jit.trace(w, (im, tr), check_trace=False)
    ml = ct.convert(tt, inputs=[ct.ImageType(name='image', shape=im.shape, scale=1 / 255, color_layout=ct.colorlayout.RGB),
                                ct.ImageType(name='trimap', shape=tr.shape, scale=1 / 255, color_layout=ct.colorlayout.GRAYSCALE)],
                    outputs=[ct.TensorType(name='alpha')], convert_to='mlprogram',
                    compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.macOS14)
    if nbits < 16:
        import coremltools.optimize.coreml as cto
        cfg = cto.OptimizationConfig(global_config=cto.OpPalettizerConfig(mode='kmeans', nbits=nbits, granularity='per_tensor'))
        ml = cto.palettize_weights(ml, cfg)
    ml.short_description = f'ViTMatte-S {size}x{size} tile (hustvl/vitmatte-small-composition-1k, Apache-2.0)' + (f', {nbits}-bit palettized' if nbits < 16 else ', fp16')
    ml.save(out); print('saved', out)
