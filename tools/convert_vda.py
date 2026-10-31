# convert_vda.py <vda repo> <vits.pth> <W> <H> <outdir> [T=32]: Video Depth Anything Small (Apache-2.0) as two Core ML
# programs: the DINOv2 ViT-S encoder per frame (image -> 4 token maps) and the temporal DPT head over a T-frame window
# (4 x (T, N, 384) -> (T, H, W) relative disparity). No rank-5 tensors: with one clip per window the head's
# (B, C, T, h, w) round trip is the identity, so the temporal modules run on (T, C, h, w) directly.
import sys, os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, coremltools as ct
repo, pth, W, H, outdir = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
T = int(sys.argv[6]) if len(sys.argv) > 6 else 32
sys.path.insert(0, repo)
from video_depth_anything.video_depth import VideoDepthAnything

m = VideoDepthAnything(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
m.load_state_dict(torch.load(pth, map_location='cpu'), strict=True)
m.eval()
ph, pw = H // 14, W // 14
bb = m.pretrained
with torch.no_grad():
    pe = bb.interpolate_pos_encoding(torch.zeros(1, ph * pw + 1, bb.embed_dim), H, W).detach().clone()
bb.interpolate_pos_encoding = lambda x, w, h: pe  # fixed grid: the bicubic resize done once here

class Enc(nn.Module):
    def __init__(s):
        super().__init__()
        s.bb = bb
        s.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        s.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    def forward(s, image):
        f = s.bb.get_intermediate_layers((image - s.mean) / s.std, [2, 5, 8, 11], return_class_token=True)
        return tuple(x[0] for x in f)

def mm(tm, x):  # TemporalModule on (T, C, h, w)
    tt = tm.temporal_transformer
    t, c, h, w = x.shape
    y = tt.proj_in(tt.norm(x).permute(0, 2, 3, 1).reshape(t, h * w, c))
    for blk in tt.transformer_blocks:
        y, _ = blk(y, video_length=t)
    return tt.proj_out(y).reshape(t, h, w, c).permute(0, 3, 1, 2) + x

class Head(nn.Module):
    def __init__(s):
        super().__init__()
        s.hd = m.head
    def forward(s, f1, f2, f3, f4):
        hd, sc = s.hd, s.hd.scratch
        out = []
        for i, x in enumerate([f1, f2, f3, f4]):
            x = x.permute(0, 2, 1).reshape(x.shape[0], x.shape[-1], ph, pw)
            out.append(hd.resize_layers[i](hd.projects[i](x)))
        l1, l2, l3, l4 = out
        l3, l4 = mm(hd.motion_modules[0], l3), mm(hd.motion_modules[1], l4)
        r1, r2, r3, r4 = sc.layer1_rn(l1), sc.layer2_rn(l2), sc.layer3_rn(l3), sc.layer4_rn(l4)
        p4 = mm(hd.motion_modules[2], sc.refinenet4(r4, size=r3.shape[2:]))
        p3 = mm(hd.motion_modules[3], sc.refinenet3(p4, r3, size=r2.shape[2:]))
        p1 = sc.refinenet1(sc.refinenet2(p3, r2, size=r1.shape[2:]), r1)
        o = F.interpolate(sc.output_conv1(p1), (ph * 14, pw * 14), mode='bilinear', align_corners=True)
        return F.relu(sc.output_conv2(o))[:, 0]

if __name__ == '__main__':
    os.makedirs(outdir, exist_ok=True)
    enc, head = Enc().eval(), Head().eval()
    x = torch.rand(T, 3, H, W)
    with torch.no_grad():
        ref = m(((x - enc.mean) / enc.std)[None])[0]      # the repo's own forward (rank 5), fp32
        feats = [torch.cat(z, 0) for z in zip(*[enc(x[i:i + 1]) for i in range(T)])]
        mine = head(*feats)
        print('split vs repo forward: max |diff|', (mine - ref).abs().max().item(), 'ref range', ref.min().item(), ref.max().item())
        te = torch.jit.trace(enc, x[:1], check_trace=False)
        th = torch.jit.trace(head, tuple(feats), check_trace=False)
    np.savez(f'{outdir}/ref.npz', x=x.numpy(), ref=ref.numpy(), **{f'f{i}': f.numpy() for i, f in enumerate(feats)})
    me = None if os.environ.get('ONLY') == 'head' else ct.convert(te, inputs=[ct.ImageType(name='image', shape=(1, 3, H, W), scale=1 / 255, color_layout=ct.colorlayout.RGB)],
                    outputs=[ct.TensorType(name=f'f{i}') for i in range(4)], convert_to='mlprogram',
                    compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.macOS14, skip_model_load=True)
    if me: me.short_description = f'Video Depth Anything Small encoder (DINOv2 ViT-S), {W}x{H}, Apache-2.0'
    if me: me.save(f'{outdir}/vda_s_enc_{W}x{H}.mlpackage'); print('saved encoder')
    mh = ct.convert(th, inputs=[ct.TensorType(name=f'f{i}', shape=tuple(feats[i].shape)) for i in range(4)],
                    outputs=[ct.TensorType(name='depth')], convert_to='mlprogram',
                    compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.macOS14, skip_model_load=True)
    mh.short_description = f'Video Depth Anything Small temporal DPT head, {T} frames of {W}x{H}, Apache-2.0'
    mh.save(f'{outdir}/vda_s_head_{W}x{H}_t{T}.mlpackage'); print('saved head')
