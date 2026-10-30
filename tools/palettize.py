# palettize.py <in.mlpackage> <out.mlpackage> <nbits>: k-means weight palettization (per tensor: macOS 14 runs it)
import sys, coremltools as ct, coremltools.optimize.coreml as cto
m = ct.models.MLModel(sys.argv[1], skip_model_load=True)
cfg = cto.OptimizationConfig(global_config=cto.OpPalettizerConfig(mode='kmeans', nbits=int(sys.argv[3]), granularity='per_tensor'))
p = cto.palettize_weights(m, cfg)
p.short_description = m.short_description + f', {sys.argv[3]}-bit palettized'
p.save(sys.argv[2]); print('saved', sys.argv[2])
