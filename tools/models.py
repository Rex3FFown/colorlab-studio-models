# Loads MoGe-2 (PyTorch) from <tools>/../weights/<which>/model.pt (huggingface.co/Ruicheng/moge-2-vitb-normal, model.pt)
# with the MoGe code (github.com/microsoft/MoGe) on the path at <tools>/../src/moge.
import sys, os
S = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [f'{S}/src/moge']

def load_moge(which):
    from moge.model.v2 import MoGeModel
    return MoGeModel.from_pretrained(f'{S}/weights/{which}/model.pt').eval()
