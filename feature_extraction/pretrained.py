from models.pvig_gaze import pvig_ti_224_gelu


def pyramid_feature_extractor(device='cpu'):
    model = pvig_ti_224_gelu() # straight from GazeGNN: https://arxiv.org/abs/2305.18221
    return model.stem.to(device)