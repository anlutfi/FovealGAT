from models.pvig_gaze import pvig_ti_224_gelu


def pyramid_feature_extractor(device='cpu'):
    model = pvig_ti_224_gelu()
    return model.stem.to(device)