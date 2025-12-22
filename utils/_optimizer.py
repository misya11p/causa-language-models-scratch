from muon import MuonWithAuxAdam


def get_optimizer(model):
    params_muon = []
    params_adamw = []
    for name, parameter in model.named_parameters():
        if (parameter.ndim >= 2) and "transformer_layers." in name:
            params_muon.append(parameter)
        else:
            params_adamw.append(parameter)

    optimizer = MuonWithAuxAdam([
        dict(params=params_muon, use_muon=True),
        dict(
            params=params_adamw,
            lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01,
            use_muon=False
        )
    ])
    return optimizer
