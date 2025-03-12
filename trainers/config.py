def get_dataset_specified_config(dataset):
    """Get dataset specific."""
    cfg = {
        "StanfordCars": {
            "TRAINER.MMRL.N_REP_TOKENS": 5,
            "TRAINER.MMRL.REG_WEIGHT": 7.0,
        },
        "FGVCAircraft": {
            "TRAINER.MMRL.N_REP_TOKENS": 5,
            "TRAINER.MMRL.REG_WEIGHT": 6.0,
        },
        "SUN397": {
            "TRAINER.MMRL.N_REP_TOKENS": 5,
            "TRAINER.MMRL.REG_WEIGHT": 6.0,
        },
        "DescribableTextures": {
            "TRAINER.MMRL.N_REP_TOKENS": 5,
            "TRAINER.MMRL.REG_WEIGHT": 6.0,
        },
        "Food101": {
            "TRAINER.MMRL.N_REP_TOKENS": 5,
            "TRAINER.MMRL.REG_WEIGHT": 5.0,
        },
        "OxfordFlowers": {
            "TRAINER.MMRL.N_REP_TOKENS": 5,
            "TRAINER.MMRL.REG_WEIGHT": 4.0,
        },
        "UCF101": {
            "TRAINER.MMRL.N_REP_TOKENS": 5,
            "TRAINER.MMRL.REG_WEIGHT": 3.0,
        },
        "ImageNet": {
            "TRAINER.MMRL.N_REP_TOKENS": 5,
            "TRAINER.MMRL.REG_WEIGHT": 0.5,
        },
        "Caltech101": {
            "TRAINER.MMRL.N_REP_TOKENS": 5,
            "TRAINER.MMRL.REG_WEIGHT": 0.5,
        },
        "OxfordPets": {
            "TRAINER.MMRL.N_REP_TOKENS": 5,
            "TRAINER.MMRL.REG_WEIGHT": 0.2,
        },
        "EuroSAT": {
            "TRAINER.MMRL.REP_DIM": 2048,
            "TRAINER.MMRL.N_REP_TOKENS": 5,
            "TRAINER.MMRL.REG_WEIGHT": 0.01,
        },
    }.get(dataset, {})

    return [item for pair in cfg.items() for item in pair]