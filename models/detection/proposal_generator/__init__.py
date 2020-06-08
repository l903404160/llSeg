from utils.registry import Registry

DET_PROPOSAL_GENERATOR_REGISTRY = Registry("DET_PROPOSAL_GENERATOR")


def proposal_generator_builder(cfg):
    builder = DET_PROPOSAL_GENERATOR_REGISTRY.get(cfg.MODEL.PROPOSAL_GENERATOR.NAME)
    proposal_generator = builder(cfg)
    return proposal_generator