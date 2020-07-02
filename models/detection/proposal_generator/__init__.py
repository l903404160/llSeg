from utils.registry import Registry

DET_PROPOSAL_GENERATOR_REGISTRY = Registry("DET_PROPOSAL_GENERATOR")

from . import rpn


def proposal_generator_builder(cfg, input_shape):
    builder = DET_PROPOSAL_GENERATOR_REGISTRY.get(cfg.MODEL.PROPOSAL_GENERATOR.NAME)
    proposal_generator = builder(cfg, input_shape)
    return proposal_generator