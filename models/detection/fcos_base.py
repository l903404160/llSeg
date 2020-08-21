import torch
import torch.nn as nn

from models.detection.backbone import backbone_builder
from models.detection.anchorfree_heads import det_onestage_anchorfree_builder
from models.detection.anchorfree_heads.anchorfree_heads import AnchorFreeHead
from structures import ImageList
from models.detection.modules.postprocessing import detector_postprocess

# Debug
from models.detection.anchorfree_heads.fcos.fcos_two_head import TwoStageFCOSHead

class GeneralizedFCOSDetector(nn.Module):
    def __init__(self, cfg):
        super(GeneralizedFCOSDetector, self).__init__()
        self.num_classes = cfg.MODEL.FCOS_HEADS.NUM_CLASSES

        # Backbone
        self.backbone = backbone_builder(cfg)
        # Anchorfree Head
        self.head = AnchorFreeHead(cfg)

        # other operations
        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
        Returns:
        """
        # Process  images
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        if self.training:
            losses = self.head(features, batched_inputs, image_sizes=images.image_sizes)
            return losses
        else:
            detections = self.head(features, image_sizes=images.image_sizes)
            detections = self._post_process(detections, batched_inputs)
            return detections

    def _post_process(self, instances, batched_inputs):
        assert not self.training
        processed_output = []

        height = batched_inputs[0]['height']
        width = batched_inputs[0]['width']
        instances = instances[0]
        instances = detector_postprocess(instances, output_height=height, output_width=width)

        processed_output.append({'instances': instances})
        return processed_output

    def preprocess_image(self, batched_inputs):
        # Normalization
        # compute heatmap, tag, regr
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images


class GeneralizedOneStageAnchorFreeDetector(nn.Module):
    def __init__(self, cfg):
        super(GeneralizedOneStageAnchorFreeDetector, self).__init__()

        # Backbone
        self.backbone = backbone_builder(cfg)
        # Anchorfree Head
        self.head = det_onestage_anchorfree_builder(cfg, self.backbone.output_shape())
        # Two stage Head
        # self.two_head = TwoStageFCOSHead(cfg)

        # other operations
        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
        Returns:
        """
        # Process  images
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        proposals, proposal_losses = self.head(images, features, gt_instances)

        if self.training:
            # two_losses = self.two_head(features, proposals['proposals'], gt_instances)
            # Switch open the two_losses or not
            # proposal_losses.update(two_losses)
            return proposal_losses
        else:
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                    proposals, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    def preprocess_image(self, batched_inputs):
        # Normalization
        # compute heatmap, tag, regr
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images
