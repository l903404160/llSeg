import torch
import torch.nn as nn

from models.detection.backbone import backbone_builder
from models.detection.anchorfree_heads import AnchorFreeHead
from structures import ImageList
from models.detection.modules.postprocessing import detector_postprocess


class GeneralizedAnchorFreeDetector(nn.Module):
    def __init__(self, cfg):
        super(GeneralizedAnchorFreeDetector, self).__init__()
        self.num_classes = cfg.MODEL.ANCHORFREE_HEADS.NUM_CLASSES

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
            losses = self.head(features, batched_inputs)
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

