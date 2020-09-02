import torch
import torch.nn as nn
import math

import torch.nn.functional as F

from layers import cat, ml_nms
from utils.nn.focal_loss import sigmoid_focal_loss_jit
from utils.nn.fcos_loss import IOULoss
from models.detection.anchorfree_heads.fcos.fcos_tools import compute_locations
from models.detection.anchorfree_heads.search_head.genotype import PRIMITIVES, Genotype, Genotype_w_box, parse_darts, parse_direct
from models.detection.anchorfree_heads.search_head.operations import *
from structures import Instances, Boxes
from models.detection.anchorfree_heads.search_head.cells import *

from models.detection.modules.postprocessing import detector_postprocess


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class SearchHead(nn.Module):
    def __init__(self, C_in, C, num_classes, layers, criterion, steps=4, multiplier=4):
        """
        Args:
            C_in: in_channels
            C:  out_channels
            num_classes:  number of classes
            layers:  number of layers
            criterion:  Loss criterion
            steps:  cell steps
            multiplier:  Cell multiplier. Because the output of a Cell is concated.
             Thus the multiplier is needed to specify the output channel of a feature
        """
        super(SearchHead, self).__init__()
        self._C_in = C_in
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier
        self._in_features = ['p3','p4','p5','p6','p7']
        self._fpn_strides = [8, 16, 32, 64, 128]
        self._focal_loss_alpha = 0.25
        self._focal_loss_gamma = 2.0

        # predict settings
        self.thresh_with_ctr = False
        self.pre_nms_thresh = 0.05
        self.pre_nms_topk = 1000
        self.post_nms_topk = 100
        self.nms_thresh = 0.7

        self._reg_loss_func = IOULoss(iou_type="giou")

        self.stem = nn.Sequential(
            nn.Conv2d(C_in, C, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(C // 8, C)
        )

        C_curr = C
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells, k = build_darts_cells(layers, C_prev_prev,C_prev, C_curr, multiplier=2, steps=4)

        # C_curr = C
        # self._depth = 4
        # self.cells, k = build_direct_cells(layers, C_curr, C_in, steps=self._depth)
        # self.cells, k = build_dense_cells(layers, C_curr, C_in, steps=self._depth)

        # Headers
        self.cls_logits = nn.Conv2d(
            multiplier * C_curr, self._num_classes,
            kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            multiplier * C_curr, 4, kernel_size=3,
            stride=1, padding=1
        )
        self.ctrness = nn.Conv2d(
            multiplier * C_curr, 1, kernel_size=3,
            stride=1, padding=1
        )
        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(len(self._in_features))])

        num_ops = len(PRIMITIVES)

        self.alphas_normal = torch.autograd.Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
        self.betas_normal = torch.autograd.Variable(1e-3 * torch.randn(k).cuda(), requires_grad=True)

        # init loss
        for modules in [self.cls_logits,self.bbox_pred, self.ctrness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

    def _process_weights2(self):
        weights2 = F.softmax(self.betas_normal[:2], dim=-1)
        n = 3
        start = 2

        for i in range(self._steps - 1):
            end = start + n
            tw2 = F.softmax(self.betas_normal[start:end], dim=-1)
            start = end
            n += 1
            weights2 = torch.cat([weights2, tw2], dim=0)
        return weights2


    def forward(self, features):
        feats = [features[f] for f in self._in_features]

        cls_preds = []
        box_preds = []
        ctr_preds = []

        for i, feat in enumerate(feats):
            s0 = s1 = self.stem(feat)
            # s1 = self.stem(feat)
            for j, cell in enumerate(self.cells):
                weights = F.softmax(self.alphas_normal, dim=-1)
                weights2 = self._process_weights2()
                s0, s1 = s1, cell(s0, s1, weights, weights2)
                # s1 = cell(s1, weights)

            cls_preds.append(self.cls_logits(s1))
            ctr_preds.append(self.ctrness(s1))
            box_pred = F.relu(self.scales[i](self.bbox_pred(s1)))
            box_preds.append(box_pred)
        return cls_preds, box_preds, ctr_preds

    def arch_parameters(self):
        arch_params = [
            self.alphas_normal,
            self.betas_normal
        ]
        return arch_params

    def genotype(self):
        # Darts
        # geno_normal = parse_darts(F.softmax(self.alphas_normal, dim=-1).detach().cpu().numpy(), self._steps)

        # Direct Cell
        geno_normal = parse_direct(F.softmax(self.alphas_normal, dim=-1).detach().cpu())

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(normal=geno_normal, normal_concat=concat)
        return genotype

    def fcos_loss(self, preds, targets):
        """
        Args:
            preds: cls_preds, box_preds, ctr_preds
            targets:
                labels, after flatten
                reg_targets, after flatten
                ctr_targets after flatten
        Returns: loss
        """
        logits_pred, reg_pred, ctr_pred = preds
        labels, reg_targets, ctr_targets = targets

        # reshape the preds
        logits_pred = cat([
            # Reshape: (N, C, Hi, Wi) -> (N, Hi, Wi, C) -> (N*Hi*Wi, C)
            x.permute(0, 2, 3, 1).reshape(-1, self._num_classes) for x in logits_pred
        ], dim=0, )

        reg_pred = cat([
            # Reshape: (N, B, Hi, Wi) -> (N, Hi, Wi, B) -> (N*Hi*Wi, B)
            x.permute(0, 2, 3, 1).reshape(-1, 4) for x in reg_pred
        ], dim=0, )

        ctr_pred = cat([
            # Reshape: (N, 1, Hi, Wi) -> (N*Hi*Wi,)
            x.permute(0, 2, 3, 1).reshape(-1) for x in ctr_pred
        ], dim=0, )

        pos_inds = torch.nonzero(labels != self._num_classes).squeeze(1)
        num_pos_local = max(pos_inds.numel(), 1.0)


        class_target = torch.zeros_like(logits_pred)
        class_target[pos_inds, labels[pos_inds]] = 1

        class_loss = sigmoid_focal_loss_jit(
            logits_pred,
            class_target,
            alpha=self._focal_loss_alpha,
            gamma=self._focal_loss_gamma,
            reduction="sum"
        )
        class_loss = class_loss / num_pos_local

        reg_targets = reg_targets[pos_inds]
        ctr_targets = ctr_targets[pos_inds]
        reg_pred = reg_pred[pos_inds]
        ctr_pred = ctr_pred[pos_inds]

        ctr_targets_sum = ctr_targets.sum()
        # loss_denorm = max(reduce_sum(ctr_targets_sum).item() / num_gpus, 1e-6)
        loss_denorm = max(ctr_targets_sum, 1e-6)

        if pos_inds.numel() > 0:
            reg_loss = self._reg_loss_func(
                reg_pred,
                reg_targets,
                ctr_targets
            ) / loss_denorm

            ctr_loss = F.binary_cross_entropy_with_logits(
                ctr_pred,
                ctr_targets,
                reduction="sum",
            ) / num_pos_local
        else:
            reg_loss = reg_pred.sum() * 0.0
            ctr_loss = ctr_pred.sum() * 0.0

        losses = {
            'loss_fcos_cls': class_loss,
            'loss_fcos_loc': reg_loss,
            'loss_fcos_ctr': ctr_loss
        }
        return losses

    def model_forward(self, data, flag=False):
        features = data['features']
        targets = data['targets']
        in_features = {
            'p3': features['p3'][0].cuda(),
            'p4': features['p4'][0].cuda(),
            'p5': features['p5'][0].cuda(),
            'p6': features['p6'][0].cuda(),
            'p7': features['p7'][0].cuda(),
        }
        preds = self(in_features)
        targets = [targets['labels'][0].cuda(), targets['reg_targets'][0].cuda(), targets['ctr_targets'][0].cuda()]
        losses = self.fcos_loss(preds, targets)
        if flag:
            return losses
        return sum(losses.values())

    def new(self):
        model_new = SearchHead(self._C_in, self._C, self._num_classes, self._layers, self._criterion, steps=self._steps, multiplier=self._multiplier).cuda()
        for x,y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def predict_proposals(self, preds):
        cls_preds, box_preds, ctr_preds = preds
        locations = self.compute_location(cls_preds)

        bundle = {
            "l": locations, "o": cls_preds,
            "r": box_preds, "c": ctr_preds,
            "s": self._fpn_strides,
        }

        sampled_boxes = []
        h, w = cls_preds[0].size()[2:]
        h, w = h * 8, w * 8
        image_sizes = [(h, w),]

        for i, per_bundle in enumerate(zip(*bundle.values())):
            # get per-level bundle
            per_bundle = dict(zip(bundle.keys(), per_bundle))
            # recall that during training, we normalize regression targets with FPN's stride.
            # we denormalize them here.
            l = per_bundle["l"]
            o = per_bundle["o"]
            r = per_bundle["r"] * per_bundle["s"]
            c = per_bundle["c"]
            t = per_bundle["t"] if "t" in bundle else None

            sampled_boxes.append(
                self.forward_for_single_feature_map(
                    l, o, r, c, image_sizes, t
                )
            )

            for per_im_sampled_boxes in sampled_boxes[-1]:
                per_im_sampled_boxes.fpn_levels = l.new_ones(
                    len(per_im_sampled_boxes), dtype=torch.long
                ) * i

        boxlists = list(zip(*sampled_boxes))
        boxlists = [Instances.cat(boxlist) for boxlist in boxlists]
        boxlists = self.select_over_all_levels(boxlists)

        return boxlists

    def forward_for_single_feature_map(
            self, locations, logits_pred, reg_pred,
            ctrness_pred, image_sizes, top_feat=None
    ):
        N, C, H, W = logits_pred.shape

        # put in the same format as locations
        logits_pred = logits_pred.view(N, C, H, W).permute(0, 2, 3, 1)
        logits_pred = logits_pred.reshape(N, -1, C).sigmoid()
        box_regression = reg_pred.view(N, 4, H, W).permute(0, 2, 3, 1)
        box_regression = box_regression.reshape(N, -1, 4)
        ctrness_pred = ctrness_pred.view(N, 1, H, W).permute(0, 2, 3, 1)
        ctrness_pred = ctrness_pred.reshape(N, -1).sigmoid()
        if top_feat is not None:
            top_feat = top_feat.view(N, -1, H, W).permute(0, 2, 3, 1)
            top_feat = top_feat.reshape(N, H * W, -1)

        # if self.thresh_with_ctr is True, we multiply the classification
        # scores with centerness scores before applying the threshold.
        if self.thresh_with_ctr:
            logits_pred = logits_pred * ctrness_pred[:, :, None]
        candidate_inds = logits_pred > self.pre_nms_thresh
        pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_topk)

        if not self.thresh_with_ctr:
            logits_pred = logits_pred * ctrness_pred[:, :, None]

        results = []
        for i in range(N):
            per_box_cls = logits_pred[i]
            per_candidate_inds = candidate_inds[i]
            per_box_cls = per_box_cls[per_candidate_inds]

            per_candidate_nonzeros = per_candidate_inds.nonzero()
            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1]

            per_box_regression = box_regression[i]
            per_box_regression = per_box_regression[per_box_loc]
            per_locations = locations[per_box_loc]
            if top_feat is not None:
                per_top_feat = top_feat[i]
                per_top_feat = per_top_feat[per_box_loc]

            per_pre_nms_top_n = pre_nms_top_n[i]

            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = \
                    per_box_cls.topk(per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]
                per_locations = per_locations[top_k_indices]
                if top_feat is not None:
                    per_top_feat = per_top_feat[top_k_indices]

            detections = torch.stack([
                per_locations[:, 0] - per_box_regression[:, 0],
                per_locations[:, 1] - per_box_regression[:, 1],
                per_locations[:, 0] + per_box_regression[:, 2],
                per_locations[:, 1] + per_box_regression[:, 3],
            ], dim=1)

            boxlist = Instances(image_sizes[i])
            boxlist.pred_boxes = Boxes(detections)
            boxlist.scores = torch.sqrt(per_box_cls)
            boxlist.pred_classes = per_class
            boxlist.locations = per_locations
            if top_feat is not None:
                boxlist.top_feat = per_top_feat
            results.append(boxlist)

        return results

    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            # multiclass nms
            result = boxlists[i]
            result = ml_nms(result, self.nms_thresh)
            number_of_detections = len(result)

            # Limit to max_per_image detections **over all classes**
            if number_of_detections > self.post_nms_topk > 0:
                cls_scores = result.scores
                image_thresh, _ = torch.kthvalue(
                    cls_scores.cpu(),
                    number_of_detections - self.post_nms_topk + 1
                )
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            results.append(result)
        return results

    def compute_location(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = compute_locations(
                h, w, self._fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    def visualization(self, file_name, output_dir):
        # get instances
        import os
        data_name = os.path.basename(file_name)[:-4] + '.pth'
        data_dir = '/home/haida_sunxin/lqx/data/search'

        data_file = os.path.join(data_dir, data_name)
        data = torch.load(data_file)
        features = {
            'p3': data['p3'].cuda(),
            'p4': data['p4'].cuda(),
            'p5': data['p5'].cuda(),
            'p6': data['p6'].cuda(),
            'p7': data['p7'].cuda(),
        }
        with torch.no_grad():
            preds = self(features)
            instances = self.predict_proposals(preds)

        # gt
        from datasets.metacatalog.catalog import DatasetCatalog, MetadataCatalog
        import cv2
        from utils.visualizer.det_visualizer import Visualizer
        import numpy as np

        dicts = DatasetCatalog.get('coco_2017_val')
        dict = [dicts[i] for i in range(len(dicts)) if os.path.basename(dicts[i]['file_name']) == os.path.basename(file_name)[:-4] + '.jpg']

        h, w = dict[0]['height'], dict[0]['width']
        predictions = detector_postprocess(instances[0], h, w)
        predictions = predictions.to(torch.device('cpu'))

        metadata = MetadataCatalog.get('coco_2017_val')

        img = cv2.imread(dict[0]["file_name"], cv2.IMREAD_COLOR)[:, :, ::-1]
        vis = Visualizer(img, metadata)
        vis_pred = vis.draw_instance_predictions(predictions).get_image()

        vis = Visualizer(img, metadata)
        vis_gt = vis.draw_dataset_dict(dict[0]).get_image()

        concat = np.concatenate((vis_pred, vis_gt), axis=1)
        cv2.imwrite(os.path.join(output_dir, data_name[:-4] + '.jpg'), concat[:, :, ::-1])


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'

    m = SearchHead(C_in=256, C=128, num_classes=80, layers=2, criterion=None, multiplier=2).cuda()
    # m = torch.load('/home/haida_sunxin/lqx/code/llseg/models/detection/anchorfree_heads/search_head/out/Epoch_49.pth', map_location='cpu')
    # depth = 4
    # m = DenseCell(C=128, C_out=256, depth=depth)
    # m = m.cuda()
    print(m)
    print(m.alphas_normal.size())

    # k = sum(i + 1 for i in range(depth))
    #
    x = torch.randn(1, 128, 64, 64).cuda()
    # weights = torch.randn(k, 7).cuda()
    # weights = torch.nn.functional.softmax(weights, dim=-1)
    # y = m(x, weights)
    # print(y.size())
    # m.load_state_dict(state_dict)
    # name = '/home/haida_sunxin/lqx/data/search/000000475678.pth'
    # out_dir = '/home/haida_sunxin/lqx'
    # m.visualization(name, out_dir)



