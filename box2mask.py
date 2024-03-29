from typing import List
import torch
from detectron2.modeling import build_model
from detectron2.data import (
    MetadataCatalog,
)
from detectron2.modeling import GeneralizedRCNN
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T
from detectron2.structures import Instances, Boxes


class Box2Mask:
    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image, xyxy_boxes, is_object=None):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            _, H, W = image.size()
            print(image.size(), original_image.shape)

            if isinstance(xyxy_boxes, List):
                xyxy_boxes = Boxes(xyxy_boxes)
            xyxy_boxes = pad_box(xyxy_boxes, 0.1)
            
            if is_object is None:
                is_object = torch.LongTensor([1, ] * len(xyxy_boxes))
            inputs = {"image": image, "height": height, "width": width}

            batched_inputs = [inputs]
            model = self.model
            images = model.preprocess_image(batched_inputs)
            features = model.backbone(images.tensor)
            device = images.tensor.device
            # use GT boxes as proposal
            proposals = Instances(
                (height, width),
                proposal_boxes=xyxy_boxes.to(device),
                objectness_logits=torch.ones([len(xyxy_boxes)]).long().to(device),
            )
            # rescale it to processed size
            proposals = detector_postprocess(proposals, H, W)
            # box_head:forward_box, 
            # this is just to obtain class score, the predicted boxes for each class will be discarded 
            box_features = [features[f] for f in model.roi_heads.box_in_features]
            box_features = model.roi_heads.box_pooler(box_features, [x.proposal_boxes for x in [proposals]]) # 256, 7, 7
            box_features = model.roi_heads.box_head(box_features)  # (1024)
            predictions = model.roi_heads.box_predictor(box_features)  # (3, 81), (3, 320=80*4)

            # pred_classes: if is object class number, else person cat (0)
            pred_classes = torch.argmax(predictions[0][:, :-1], 1) * is_object.to(device) # (3, )
            proposals.pred_boxes = proposals.proposal_boxes
            proposals.pred_classes = pred_classes

            # pred_instances: top(proposal * 80): pred_boxes, pred_classes, scores
            instances = model.roi_heads.forward_with_given_boxes(features, [proposals])
            predictions = GeneralizedRCNN._postprocess(instances, batched_inputs, images.image_sizes)[0]

            return predictions

def pad_box(boxes: Boxes, ratio):
    box = boxes.tensor
    widths = box[:, 2] - box[:, 0]
    heights = box[:, 3] - box[:, 1]
    center = boxes.get_centers()

    widths *= (1 + 2 * ratio)
    heights *= (1 + 2 * ratio)

    boxes = torch.stack([
        center[:, 0] - widths / 2, 
        center[:, 1] - heights / 2, 
        center[:, 0] + widths / 2, 
        center[:, 1] + heights / 2
        ], 1    )
    boxes = Boxes(boxes)
    return boxes