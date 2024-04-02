"""
    mid.metric 
    Copyright (c) 2022-present NAVER Corp.
    Apache-2.0
"""

from PIL import Image
import clip
import os
import random

from typing import *

import torch
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_info

import torch.nn as nn
import numpy as np
import pickle


def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names


def load_file(path):
    with open(path, "rb") as f:
        _file = pickle.load(f)
    return _file


def get_label(path):
    idx = path.find("label_")
    try:
        label = int(path[idx+6:idx+8])
    except:
        label = int(path[idx + 6:idx + 7])
    return label



def log_det(X):
    eigenvalues = X.svd()[1]
    return eigenvalues.log().sum()


def robust_inv(x, eps=0):
    Id = torch.eye(x.shape[0]).to(x.device)
    return (x + eps * Id).inverse()


def exp_smd(a, b, reduction=True):
    a_inv = robust_inv(a)
    if reduction:
        assert b.shape[0] == b.shape[1]
        return (a_inv @ b).trace()
    else:
        return (b @ a_inv @ b.t()).diag()


def _compute_pmi(x: Tensor, y: Tensor, x0: Tensor, limit: int = 30000,
                 reduction: bool = True, full: bool = False) -> Tensor:
    r"""
    A numerical stable version of the MID score.

    Args:
        x (Tensor): features for real samples
        y (Tensor): features for text samples
        x0 (Tensor): features for fake samples
        limit (int): limit the number of samples
        reduction (bool): returns the expectation of PMI if true else sample-wise results
        full (bool): use full samples from real images

    Returns:
        Scalar value of the mutual information divergence between the sets.
    """
    N = x.shape[0]
    excess = N - limit
    if 0 < excess:
        if not full:
            x = x[:-excess]
            y = y[:-excess]
        x0 = x0[:-excess]
    N = x.shape[0]
    M = x0.shape[0]
    print(f"real: {N}, fake: {M}",
          f"x.shape_1: {x.shape[1]}, ")

    assert N >= x.shape[1], "not full rank for matrix inversion!"
    if x.shape[0] < 30000:
        rank_zero_info("if it underperforms, please consider to use "
                       "the epsilon of 5e-4 or something else.")

    z = torch.cat([x, y], dim=-1)
    z0 = torch.cat([x0, y[:x0.shape[0]]], dim=-1)
    x_mean = x.mean(dim=0, keepdim=True)
    y_mean = y.mean(dim=0, keepdim=True)
    z_mean = torch.cat([x_mean, y_mean], dim=-1)
    x0_mean = x0.mean(dim=0, keepdim=True)
    z0_mean = z0.mean(dim=0, keepdim=True)

    X = (x - x_mean).t() @ (x - x_mean) / (N - 1)
    Y = (y - y_mean).t() @ (y - y_mean) / (N - 1)
    Z = (z - z_mean).t() @ (z - z_mean) / (N - 1)
    X0 = (x0 - x_mean).t() @ (x0 - x_mean) / (M - 1)  # use the reference mean
    Z0 = (z0 - z_mean).t() @ (z0 - z_mean) / (M - 1)  # use the reference mean

    alternative_comp = False
    # notice that it may have numerical unstability. we don't use this.
    if alternative_comp:
        def factorized_cov(x, m):
            N = x.shape[0]
            return (x.t() @ x - N * m.t() @ m) / (N - 1)
        X0 = factorized_cov(x0, x_mean)
        Z0 = factorized_cov(z0, z_mean)

    # assert double precision
    for _ in [X, Y, Z, X0, Z0]:
        assert _.dtype == torch.float64

    # Expectation of PMI
    mi = (log_det(X) + log_det(Y) - log_det(Z)) / 2
    rank_zero_info(f"MI of real images: {mi:.4f}")

    # Squared Mahalanobis Distance terms
    if reduction:
        smd = (exp_smd(X, X0) + exp_smd(Y, Y) - exp_smd(Z, Z0)) / 2
    else:
        smd = (exp_smd(X, x0 - x_mean, False) + exp_smd(Y, y - y_mean, False)
               - exp_smd(Z, z0 - z_mean, False)) / 2
        mi = mi.unsqueeze(0)  # for broadcasting

    return mi + smd

def load_images_from_folder(folder: str, transform: T.Compose) -> List[torch.Tensor]:
    images = []
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        if os.path.isfile(path):
            image = Image.open(path).convert("RGB")
            images.append(transform(image).unsqueeze(0))
    return torch.cat(images, 0)

def get_clip_model_and_preprocessor(eval_model: str = "ViT-B/32", device: Union[torch.device, int] = torch.device("cuda")) -> Tuple[torch.nn.Module, T.Compose]:
    clip_model, _ = clip.load(eval_model, device=device, download_root = '/data2/cache/CLIP')
    clip_prep = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    return clip_model, clip_prep

def process_images_and_text(generated_imgs_folder: Union[str, torch.Tensor], prompts: List[str], clip_model: torch.nn.Module, device: torch.device, real_imgs_folder: Union[str, torch.Tensor] = None) -> None:
    
    
    if clip_model is None:
        clip_model, clip_prep = get_clip_model_and_preprocessor("ViT-B/32", device)
    else:
        _, clip_prep = get_clip_model_and_preprocessor("ViT-B/32", device)
        
    if real_imgs_folder is not None:
        if isinstance(real_imgs_folder, torch.Tensor):
            real_images = real_imgs_folder
        else:
            real_images = load_images_from_folder(real_imgs_folder, clip_prep)
            with torch.no_grad():
                real_image_features = clip_model.encode_image(real_images.to(device))
    else:
        real_image_features = None
   
    if isinstance(generated_imgs_folder, torch.Tensor):
        generated_images = generated_imgs_folder
    else: 
        generated_images = load_images_from_folder(generated_imgs_folder, clip_prep)
    
    if isinstance(prompts, list):
        text_tokens = clip.tokenize(prompts).to(device)
    else:
        text_tokens = prompts
    with torch.no_grad():
        generated_image_features = clip_model.encode_image(generated_images.to(device))
        text_features = clip_model.encode_text(text_tokens)
    
    
    return real_image_features, generated_image_features, text_features
def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Object confidence times class confidence
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
        # Sort by it
        image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
        # Perform non-maximum suppression
        keep_boxes = []
        while detections.size(0):
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            label_match = detections[0, -1] == detections[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            weights = detections[invalid, 4:5]
            # Merge overlapping bboxes by order of confidence
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)

    return output

def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA=True):
    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    anchors = [(a[0] / stride, a[1] / stride) for a in anchors]

    prediction = prediction.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)

    # Sigmoid the  centre_X, centre_Y. and object confidencce
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

    # Add the center offsets
    grid_len = np.arange(grid_size)
    a, b = np.meshgrid(grid_len, grid_len)

    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)

    prediction[:, :, :2] += x_y_offset

    # log space transform height and the width
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors

    # Softmax the class scores
    prediction[:, :, 5: 5 + num_classes] = torch.sigmoid((prediction[:, :, 5: 5 + num_classes]))

    prediction[:, :, :4] *= stride

    return prediction


def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def bbox_wh_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou
