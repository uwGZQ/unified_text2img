import os
os.environ['HF_HOME'] = '/data2/cache'
os.environ['TORCH_HOME'] = '/data2/cache'
import torch
from utils import *

# N x 3 x H x W
class InceptionScore:
    def __init__(self,features=64,normalize = True):
        from torchmetrics.image.inception import InceptionScore
        self.inception = InceptionScore(feature=features,normalize=normalize)
        
    def update(self, imgs):
        self.inception.update(imgs)
        
    def compute(self):
        return self.inception.compute()

# # N x 3 x H x W
class FrechetInceptionDistance:
    def __init__(self, features=64, normalize=True):
        from torchmetrics.image.fid import FrechetInceptionDistance
        self.fid = FrechetInceptionDistance(feature=features, normalize=normalize)
        
    def update(self, real_imgs, fake_imgs):
        self.fid.update(real_imgs, real=True)
        self.fid.update(fake_imgs, real=False)
        
    def compute(self):
        return self.fid.compute()

# N x 3 x H x W
class LPIPS:
    def __init__(self, net_type = 'alex',normalize = True):
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type=net_type, normalize=normalize)
    def update(self, real_imgs, fake_imgs):
        self.lpips.update(real_imgs, fake_imgs)
    def compute(self):
        return self.lpips.compute()
    
# N x 3 x H x W
class KernelInceptionDistance:
    def __init__(self, kernel_type = 'linear'):
        from torchmetrics.image.inception import KernelInceptionDistance
        self.kid = KernelInceptionDistance(subset_size=50)
    def update(self, real_imgs, fake_imgs):
        self.kid.update(real_imgs, real = True)
        self.kid.update(fake_imgs, real = False)
    def compute(self):
        return self.kid.compute()
# img: N x 3 x H x W, prompt: str
class ClipScore():
    def __init__(self, model_name_or_path="openai/clip-vit-base-patch16"):
        from torchmetrics.functional.multimodal import clip_score
        self.clip_score_fn = clip_score(model_name_or_path="openai/clip-vit-base-patch16")
    def update(self, prompts, images):
        self.clip_score_fn.update(text = prompts, images = images)
    def compute(self):
        return self.clip_score_fn.compute()

# Path list / Image.Image list; str
class HPSv2:
    def __init__(self):
        pass
    def compute(self,imgs_path, prompt = '<prompt>', hps_version="v2.1"):
        import hpsv2
        return hpsv2.score(imgs_path, prompt, hps_version) 
"""
pil_images = [Image.open("my_amazing_images/1.jpg"), Image.open("my_amazing_images/2.jpg")]
prompt = "fantastic, increadible prompt"
print(model.compute(prompt, pil_images))
"""
class PickScore_v1:
    def __init__(self,device = "cuda",proceesor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"):
        from transformers import AutoProcessor, AutoModel
        self.device = device
        self.processor = AutoProcessor.from_pretrained(processor_name_or_path)
        self.model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(device)
    def compute(self, prompt, images):
        image_inputs = self.processor(
            images=images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)
    
        text_inputs = self.processor(
            text=prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        image_embs = self.model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
    
        text_embs = self.model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
    
        # score
        scores = self.model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
        
        # get probabilities if you have multiple images to choose from
        probs = torch.softmax(scores, dim=-1)
        return probs.cpu().tolist()

"""from PIL import Image
import numpy as np
model = ImageReward()
pil_images = [Image.open("/data1/ziqi/Metrics/frames/frame_0.jpg"), Image.open("/data1/ziqi/Metrics/frames/frame_1.jpg")]
prompt = "fantastic, increadible prompt"
print(model.compute_score(prompt, pil_images))
print(model.compute_rank(prompt, pil_images))"""
class ImageReward:
    def __init__(self, model_name = "ImageReward-v1.0"):
        import ImageReward as RM
        self.model = RM.load(model_name)
    def compute_rank(self, prompt, images):
        ranking, reward = self.model.inference_rank(prompt, images)
        return ranking, reward
    def compute_score(self, prompt, images):
        return self.model.score(prompt, images)
    def compile(self,prompt, images):
        return self.compute_score(prompt, images), self.compute_rank(prompt, images)






class RPrecision(Metric):
    r"""
    Calculates R-Precision which is used to assess the alignment between the 
    conditional texts and the generated images.

    Args:
        feature (int): the number of features
        limit (int): limit the number of samples to calculate
        kwargs:
            Additional keyword arguments, see :ref:`Metric kwargs` for more info.
    """
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(self, feature: int = 512, limit: int = 30000,
                 **kwargs: Dict[str, Any]) -> None:
        super().__init__(**kwargs)

        self.limit = limit
        self._debug = True
        self._dtype = torch.float64

        for k in ['x', 'y', 'x0']:  # x: real, y: text, x0: fake
            self.add_state(f"{k}_feat", [], dist_reduce_fx=None)

    def update(self, x: Tensor, y: Tensor, x0: Tensor) -> None:
        r"""
        Update the state with extracted features in double precision. This 
        method changes the precision of features into double-precision before 
        saving the features.

        Args:
            x (Tensor): tensor with the extracted real image features
            y (Tensor): tensor with the extracted text features
            x0 (Tensor): tensor with the extracted fake image features
        """
        assert x.shape[0] == y.shape[0] and x.shape[1] == y.shape[1]

        self.orig_dtype = x.dtype
        x, y, x0 = [x.double() for x in [x, y, x0]]
        self.x_feat.append(x)
        self.y_feat.append(y)
        self.x0_feat.append(x0)

    def _modify(self, mode: Any = None):
        r"""
        Modify the distribution of generated images for ablation study.

        Arg:
            mode (str): if `mode` is "real", it measure the real's score, if
                `mode` is "shuffle", deliberately break the alignmnet with 
                the condition by randomly-shuffling their counterparts.
        """
        if "real" == mode:
            self.x0_feat = self.x_feat
        elif "shuffle" == mode:
            random.shuffle(self.x0_feat)
        return self

    def compute(self, reduction: bool = True, mode=None) -> Tensor:
        r"""
        Calculate the R-Precision score based on accumulated extracted features.
        """
        feats = [torch.cat(getattr(self, f"{k}_feat"), dim=0)
                 for k in ['x', 'y', 'x0']]

        return self._compute(*feats, reduction).to(self.orig_dtype)

    def _compute(self, X: Tensor, Y: Tensor, Z: Tensor, reduction):
        def dot(x, y):
            return (x * y).sum(dim=-1)

        excess = X.shape[0] - self.limit
        if 0 < excess:
            X, Y, Z = [x[:-excess] for x in [X, Y, Z]]

        scores = []
        scores.append(dot(Z, Y))
        for i in range(99):  # negative scores
            Y_ = Y[torch.randperm(Y.shape[0])]
            scores.append(dot(Z, Y_))
        scores = torch.stack(scores, dim=-1)  # N x 100
        _, idx = scores.max(dim=-1)

        if reduction:
            return (idx == 0).float().mean()
        else:
            return (idx == 0).float()




# from .darknet import *


class SemanticObjectAccuracy(Metric):
    r"""
    Calculates the Semantic Object Accuracy which is used to assess the 
    alignment between the conditional texts and the generated images. This
    metric is a little different from SOA-I and SOA-C since this is a piece-wise
    evaluating metric of SOA-I.

    Args:
        root (str): Path to darknet for the YOLO-V3
        img_size (int): Image size
        confidence (float): confidence for the YOLO-V3
        nms_thresh (float): NMS threshold for the YOLO-V3
        limit (int): Limit the number of samples to calculate
        kwargs:
            Additional keyword arguments, see :ref:`Metric kwargs` for more info.
    """
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(self, root: str = "darknet",
                 img_size: int = 256, confidence: float = .5,
                 nms_thresh: float = .4, limit: int = 30000,
                 **kwargs: Dict[str, Any]) -> None:
        super().__init__(**kwargs)
        self._debug = True
        self.root = root
        self.img_size = img_size
        self.confidence = confidence
        self.nms_thresh = nms_thresh
        self.limit = limit
        self.setup()
        self.classes = load_classes(os.path.join(self.root, "data/coco.names"))

        self.add_state(f"reals", [], dist_reduce_fx=None)
        self.add_state(f"predictions", [], dist_reduce_fx=None)
        self.add_state(f"labels", [], dist_reduce_fx=None)

    def setup(self):
        # Set up the neural network
        print("Loading network ...")
        try:
            self.model = Darknet(os.path.join(self.root, "cfg/yolov3.cfg"))
            self.model.load_weights(os.path.join(self.root, "yolov3.weights"))
        except:
            print("Did you install darknet for YOLO-V3?")
            print("$ git clone https://github.com/pjreddie/darknet.git")
            print("$ cd darknet")
            print("$ make")
            print("$ wget https://pjreddie.com/media/files/yolov3.weights")
        print("Network successfully loaded")

        self.model.net_info["height"] = 256
        inp_dim = int(self.model.net_info["height"])
        assert inp_dim % 32 == 0
        assert inp_dim > 32

        # Set the model in evaluation mode
        self.model.eval()

    def get_labels(self, caption):
        # The rules from Table 4 (Hinz et al., 2020)
        # Todo: add more rules
        labels = []
        tokens = caption.lower().split(' ')

        WORDS = {
            "person": ["person", "people", "human", "man", "men", "woman",
                       "women", "child", "children"],
            "diningtable": ["dining table", "table", "desk"],
            "cat": ["cat", "kitten"],
            "dog": ["dog", "pup"],
            "boat": ["boat", "ship"],
            "car": ["car", "auto"],
            "sports ball": ["ball"],
            "bicycle": ["bicycle", "bike"],
            "monitor": ["monitor", "tv", "screen"],
            "hot dog": ["hot dog", "chili dog", "cheese dog", "corn dog"],
            "fire hydrant": ["fire hydrant", "hydrant"],
            "sofa": ["sofa", "couch"],
            "aeroplane": ["plane", "jet", "aircraft"],
            "cell phone": ["cell phone", "mobile phone"],
            "refrigerator": ["refrigerator", "fridge"],
            "motocycle": ["motocycle", "dirt bike", "motobike", "scooter"],
            "backpack": ["backpack", "rucksack"],
            "handbag": ["handbag", "purse"],
            "mouse": ["computer mouse"],
            "scissor": ["scissors"],
            "orange": ["oranges"]
        }

        def remove_words(caption, excludes):
            for w in excludes:
                caption = caption.replace(w, "")
            return caption

        for c in self.classes:
            if c not in WORDS.keys():
                WORDS[c] = [c]

        # multiple words
        for label, words in WORDS.items():
            if "dog" == label:
                caption_ = remove_words(caption, [
                    "hot dog", "cheese dog", "chili dog", "corn dog"])
            elif "elephant" == label:
                caption_ = remove_words(caption, [
                    "toy elephant", "stuffed elephant"])
            elif "car" == label:
                caption_ = remove_words(caption, [
                    "train car", "car window", "side car", "passenger car",
                    "subway car", "car tire", "rail car", "tram car",
                    "street car", "trolly car"])
            elif "kite" == label:
                caption_ = remove_words(caption, ["kite board", "kiteboard"])
            elif "cake" == label:
                caption_ = remove_words(caption, ["cupcake"])
            elif "bicycle" == label:
                caption_ = remove_words(caption, [
                    "train car", "car window", "side car", "passenger car",
                    "subway car", "car tire", "rail car", "tram car",
                    "street car", "trolly car"])
            elif "tie" == label:
                caption_ = remove_words(caption, ["to tie"])
            elif "apple" == label:
                caption_ = remove_words(caption, ["pineapple"])
            elif "oven" == label:
                caption_ = remove_words(caption, ["microwave oven"])
            else:
                caption_ = caption

            for w in words:
                if 1 == len(w.split(' ')):
                    if w in caption_.lower().split(' '):
                        labels.append(label)
                else:
                    if w in caption_.lower():
                        labels.append(label)
        return labels

    def update(self, images: Tensor, captions: List[str],
               is_real: bool = False) -> None:
        r"""
        Update the state with images and captions.

        Args:
            images (Tensor): tensor with the extracted fake images
            captions (List[str]): List of captions
            is_real (bool): Is the real image?
        """
        with torch.no_grad():
            predictions = self.model(images)
            predictions = non_max_suppression(
                predictions, self.confidence, self.nms_thresh)

            for preds in predictions:
                img_preds_id = set()
                img_preds_name = set()  # handling multiple object
                img_bboxs = []
                if preds is not None and len(preds) > 0:
                    for pred in preds:
                        pred_id = int(pred[-1])
                        pred_name = self.classes[pred_id]

                        bbox_x = pred[0] / self.img_size
                        bbox_y = pred[1] / self.img_size
                        bbox_width = (pred[2] - pred[0]) / self.img_size
                        bbox_height = (pred[3] - pred[1]) / self.img_size

                        img_preds_id.add(pred_id)
                        img_preds_name.add(pred_name)
                        img_bboxs.append([bbox_x.cpu().numpy(),
                                          bbox_y.cpu().numpy(),
                                          bbox_width.cpu().numpy(),
                                          bbox_height.cpu().numpy()])
                if not is_real:
                    self.predictions.append(list(img_preds_name))
                else:
                    self.reals.append(img_preds_name)

        if not is_real:
            for caption in captions:
                self.labels.append(self.get_labels(caption))

            assert len(self.predictions) == len(self.labels)

    def _modify(self, mode: Any = None):
        r"""
        Modify the distribution of generated images for ablation study.

        Arg:
            mode (str): if `mode` is "real", it measure the real's score, if
                `mode` is "shuffle", deliberately break the alignmnet with 
                the condition by randomly-shuffling their counterparts.
        """
        if "real" == mode:
            self.predictions = self.reals
        elif "shuffle" == mode:
            random.shuffle(self.predictions)
        return self

    def compute(self, reduction: bool = True) -> Tensor:
        r"""
        Calculate the point-wise SOA score.
        """
        accuracy = []
        division_by_zero = 0
        for preds, labels in zip(self.predictions, self.labels):
            if 0 == len(labels):
                division_by_zero += 1
                accuracy.append(-1)
            else:
                accuracy.append(
                    sum([1. for x in set(preds) if x in labels]) / len(labels))
        accuracy = torch.Tensor(accuracy)
        if 0 < division_by_zero:
            print(f"warning: {division_by_zero} samples have no detection.")

        if reduction:
            return accuracy[:self.limit].mean()
        else:
            return accuracy[:self.limit]



class MutualInformationDivergence(Metric):
    r"""
    Calculates the Mutual Information Divergence (MID) which is used to assess 
    the text-image alignment between the conditional texts and the 
    generated images compared with the same texts and the real images as 
    follows:

    .. math::
        \mathbb{E}_{\hat{x}, \hat{y}} \text{PMI}(\hat{x}; \hat{y}) = I(\mathbf{X}; \mathbf{Y}) + 
            \frac{1}{2} \mathbb{E}_{\hat{x}, \hat{y}} \big[ D_M^2(\hat{x}) + D_M^2(\hat{y}) - D_M^2(\hat{z}) \big]
    where

    .. math::
        I(\mathbf{X}; \mathbf{Y}) = \frac{1}{2}\log\Big( \frac{\det(\Sigma_x) \det(\Sigma_y)}{\det(\Sigma_z)} \Big), 
        D_M^2(x) = (x - \mu_x)^\intercal \Sigma_x^{-1} (x - \mu_x).

    The two multivariate normal distributions are estimated from the 
    CLIP (Radford et al., 2021) features calculated on conditional texts and 
    generated images. The joint distribution :math:`\mathcal{N}(\mu, \Sigma)` 
    is from the concatenation of the two features.

    Args:
        feature (int): the number of features
        limit (int): limit the number of samples to calculate
        kwargs:
            Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> import torch
        >>> _ = torch.manual_seed(123)
        >>> from metrics.mid import MutualInformationDivergence
        >>> mid = MutualInformationDivergence(2)
        >>> dist1 = torch.rand(10, 2, dtype=torch.float64) * 2 - 1
        >>> dist2 = torch.rand(10, 2, dtype=torch.float64) * 2 - 1
        >>> dist3 = dist1 * 1.1
        >>> mid.update(dist1, dist2, dist3)
        >>> mid.compute()
        MI of real images: 0.1543
        tensor(0.1516, dtype=torch.float64)

    """
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(self, feature: int = 512, limit: int = 30000,
                 **kwargs: Dict[str, Any]) -> None:
        super().__init__(**kwargs)

        self.limit = limit
        self._debug = False
        self._dtype = torch.float64

        for k in ['x', 'y', 'x0']:  # x: real, y: text, x0: fake
            self.add_state(f"{k}_feat", [], dist_reduce_fx=None)

    def update(self, x: Tensor, y: Tensor, x0: Tensor) -> None:
        r"""
        Update the state with extracted features in double precision. It is 
        recommended to use the CLIP ViT-B/32 or ViT-L/14 features which is 
        L2-normalized. This method changes the precision of features into 
        double-precision before saving the features.

        Args:
            x (Tensor): tensor with the extracted real image features
            y (Tensor): tensor with the extracted text features
            x0 (Tensor): tensor with the extracted fake image features
        """
        assert x.shape[0] == y.shape[0] and x.shape[1] == y.shape[1]

        self.orig_dtype = x.dtype
        x, y, x0 = [x.double() for x in [x, y, x0]]
        self.x_feat.append(x)
        self.y_feat.append(y)
        self.x0_feat.append(x0)

    def _modify(self, mode: str = None):
        r"""
        Modify the distribution of generated images for ablation study.

        Arg:
            mode (str): if `mode` is "real", it measure the real's score, if
                `mode` is "shuffle", deliberately break the alignmnet with 
                the condition by randomly-shuffling their counterparts.
        """
        if "real" == mode:
            self.x0_feat = self.x_                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       