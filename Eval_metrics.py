import os
import torch
from utils import *
from utils import _compute_pmi
import clip
from typing import *
Eval_metrics = {
    "InceptionScore":               "InceptionScore",
    "FrechetInceptionDistance":     "FrechetInceptionDistance",
    "LPIPS":                        "LPIPS",
    "KernelInceptionDistance":      "KernelInceptionDistance",
    "ClipScore":                    "ClipScore",
    "HPSv2":                        "HPSv2",
    "PickScore_v1":                 "PickScore_v1",
    "ImageReward":                  "ImageReward",
    "RPrecision":                   "RPrecision",
    "SemanticObjectAccuracy":       "SemanticObjectAccuracy",
    "MutualInformationDivergence":  "MutualInformationDivergence",
    "Vbench":                       "Vbench",
}


def list_metrics():
    return list(Eval_metrics.keys())


class AbstractModel:
    def compute(self):
        "(Abstract method) abstract compute method"


class Compute_Metrics:
    def __init__(
        self,
        metric_name: str = "InceptionScore",
        metric: AbstractModel = None,
        torch_device: str = "cuda",
    ):
        if metric is None:
            print(f"Loading {metric_name}...")
            metric_name = Eval_metrics[metric_name]
            self.metric = eval(metric_name)(device=torch_device)
            print(f"Finished loading {metric_name}")
        else:
            print(f"Using the provided metric ...")
            metric_name = metric.__class__.__name__
            self.metric = eval(metric_name)(device=torch_device)

    @torch.no_grad()
    def update(self, **kwargs):
        return self.metric.update(**kwargs)

    @torch.no_grad()
    def compute(self, **kwargs):
        return self.metric.compute(**kwargs)


class InceptionScore(AbstractModel):
    def __init__(self, features=64, normalize=True, device="cuda"):
        from torchmetrics.image.inception import InceptionScore

        self.inception = InceptionScore(feature=features, normalize=normalize)

    def update(self, imgs: Union[str, List[Image.Image], Tensor]):
        if isinstance(imgs, str):
            from PIL import Image
            from torchvision.transforms import ToTensor
            import os

            # imgs is a folder name
            imgs = [os.path.join(imgs, img) for img in os.listdir(imgs)]
            imgs = [Image.open(img) for img in imgs]
            imgs = [ToTensor()(img).unsqueeze(0) for img in imgs]
            imgs = torch.cat(imgs, 0)
        elif isinstance(imgs, list):
            from torchvision.transforms import ToTensor

            imgs = [ToTensor()(img).unsqueeze(0) for img in imgs]
            imgs = torch.cat(imgs, 0)
        self.inception.update(imgs)

    def compute(self):
        return self.inception.compute()


class FrechetInceptionDistance(AbstractModel):
    def __init__(self, features=64, normalize=True, device="cuda"):
        from torchmetrics.image.fid import FrechetInceptionDistance

        self.fid = FrechetInceptionDistance(feature=features, normalize=normalize)

    def update(
        self,
        real_imgs: Union[str, List[Image.Image], Tensor],
        fake_imgs: Union[str, List[Image.Image], Tensor],
    ):
        if isinstance(real_imgs, str):
            from PIL import Image
            from torchvision.transforms import ToTensor

            real_imgs = [os.path.join(real_imgs, img) for img in os.listdir(real_imgs)]
            real_imgs = [Image.open(img) for img in real_imgs]
            real_imgs = [ToTensor()(img).unsqueeze(0) for img in real_imgs]
            real_imgs = torch.cat(real_imgs, 0)
        elif isinstance(real_imgs, list):
            from torchvision.transforms import ToTensor

            real_imgs = [ToTensor()(img).unsqueeze(0) for img in real_imgs]
            real_imgs = torch.cat(real_imgs, 0)

        if isinstance(fake_imgs, str):
            from PIL import Image
            from torchvision.transforms import ToTensor

            fake_imgs = [os.path.join(fake_imgs, img) for img in os.listdir(fake_imgs)]
            fake_imgs = [Image.open(img) for img in fake_imgs]
            fake_imgs = [ToTensor()(img).unsqueeze(0) for img in fake_imgs]
            fake_imgs = torch.cat(fake_imgs, 0)
        elif isinstance(fake_imgs, list):
            from torchvision.transforms import ToTensor

            fake_imgs = [ToTensor()(img).unsqueeze(0) for img in fake_imgs]
            fake_imgs = torch.cat(fake_imgs, 0)
        self.fid.update(real_imgs, real=True)
        self.fid.update(fake_imgs, real=False)

    def compute(self):
        return self.fid.compute()


class LPIPS(AbstractModel):
    def __init__(self, net_type="alex", normalize=True, device="cuda"):
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.lpips = LearnedPerceptualImagePatchSimilarity(
            net_type=net_type, normalize=normalize
        )

    def update(
        self,
        real_imgs: Union[str, List[Image.Image], Tensor],
        fake_imgs: Union[str, List[Image.Image], Tensor],
    ):
        import torchvision.transforms as T

        trans = T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
            ]
        )

        if isinstance(real_imgs, str):
            from PIL import Image

            real_imgs = [os.path.join(real_imgs, img) for img in os.listdir(real_imgs)]
            real_imgs = [Image.open(img) for img in real_imgs]
            real_imgs = [trans(img).unsqueeze(0) for img in real_imgs]
            real_imgs = torch.cat(real_imgs, 0)
        elif isinstance(real_imgs, list):
            from torchvision.transforms import ToTensor

            real_imgs = [trans(img).unsqueeze(0) for img in real_imgs]
            real_imgs = torch.cat(real_imgs, 0)

        if isinstance(fake_imgs, str):
            from PIL import Image
            from torchvision.transforms import ToTensor

            fake_imgs = [os.path.join(fake_imgs, img) for img in os.listdir(fake_imgs)]
            fake_imgs = [Image.open(img) for img in fake_imgs]
            fake_imgs = [trans(img).unsqueeze(0) for img in fake_imgs]
            fake_imgs = torch.cat(fake_imgs, 0)
        elif isinstance(fake_imgs, list):
            from torchvision.transforms import ToTensor

            fake_imgs = [trans(img).unsqueeze(0) for img in fake_imgs]
            fake_imgs = torch.cat(fake_imgs, 0)

        self.lpips.update(real_imgs, fake_imgs)

    def compute(self):
        return self.lpips.compute()


class KernelInceptionDistance(AbstractModel):
    def __init__(self, kernel_type="linear", device="cuda", normalize=True):
        from torchmetrics.image.kid import KernelInceptionDistance

        self.kid = KernelInceptionDistance(subset_size=5, normalize=normalize)

    def update(
        self,
        real_imgs: Union[str, List[Image.Image], Tensor],
        fake_imgs: Union[str, List[Image.Image], Tensor],
    ):
        if isinstance(real_imgs, str):
            from PIL import Image
            from torchvision.transforms import ToTensor

            real_imgs = [os.path.join(real_imgs, img) for img in os.listdir(real_imgs)]
            real_imgs = [Image.open(img) for img in real_imgs]
            real_imgs = [ToTensor()(img).unsqueeze(0) for img in real_imgs]
            real_imgs = torch.cat(real_imgs, 0)
        elif isinstance(real_imgs, list):
            from torchvision.transforms import ToTensor

            real_imgs = [ToTensor()(img).unsqueeze(0) for img in real_imgs]
            real_imgs = torch.cat(real_imgs, 0)

        if isinstance(fake_imgs, str):
            from PIL import Image
            from torchvision.transforms import ToTensor

            fake_imgs = [os.path.join(fake_imgs, img) for img in os.listdir(fake_imgs)]
            fake_imgs = [Image.open(img) for img in fake_imgs]
            fake_imgs = [ToTensor()(img).unsqueeze(0) for img in fake_imgs]
            fake_imgs = torch.cat(fake_imgs, 0)
        elif isinstance(fake_imgs, list):
            from torchvision.transforms import ToTensor

            fake_imgs = [ToTensor()(img).unsqueeze(0) for img in fake_imgs]
            fake_imgs = torch.cat(fake_imgs, 0)
        self.kid.update(real_imgs, real=True)
        self.kid.update(fake_imgs, real=False)

    def compute(self):
        return self.kid.compute()


class ClipScore(AbstractModel):
    def __init__(
        self, model_name_or_path="openai/clip-vit-base-patch16", device="cuda"
    ):
        from torchmetrics.multimodal.clip_score import CLIPScore

        self.clip_score_fn = CLIPScore(
            model_name_or_path=model_name_or_path
        )

    def update(self, prompts: List[str], images: Union[str, List[Image.Image]]):
        if isinstance(images, str):
            from PIL import Image
            from torchvision.transforms import ToTensor

            images = [os.path.join(images, img) for img in os.listdir(images)]
            images = [Image.open(img) for img in images]
            images = [ToTensor()(img).unsqueeze(0) for img in images]
            images = torch.cat(images, 0)
        elif isinstance(images, list):
            from torchvision.transforms import ToTensor

            images = [ToTensor()(img).unsqueeze(0) for img in images]
            images = torch.cat(images, 0)

        self.clip_score_fn.update(text=prompts, images=images)

    def compute(self):
        return self.clip_score_fn.compute()


# Path list / Image.Image list; str
class HPSv2(AbstractModel):
    def __init__(self, device="cuda"):
        pass

    def update(self, imgs_path: Union[str, List[Image.Image]], prompt: str):
        self.imgs_path = [os.path.join(imgs_path, img) for img in os.listdir(imgs_path)]
        self.prompt = prompt

    def compute(self, hps_version="v2.1"):
        import hpsv2

        return hpsv2.score(self.imgs_path, self.prompt, hps_version)



class PickScore_v1(AbstractModel):
    def __init__(
        self,
        device="cuda",
        processor_name_or_path="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        model_pretrained_name_or_path="yuvalkirstain/PickScore_v1",
    ):
        from transformers import AutoProcessor, AutoModel

        self.device = device
        self.processor = AutoProcessor.from_pretrained(processor_name_or_path)
        self.model = (
            AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(device)
        )

    def update(self, prompt: str, images: Union[str, List[Image.Image]]):
        if isinstance(images, str):
            from PIL import Image

            images = [os.path.join(images, img) for img in os.listdir(images)]
            self.images = [Image.open(img) for img in images]
        else:
            self.images = images
        self.prompt = prompt

    def compute(self):
        image_inputs = self.processor(
            images=self.images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        text_inputs = self.processor(
            text=self.prompt,
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



class ImageReward(AbstractModel):
    def __init__(self, model_name="ImageReward-v1.0", device="cuda"):
        import ImageReward as RM

        self.model = RM.load(model_name)

    def update(self, prompt: str, images: Union[str, List[Image.Image]]):
        if isinstance(images, str):
            from PIL import Image

            images = [os.path.join(images, img) for img in os.listdir(images)]
            self.images = [Image.open(img) for img in images]
        else:
            self.images = images
        self.prompt = prompt

    def compute_rank(self):
        ranking, reward = self.model.inference_rank(self.prompt, self.images)
        return ranking, reward

    def compute_score(self):
        return self.model.score(self.prompt, self.images)

    def compute(self):
        return self.compute_score(), self.compute_rank()


class RPrecision(Metric):
    def __init__(
        self,
        feature: int = 512,
        limit: int = 30000,
        device="cuda",
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(**kwargs)

        self.limit = limit
        self._dtype = torch.float64

        for k in ["x", "y"]:  # x: real, y: text, x0: fake
            self.add_state(f"{k}_feat", [], dist_reduce_fx=None)
        if device == "cuda":
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")

    def update(self, x: Union[Tensor, str], y: Union[Tensor, list[str]]) -> None:
        _, x, y = process_images_and_text(x, y, clip_model=None, device=self.device)

        assert x.shape[0] == y.shape[0] and x.shape[1] == y.shape[1]

        self.orig_dtype = x.dtype
        x, y = [x.double() for x in [x, y]]
        self.x_feat.append(x)
        self.y_feat.append(y)

    def compute(self, reduction: bool = True, mode=None) -> Tensor:
        r"""
        Calculate the R-Precision score based on accumulated extracted features.
        """
        feats = [torch.cat(getattr(self, f"{k}_feat"), dim=0) for k in ["x", "y"]]

        return self._compute(*feats, reduction).to(self.orig_dtype)

    def _compute(self, X: Tensor, Y: Tensor, reduction):
        def dot(x, y):
            return (x * y).sum(dim=-1)

        excess = X.shape[0] - self.limit
        if 0 < excess:
            X, Y = [x[:-excess] for x in [X, Y]]

        scores = []
        scores.append(dot(X, Y))
        for i in range(99):  # negative scores
            Y_ = Y[torch.randperm(Y.shape[0])]
            scores.append(dot(X, Y_))
        scores = torch.stack(scores, dim=-1)  # N x 100
        _, idx = scores.max(dim=-1)

        if reduction:
            return (idx == 0).float().mean()
        else:
            return (idx == 0).float()



class SemanticObjectAccuracy(Metric):
    def __init__(
        self,
        root: str = "./darknet",
        img_size: int = 256,
        confidence: float = 0.5,
        nms_thresh: float = 0.4,
        limit: int = 30000,
        device="cuda",
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(**kwargs)
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
        self.transform = T.Compose(
            [T.Resize((self.img_size, self.img_size)), T.ToTensor()]
        )

    def setup(self):
        print("Loading network ...")

        from darknet import Darknet

        self.model = Darknet(os.path.join(self.root, "cfg/yolov3.cfg"))
        self.model.load_weights(os.path.join(self.root, "yolov3.weights"))
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
        tokens = caption.lower().split(" ")

        WORDS = {
            "person": [
                "person",
                "people",
                "human",
                "man",
                "men",
                "woman",
                "women",
                "child",
                "children",
            ],
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
            "orange": ["oranges"],
        }

        def remove_words(caption, excludes):
            for w in excludes:
                caption = caption.replace(w, "")
            return caption

        # print(self.classes)
        for c in self.classes:
            if c not in WORDS.keys():
                WORDS[c] = [c]
        # multiple words
        for label, words in WORDS.items():
            if "dog" == label:
                caption_ = remove_words(
                    caption,
                    [
                        "hot dog",
                        "cheese dog",
                        "chili dog",
                        "corn dog",
                        "hotdog",
                        "hot-dog",
                    ],
                )
            elif "elephant" == label:
                caption_ = remove_words(caption, ["toy elephant", "stuffed elephant"])
            elif "car" == label:
                caption_ = remove_words(
                    caption,
                    [
                        "train car",
                        "car window",
                        "side car",
                        "passenger car",
                        "subway car",
                        "car tire",
                        "rail car",
                        "tram car",
                        "street car",
                        "trolly car",
                    ],
                )
            elif "kite" == label:
                caption_ = remove_words(caption, ["kite board", "kiteboard"])
            elif "cake" == label:
                caption_ = remove_words(caption, ["cupcake"])
            elif "bicycle" == label:
                caption_ = remove_words(
                    caption, ["motorbike", "dirt bike", "motocycle", "motor bike"]
                )
            elif "bear" == label:
                caption_ = remove_words(
                    caption, ["teddy bear", "stuffed bear", "care bear", "toy bear"]
                )
            elif "bowl" == label:
                caption_ = remove_words(caption, ["toilet bowl"])
            elif "tie" == label:
                caption_ = remove_words(caption, ["to tie"])
            elif "apple" == label:
                caption_ = remove_words(caption, ["pineapple"])
            elif "oven" == label:
                caption_ = remove_words(caption, ["microwave oven"])
            else:
                caption_ = caption

            for w in words:
                if 1 == len(w.split(" ")):
                    if w in caption_.lower().split(" "):
                        labels.append(label)
                else:
                    if w in caption_.lower():
                        labels.append(label)
        return labels

    def update(
        self, images: Union[Tensor, str], captions: List[str], is_real: bool = False
    ) -> None:
        if isinstance(images, str):
            images = load_images_from_folder(images, self.transform)

        with torch.no_grad():
            predictions = self.model(images)
            predictions = non_max_suppression(
                predictions, self.confidence, self.nms_thresh
            )

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
                        img_bboxs.append(
                            [
                                bbox_x.cpu().numpy(),
                                bbox_y.cpu().numpy(),
                                bbox_width.cpu().numpy(),
                                bbox_height.cpu().numpy(),
                            ]
                        )
                if not is_real:
                    self.predictions.append(list(img_preds_name))
                else:
                    self.reals.append(img_preds_name)

        if not is_real:
            for caption in captions:
                self.labels.append(self.get_labels(caption))

            print(self.labels)
            print(self.predictions)
            assert len(self.predictions) == len(self.labels)

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
                    sum([1.0 for x in set(preds) if x in labels]) / len(labels)
                )
        accuracy = torch.Tensor(accuracy)
        if 0 < division_by_zero:
            print(f"warning: {division_by_zero} samples have no detection.")

        if reduction:
            return accuracy[: self.limit].mean()
        else:
            return accuracy[: self.limit]


class MutualInformationDivergence(Metric):
    def __init__(
        self,
        feature: int = 512,
        limit: int = 30000,
        device="cuda",
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(**kwargs)

        self.limit = limit
        self._debug = False
        self._dtype = torch.float64
        if device == "cuda":
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")

        for k in ["x", "y", "x0"]:  # x: real, y: text, x0: fake
            self.add_state(f"{k}_feat", [], dist_reduce_fx=None)

    def update(
        self, x: Union[Tensor, str], y: Union[Tensor, list[str]], x0: Union[Tensor, str]
    ) -> None:
        if isinstance(x, str) and isinstance(x0, str) and isinstance(y, list):
            x, y, x0 = process_images_and_text(
                x, y, clip_model=None, device=self.device, real_imgs_folder=x0
            )
        assert x.shape[0] == y.shape[0] and x.shape[1] == y.shape[1]

        self.orig_dtype = x.dtype
        x, y, x0 = [x.double() for x in [x, y, x0]]
        self.x_feat.append(x)
        self.y_feat.append(y)
        self.x0_feat.append(x0)

    def compute(self, reduction: bool = True, mode=None) -> Tensor:
        r"""
        Calculate the MID score based on accumulated extracted features.
        """
        feats = [torch.cat(getattr(self, f"{k}_feat"), dim=0) for k in ["x", "y", "x0"]]

        return _compute_pmi(*feats, self.limit, reduction).to(self.orig_dtype)


class Vbench(AbstractModel):
    def __init__(
        self,
        device="cuda",
        full_json_dir="VBench_full_info.json",
        output_path="./vbench_evaluation_results/",
    ):
        from vbench import VBench

        self.my_VBench = VBench(device, full_json_dir, output_path)

    def update(self, videos_path, dimension=["temporal_flickering"]):
        self.videos_path = videos_path
        self.dimension = dimension

    def compute(self):
        import datetime
        import json

        current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        self.my_VBench.evaluate(
            videos_path=self.videos_path,
            name=f"results_{current_time}",
            dimension_list=self.dimension,
        )
        file = open(
            f"./vbench_evaluation_results/results_{current_time}_eval_results.json"
        )
        json_result = json.load(file)
        file.close()
        return json_result



