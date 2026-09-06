from fastai.vision.all import (
    DataBlock,
    ImageBlock,
    CategoryBlock,
    TransformBlock,
    RandomSplitter,
    Resize,
    ShowGraphCallback,
    error_rate,
    get_image_files,
    minimum,
    parent_label,
    aug_transforms,
    slide,
    valley,
    vision_learner,
    load_learner,
)
from fastai.learner import Learner
import torch, torchvision
from fastai.vision.models import resnet34

from settings import garage


def create_dataloaders(root, categories, show_batch=True):
    """f(root:Path(string), categories:{String:{String:String[]}}) => learners:Learner[]"""
    dataloaders = {}
    for key in categories.keys():
        dataloaders[key] = DataBlock(
            blocks=(ImageBlock, CategoryBlock),
            get_items=get_image_files,
            splitter=RandomSplitter(valid_pct=0.2, seed=666),
            get_y=parent_label,
            item_tfms=[Resize(256, method="pad")],
        ).dataloaders(root / key)

        if show_batch:
            dataloaders[key].show_batch(max_n=9)

    return dataloaders


def create_dataloaders_multicat(root, categories):
    dataloaders = DataBlock(
        blocks=(
            ImageBlock,
            CategoryBlock,
            CategoryBlock,
        ),  # ovo bi bilo gdje definiram da će ulaz biti slika, ali da će izlaz biti klasifikacija po dvije kategorije. No to mi za ništo ne treba
        m_inp=1,  # definicija je da je ulaz samo jedan (dakle od gornje tri rubrike, dva su izlaza)
        get_items=get_image_files,
        get_y=[parent_label, parent_label],
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        item_tfms=Resize(256, method="pad"),
    ).dataloaders(
        root
    )  # ovdje trebam otkriti kako ga natjerati da skuži da se radi o dvjema protegama, a ne o četirima nezavisnim kategorijama

    return dataloaders

# for the CBM model training 
# (and I should also refactor the create_dataloaders function to accommodate for new image naming scheme -- same as dataframe index --- contentId) 
# ----------------------------------------------------------------------------------------------------------------------------

def path_to_content_id(f_path):
    """
    artefacts/{axis}/{class}/{contentId}.jpg -> integer WikiArt contentId.
    """
    return int(f_path.stem)


def get_y_cbm(f_path, class_map, concept_map, concept_labels):
    cls = f_path.parent.name

    if cls not in class_map:
        raise KeyError(
            f"Folder class {cls!r} is not in class_map={list(class_map)}. "
            f"File: {f_path}"
        )

    class_id = class_map[cls]
    content_id = path_to_content_id(f_path)

    if content_id not in concept_map:
        raise KeyError(
            f"contentId={content_id} from {f_path} is absent from the "
            "concept dataframe. Do not use fallback concept values."
        )

    concepts_dict = concept_map[content_id]

    missing_concepts = [
        concept_name
        for concept_name in concept_labels
        if concept_name not in concepts_dict
    ]

    if missing_concepts:
        raise KeyError(
            f"contentId={content_id} lacks concept columns: "
            f"{missing_concepts}"
        )

    concepts = [
        float(concepts_dict[concept_name])
        for concept_name in concept_labels
    ]

    return class_id, torch.tensor(concepts, dtype=torch.float32)


def make_get_y(class_map, concept_map, concept_labels):
    def get_y_fn(f_path):
        return get_y_cbm(
            f_path=f_path,
            class_map=class_map,
            concept_map=concept_map,
            concept_labels=concept_labels,
        )
    return get_y_fn


def create_dataloaders_cbm(
    root,
    categories,
    dataframe,
    concept_labels,
    bs=32,
    show_batch=True,
    valid_pct=0.2,
    seed=666,
):
    df = dataframe.copy()

    if df.index.name != "contentId":
        raise ValueError(
            f"Expected dataframe index named 'contentId'; "
            f"received {df.index.name!r}."
        )

    df.index = df.index.astype("int64")

    if not df.index.is_unique:
        raise ValueError("contentId index must be unique.")

    missing_columns = [
        col for col in concept_labels
        if col not in df.columns
    ]
    if missing_columns:
        raise KeyError(
            f"Missing concept columns: {missing_columns}"
        )

    concept_map = (
        df[concept_labels]
        .astype("float32")
        .to_dict(orient="index")
    )

    class_maps = {
        axis: {
            class_name: class_idx
            for class_idx, class_name
            in enumerate(categories[axis].keys())
        }
        for axis in categories
    }

    dataloaders = {}

    for axis in categories:
        images = list(get_image_files(root / axis))

        if not images:
            raise FileNotFoundError(
                f"No images found under {root / axis}"
            )

        file_ids = [path_to_content_id(path) for path in images]

        unknown_ids = sorted(set(file_ids) - set(concept_map))
        if unknown_ids:
            raise ValueError(
                f"{axis}: found {len(unknown_ids)} image files whose "
                "contentId is absent from dataframe. "
                f"First 20: {unknown_ids[:20]}"
            )

        get_y_fn = make_get_y(
            class_map=class_maps[axis],
            concept_map=concept_map,
            concept_labels=concept_labels,
        )

        dataloaders[axis] = DataBlock(
            blocks=(ImageBlock, (CategoryBlock, TransformBlock)),
            get_items=lambda items=images: items,
            splitter=RandomSplitter(
                valid_pct=valid_pct,
                seed=seed,
            ),
            get_y=get_y_fn,
            item_tfms=[Resize(256, method="pad")],
            batch_tfms=aug_transforms(),
        ).dataloaders(
            images,
            bs=bs,
            num_workers=0,
        )

        if show_batch:
            print(f"{axis} batch:")
            dataloaders[axis].show_batch(max_n=9)

    return dataloaders

# ----------------------------------------------------------------------------------------------------------------------------
def get_device():
    print("torch version:", torch.__version__)
    print("torchvision version:", torchvision.__version__)
    print(f"CUDA is available: {torch.cuda.is_available()}.", f"Activated version of CUDA is {torch.version.cuda}.", torch.cuda.device(0))
    print(f"Volta's gencode (sm_86) for RTX 3050 and 3070 is in arch_list: {'sm_86' in torch.cuda.get_arch_list()}.")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_learners(dataloaders, model=resnet34):
    """f(dataloaders:DataBlock.loaders(), model:fastai.vision.models) ==> learners:{ "axis":vision.models[n]... }"""
    device = get_device()
    learners = {}
    for key in dataloaders.keys():
        print(f">>> Preparing learner for {key}! >>>")
        learners[key] = vision_learner(dataloaders[key], model, metrics=error_rate)
        learners[key].to(device)
    return learners


def train_learners( learners, model_pick, iters=3, lr=None, show_results=False, export=False ):
    """f( learners:{ "category_n": vision.models[n]... }, iters:Int, export:Bool ) => void"""

    for key in learners.keys():
        if lr is None:
            lrs = learners[key].lr_find(suggest_funcs=(minimum, valley, slide))
            lr = sum(lrs) / len(lrs)
            print(
                f"Best learning rate for {key} {iters}x is {lr}, because thats the μ of {lrs}."
            )

        print(f">>> Training {key} {iters}x! >>>")
        learners[key].fine_tune(iters, lr, cbs=[ShowGraphCallback()])

        if show_results:
            learners[key].show_results()

        if export:
            # spremi model na disk
            spot = garage / f"{model_pick}_{iters}x"
            if not spot.exists():
                spot.mkdir()
            learners[key].export(spot / f"{key}.pkl")

# -------------------------------------------------------------------------------------------------------------


import torch.nn as nn
from fastai.basics import Module
from fastai.metrics import CrossEntropyLossFlat

class CBMModule(Module):
    def __init__(self, backbone, n_concepts, n_classes):
        super().__init__()
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool2d(1)

        backbone.eval()
        with torch.no_grad():
            dummy = torch.randn(2, 3, 256, 256).to(next(backbone.parameters()).device)
            feats = backbone(dummy)
            pooled = self.pool(feats).flatten(1)
            in_features = pooled.shape[1]

        self.concept_head = nn.Linear(in_features, n_concepts)
        self.class_head   = nn.Linear(n_concepts, n_classes)

    def forward(self, x):
        feats = self.backbone(x)              # [B, C, H, W]
        feats = self.pool(feats)              # [B, C, 1, 1]
        feats = feats.view(feats.size(0), -1) # [B, in_features], safe flatten
        concepts = torch.sigmoid(self.concept_head(feats))  # [B, n_concepts]
        logits   = self.class_head(concepts)                # [B, n_classes]
        return logits, concepts

class CBMLoss(Module):
    def __init__(self, concept_weight=1.0, class_weight=1.0):
        super().__init__()
        self.concept_weight = concept_weight
        self.class_weight   = class_weight
        self.ce = CrossEntropyLossFlat()
        self.bce = nn.BCELoss()

    def forward(self, preds, target):
        logits, concepts = preds
        y_class, y_concepts = target
        class_loss   = self.ce(logits, y_class)
        concept_loss = self.bce(concepts, y_concepts)
        return self.class_weight * class_loss + self.concept_weight * concept_loss

def cbm_accuracy(inp, targ):
    # inp is (logits, concepts), targ is (class_ids, concept_targets)
    logits, _ = inp
    class_ids, _ = targ
    # standard accuracy on logits vs class_ids
    preds = logits.argmax(dim=-1)
    return (preds == class_ids).float().mean()


def load_cbm_learners_from_weights(dataloaders, model_choice, n_concepts, n_classes, concept_weight=1.0, class_weight=1.0):
    """f(dataloaders:DataBlock.loaders(), model_choice:str, n_concepts:Int=3, n_classes:Int, concept_weight:Float=1.0, class_weight:Float=1.0) ==> learners:{ "axis": vision.models[n]... }"""

    device = get_device()
    learners = {}

    for key in dataloaders.keys():
        print(f">>> Preparing learner for {key}! >>>")

        if model_choice:
            base_model = load_learner(garage / model_choice / f"{key}.pkl")
        else:
            raise ValueError(f"Base model's name must be provided! It must be at location {str(garage)}/model_choice/{key}.pkl, and also must have weights stored in accompanying {key}_cbm_weights.pth.")

        full_model = base_model.model           # Sequential with head
        conv_backbone = full_model[0]           # only conv features

        # Recreate CBM
        cbm = CBMModule(conv_backbone, n_concepts=n_concepts, n_classes=n_classes)
        cbm = cbm.to(device)

        # Load CBM weights
        weight_path = garage / model_choice / f"{key}_cbm_weights.pth"
        state = torch.load(weight_path, map_location=device, weights_only=True)
        cbm.load_state_dict(state)
        cbm.eval()

        # Wrap in Learner (optional but convenient)
        loss_func = CBMLoss(concept_weight=concept_weight, class_weight=class_weight)
        learn = Learner(
            dataloaders[key],
            cbm,
            loss_func=loss_func,
            metrics=[cbm_accuracy],
        )
        learn.model.eval()
        learners[key] = learn

    return learners
