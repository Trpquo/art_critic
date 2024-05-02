from audioop import avg

from fastai.vision.all import (
    CategoryBlock,
    DataBlock,
    ImageBlock,
    RandomSplitter,
    Resize,
    error_rate,
    get_image_files,
    minimum,
    parent_label,
    slide,
    steep,
    valley,
    vision_learner,
)
from fastai.vision.models import resnet34


def create_dataloaders(root, categories):
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


def create_learners(dataloaders, model=resnet34):
    """f(dataloaders:DataBlock.loaders(), model:fastai.vision.models, iters:Int=3, export:Bool=False) ==> learners:{ "category_n":vision.models[n]... }"""

    learners = {}
    for key in dataloaders.keys():
        print(f">>> Preparing {key}! >>>")
        learners[key] = vision_learner(dataloaders[key], model, metrics=error_rate)
    return learners


def train_learners(learners, iters, lr, export=False):
    """f( learners:{ "category_n": vision.models[n]... }, iters:Int, export:Bool ) => void"""

    for key in learners.keys():
        if lr == None:
            lrs = learners[key].lr_find(suggest_funcs=(minimum, valley, slide))
            lr = sum(lrs) / len(lrs)
            print(
                f"Best learning rate for {key} is {lr}, because thats the μ of {lrs}."
            )
        print(f">>> Training {key}! >>>")
        learners[key].fine_tune(iters, lr)
        if export:
            learners[key].export(f"../models/{key}.pkl")
