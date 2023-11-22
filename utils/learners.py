from fastai.vision.all import  DataBlock, get_image_files, vision_learner, ImageBlock, CategoryBlock, RandomSplitter, Resize, parent_label, error_rate
from fastai.vision.models import resnet34

def create_dataloaders(root, categories):
    """f(root:Path(string), categories:{String:{String:String[]}}) => learners:Learner[]"""
    dataloaders = {}
    for key in categories.keys():
        dataloaders[key] = DataBlock(
            blocks=(ImageBlock, CategoryBlock),
            get_items=get_image_files,
            splitter=RandomSplitter(valid_pct=.2, seed=666),
            get_y=parent_label,
            item_tfms=[ Resize(256, method="pad") ]
        ).dataloaders( root/key )
        dataloaders[key].show_batch( max_n=9 )

    return dataloaders


def create_learners(dataloaders, model=resnet34):
    """f(dataloaders:DataBlock.loaders(), model:fastai.vision.models, iters:Int=3, export:Bool=False) ==> learners:{ "category_n":vision.models[n]... }"""
    
    learners = {}
    for key in dataloaders.keys():
        print(f">>> Preparing {key}! >>>")
        learners[key] = vision_learner(dataloaders[key], model, metrics=error_rate)
        learners[key].lr_find()
    return learners

def train_learners(learners, iters, lr, export=False):
    """f( learners:{ "category_n": vision.models[n]... }, iters:Int, export:Bool ) => void"""

    for key in learners.keys():
        print(f">>> Training {key}! >>>")
        learners[key].fine_tune(iters, lr)
        if export:
            learners[key].export(f"../models/{key}.pkl")
