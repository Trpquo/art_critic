#%%
# test je li sve dobro instalirano
import torch
print(torch.version.cuda, torch.cuda.is_available(), torch.cuda.device(0))

#%% 
import pandas as pd
data = pd.read_parquet("../data/wikiart_composed.parquet")

#%% 
from fastai.vision.all import *
# from fastdownload import download_images
root = Path('../artefacts')
categories = {
    "breath": { 
                "abstract": ['Hard Edge Painting', "Concretism", "Suprematism"], 
                "figurative": ["Northern Renaissance", "Mannerism \(Late Renaissance\)", "Hyper-Realism", "Tenebrism" ] 
            }, 
    "depth": {
                "symbolic": ["International Gothic", "Constructivism", "Byzantine", "Pop Art", "Art Deco", "Art Nouveau \(Modern\)"], 
                "iconic": ["Color Field Painting", "Abstract Expressionism", "Action painting", "Lyrical Abstraction", "Post-Impressionism"]
            }
    }
sample_size = 200

for key, cats in categories.items():
    path = root/key
    if not path.exists():
        path.mkdir()
    for cat, examples in cats.items():
        print(cat, len(examples), int( sample_size / len(examples) ))
        for example in examples:
            print(example)
            sample = data.loc[ data["style"].str.contains(example) ].sample( int(sample_size / len(examples)) )
            urls = sample["webUrl"].values
            download_images(dest=path/cat, urls=urls )
            resize_images(path/cat, max_size=512, dest=path/cat)


#%%
failed = verify_images( get_image_files( root ) )
failed.map( Path.unlink )
print( len(failed) )


#%% 
dls_breath = DataBlock(
    blocks=( ImageBlock, CategoryBlock ),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=.2, seed=666),
    get_y=parent_label,
    item_tfms=[ Resize(256, method="pad") ]
).dataloaders( root/"breath" )
dls_breath.show_batch( max_n=9 )

dls_depth = DataBlock(
    blocks=( ImageBlock, CategoryBlock ),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=.2, seed=666),
    get_y=parent_label,
    item_tfms=[ Resize(256, method=ResizeMethod.Pad) ]
).dataloaders( root/"depth" )
dls_depth.show_batch( max_n=9 )

#%% 
print(">>> Training breath! >>>")
learn_breath = vision_learner(dls_breath, resnet34, metrics=error_rate)
learn_breath.fine_tune(3)
print(">>> Training depth! >>>")
learn_depth = vision_learner(dls_depth, resnet34, metrics=error_rate)
learn_depth.fine_tune(4)

#%%
from fastdownload import download_url
dest = root/"temp"
if not dest.exists(): 
    dest.mkdir()
internet = [
    "https://tse2.mm.bing.net/th?id=OIP.UPZ1-G8gpc5FkNIC2RCWSgHaFj&pid=Api",
    "https://tse2.mm.bing.net/th?id=OIP.if_cidFAKZ49wY7BLA3feQHaGE&pid=Api",
    "https://tse4.mm.bing.net/th?id=OIP.Px4ySbgcqEgFJOhOq8k5mAHaEo&pid=Api",
    "https://tse4.mm.bing.net/th?id=OIP.eNfpYf9Oqyh0u3_b1Eu20wHaGL&pid=Api",
    "https://tse3.mm.bing.net/th?id=OIP.NEkbbYyu56hdqekgPKxmoQAAAA&pid=Api",
    "https://tse3.mm.bing.net/th?id=OIF.jw1Qziy4XGJbDtTYtrBY3Q&pid=Api",
    "https://tse2.mm.bing.net/th?id=OIP.pJwVOqij6rPFrOjcHc1jbAHaKj&pid=Api",
    "https://tse1.mm.bing.net/th?id=OIP.-Hy08hdnnfvCwor0Y_fyGAHaLA&pid=Api",
    "https://tse4.mm.bing.net/th?id=OIP.mUVhO-Zld2qEJKe_r9tJaAHaHn&pid=Api",
    "https://tse2.mm.bing.net/th?id=OIP.DqjuCxnocoFQ0hvIH_Cm8wHaEK&pid=Api",
    "https://tse1.mm.bing.net/th?id=OIP.EFoByYTKu1KbZgoVfocCDAHaFU&pid=Api",
    "https://tse2.mm.bing.net/th?id=OIP.BFRcQqZ5Bpw1a-6u_XoBrwHaFj&pid=Api",
    "https://tse2.mm.bing.net/th?id=OIP.2qqjQQ0wW37Lbvkx50zPxgHaEB&pid=Api"
]

#%%
download_url(internet[0], dest=dest/"test.jpg")
im = Image.open(dest/"test.jpg")
im.to_thumb(128, 128)


#%%
for image in range(len(internet)):
    base = f"test{image+1}.jpg"
    test = dest/base
    print("\n\n#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%")
    print((str(image+1)+" - ")*10)
    download_url(internet[image], dest=test, show_progress=False)
    prediction,_,probs = learn_breath.predict(PILImage.create(test))
    print(f'Image {base} is {prediction}.')
    print(*zip( ["abstract", "figurative"], probs ))
    prediction,_,probs = learn_depth.predict(PILImage.create(test))
    print(f'Image {base} is {prediction}.')
    print(*zip( ["iconic", "symbolic"], probs ))
# %%
learn_breath.export("../models/breath.pkl")
learn_depth.export("../models/depth.pkl")