import math
import random
import re
import shutil
from xml.parsers.expat import model

import pandas as pd
from fastai.vision.all import (  # , verify_images
    Image,
    Path,
    PILImage,
    get_image_files,
    resize_image,
)
from fastdownload import download_url

categories = {"depth": ["iconic", "symbolic"], "breath": ["abstract", "concrete"]}


def test_learners(learners, test_set, model_name, root, preview=False):
    """f( learners:{"String":model...}, test_set:String[], root:Path, preview:Bool )"""

    container = root / "temp"

    print(test_set)
    if test_set:
        print("ajmo00oooooo!")
        if container.exists():
            shutil.rmtree(container)
        else:
            container.mkdir()
        counter = 1
        for image in range(len(test_set)):
            if counter > 9:
                break
            # trunk-ignore(bandit/B311)
            if random.randint(0, 3) > 1:
                continue
            # image_name = test_set[image].split("/")[-1]
            base = f"test{counter}.jpg"
            dest = container / base
            if preview:
                download_url(test_set[0], dest=dest, show_progress=False)
                im = Image.open(dest)
                im.to_thumb(128, 128)
            download_url(test_set[image], dest=dest, show_progress=False)

            if math.prod(PILImage.create(dest).size) > 5e4:
                resize_image(dest, dest=Path("."), max_size=244)
                counter += 1
            else:
                try:
                    dest.unlink()
                except Exception as ex:
                    print("\n\n", f"!! ////// Slika {dest} je premala. ////// !!")
                    print(ex)

    test_images = get_image_files(container).sorted()
    print(f"//////////////// {model_name.upper()} ////////////////")
    for src in test_images:
        sample = PILImage.create(src)
        print("\n\n" + "#%&%#" * 15)
        print("- " + (re.findall("\d+", str(src.stem))[0] + " - ") * 15)
        for key in learners.keys():
            prediction, _, probs = learners[key].predict(sample)
            print(f"Image {src.name} is {prediction}.")
            print(*zip(categories[key], probs.numpy()))


def predict_columns(learners, database, model_name, root):
    """f( learners:{"String":model...}, database, model_name:String[]) => result:list(of files)"""

    container = root / "temp"
    warehouse = root / "data" / model_name
    result = []
    counter = 1

    for directory in (container, warehouse):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir()

    if isinstance(database, pd.DataFrame):
        data_left = len(database)
        for index, row in database.iterrows():
            data_left -= 1
            base = f"test{index}.jpg"
            dest = container / base
            try:
                download_url(
                    row["webUrl"].replace("!Large.jpg", ""),
                    dest=dest,
                    show_progress=False,
                )
                sample = PILImage.create(dest)
                dest.unlink()
            except Exception as e:
                print(e)
                sample = None
            if sample:
                for key in learners.keys():
                    prediction, _, probs = learners[key].predict(sample)
                    row[f"{model_name}_{key}"] = prediction
                    row[f"{model_name}_{key}_probs0"] = probs[0].item()
                    row[f"{model_name}_{key}_probs1"] = probs[1].item()
                result.append(row)

                datafile = f"{warehouse}/critic_output{counter}.parquet"
                if len(result) >= 1000 or data_left == 0:
                    output = pd.DataFrame(result)
                    result.append(datafile)
                    column_selection = [
                        "artistName",
                        "title",
                        "year",
                        "style",
                        f"{model_name}_breath",
                        f"{model_name}_breath_probs0",
                        f"{model_name}_breath_probs1",
                        f"{model_name}_depth",
                        f"{model_name}_depth_probs0",
                        f"{model_name}_depth_probs1",
                        "genre",
                        "artemis",
                        "emotions",
                        "webUrl",
                    ]
                    print(column_selection)
                    output[column_selection].to_parquet(datafile)
                    print(f"Saved the output {counter}!")
                    counter += 1
                    result = []

            else:
                print("NEMA!!!", row["webUrl"])

    return result



















 
