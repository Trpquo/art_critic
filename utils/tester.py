import math
import random
import re
import shutil

import pandas as pd
from fastai.vision.all import (  # , verify_images
    Image,
    Path,
    PILImage,
    get_image_files,
    resize_image,
)
from fastdownload import download_url
import torch

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


@torch.inference_mode()
def predict_cbm_image(learn, sample):
    """
    Predict the binary task class from one PIL image using a CBM Learner.

    Returns
    -------
    predicted_label : str
    predicted_idx : int
    probabilities : list[float]
        [P(class_0), P(class_1)]
    concept_scores : list[float]
        Predicted scores for every concept node.
    """
    model = learn.model.eval()
    device = next(model.parameters()).device

    # Applies exactly the image transforms retained by the Learner/DataLoaders.
    test_dl = learn.dls.test_dl([sample], with_labels=False)
    batch = test_dl.one_batch()

    # `with_labels=False` should give xb directly, but this handles either form.
    xb = batch[0] if isinstance(batch, (tuple, list)) else batch
    xb = xb.to(device)

    class_logits, concept_scores = model(xb)

    class_probabilities = torch.softmax(class_logits, dim=1)[0]
    predicted_idx = int(class_probabilities.argmax().item())

    # Your original category ordering is encoded explicitly here.
    # This avoids fragile fastai decoding for the multi-output CBM.
    class_names = {
        "breath": ["abstract", "concrete"],
        "depth": ["iconic", "symbolic"],
    }

    # Infer the semantic axis from the supplied learner at call-site
    # rather than guessing it inside this helper.
    return (
        predicted_idx,
        class_probabilities.detach().cpu().tolist(),
        concept_scores[0].detach().cpu().tolist(),
    )

def predict_columns(learners, database, model_name, root):
    """f( learners:{"String":model...}, database, model_name:String[]) => result:list(of files)"""

    container = root / "temp"
    warehouse = root / "data" / model_name
    datafiles = []
    counter = 1
    result = []
    is_cbm = "_cbm" in model_name

    for directory in (container, warehouse):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir()

    class_names = {
        "breath": ["abstract", "concrete"],
        "depth": ["iconic", "symbolic"],
    }

    if isinstance(database, pd.DataFrame):
        data_left = len(database)
        for index, row in database.iterrows():
            data_left -= 1
            base = f"test{index}.jpg"
            dest = container / base
            row["webUrl"] = row["webUrl"].replace("!Large.jpg", "")
            try:
                download_url(
                    row["webUrl"],
                    dest=dest,
                    show_progress=False,
                )
                sample = PILImage.create(dest)
                dest.unlink()
            except Exception as e:
                # print(e)
                sample = None
            if sample:
                for key in learners.keys():
                    if is_cbm:
                        prediction_idx, probs, _ = predict_cbm_image(learners[key], sample)
                        row[key] = class_names[key][prediction_idx]
                        row[f"{key}_probs"] = probs[0]
                    else:
                        prediction, _, probs = learners[key].predict(sample)
                        row[key] = prediction
                        row[f"{key}_probs"] = probs[0].item()
                result.append(row)

                datafile = f"{warehouse}/critic_output{counter}.parquet"
                if len(result) >= 1000 or data_left == 0:
                    output = pd.DataFrame(result)
                    datafiles.append(datafile)
                    column_selection = [
                        "artistName",
                        "title",
                        "year",
                        "style",
                        "breath",
                        "breath_probs",
                        "depth",
                        "depth_probs",
                        "genre",
                        "artemis",
                        "emotions",
                        "webUrl",
                    ]
                    output[column_selection].to_parquet(datafile)
                    counter += 1
                    result = []

            else:
                print("NEMA!!!", row["webUrl"])

    return datafiles