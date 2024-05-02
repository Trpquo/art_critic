import math
import random
import re
import shutil

from fastai.vision.all import (  # , verify_images
    Image,
    Path,
    PILImage,
    get_image_files,
    resize_image,
)
from fastdownload import download_url

categories = {"depth": ["iconic", "symbolic"], "breath": ["abstract", "concrete"]}


def test_learners(learners, test_set, root, preview=False):
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
    # print(test_images)
    for src in test_images:
        sample = PILImage.create(src)
        print("\n\n" + "#%&%#" * 15)
        print("- " + (re.findall("\d+", str(src.stem))[0] + " - ") * 15)
        for key in learners.keys():
            prediction, _, probs = learners[key].predict(sample)
            print(f"Image {src.name} is {prediction}.")
            print(*zip(categories[key], probs.numpy()))


def fill_database(learners, database):
    """f( learners:{"String":model...}, database, :String[])"""

    # container = root / "temp"

    if database:
        for row in database:
            sample = PILImage.create(row["webUrl"])
            for key in learners.keys():
                prediction, _, probs = learners[key].predict(sample)
                database[key] = prediction
                database[f"{key}_probs"] = probs

    return database
