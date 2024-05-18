import os
import re
from urllib.request import urlopen

import imagehash
import numpy as np
import pandas as pd
from fastai.vision.all import (
    Image,
    Path,
    download_images,
    get_image_files,
    resize_images,
    verify_images,
)

from settings import container, root


def image_downloader(
    base, categories=None, sample_size=200, image_size=512, skip_downloads=False
):
    """
    f(base:String, categories:{String:String[]}, sample_size:Int) => root:String

    funkcija za preuzimanje slika s wikiarta ravnajući se prema dobivenoj bazi i kategorijama stilova.
    Slike se pune u root diektoriju u mapu "artefacts"
    """

    data = pd.read_parquet(f"{root}/data/{base}.parquet")

    if categories is None:
        categories = {
            "breath": {"abstract": ["Abstract Art"], "figurative": ["Naturalism"]},
            "depth": {"symbolic": ["Classicism"], "iconic": ["Post-Impressionism"]},
        }
    sample_size = sample_size
    if not skip_downloads:
        for key, cats in categories.items():
            path = container / key
            if not path.exists():
                path.mkdir()
            for cat, styles in cats.items():
                print("\n", cat, len(styles), "×", int(sample_size / len(styles)))
                for style in styles:
                    print(style)
                    sample = data.loc[data["style"].str.contains(style)].sample(
                        int(sample_size / len(styles))
                    )
                    urls = sample["webUrl"].values
                    download_images(dest=path / cat, urls=urls)
                    resize_images(path / cat, max_size=image_size, dest=path / cat)

    deleted = remove_duplicates(container)
    failed = verify_images(get_image_files(container))
    failed.map(Path.unlink)
    print("\n Neispravnih:", len(failed))

    return categories, deleted


def page_scraper(url):

    page = urlopen(url)
    html = page.read().decode("utf-8")

    src = "<(?:img|figure)[\s]*.*?(?:src|image)(?:set|Set)?=[\"'\s]+((?:http)[^'\"\s]+(?:.jpg|.jpeg|.png|.gif|,))[^'\"]*?[\"']+.*?>"
    bgurl = "url\([\s]*?((?:http)[^\"'\s]+(?:.jpg|.jpeg|.png))[^\"']*?\)"

    # find CSS background images
    bgimage_list = re.findall(bgurl, html, re.IGNORECASE)
    # find img tags
    image_list = re.findall(src, html, re.IGNORECASE)
    all_images = bgimage_list + image_list

    # print(all_images)
    return [im for im in all_images if "data:" not in im]


def alpha_remover(image):
    if image.mode != "RGBA":
        return image
    canvas = Image.new("RGBA", image.size, (255, 255, 255, 255))
    canvas.paste(image, mask=image)
    return canvas.convert("RGB")


def with_ztransform_preprocess(hashfunc, hash_size=8):
    def function(path):
        image = alpha_remover(Image.open(path))
        image = image.convert("L").resize((hash_size, hash_size), Image.LANCZOS)
        data = image.getdata()
        quantiles = np.arange(100)
        quantiles_values = np.percentile(data, quantiles)
        zdata = (np.interp(data, quantiles_values, quantiles) / 100 * 255).astype(
            np.uint8
        )
        image.putdata(zdata)
        return hashfunc(image)

    return function


def remove_duplicates(directory=container):
    """
    prema [towardsdatascience.com](https://towardsdatascience.com/removing-duplicate-or-similar-images-in-python-93d447c1c3eb)
    """
    container = Path(directory)
    deleted = []
    dhash_z_transformed = with_ztransform_preprocess(imagehash.dhash, hash_size=8)
    for folder in next(os.walk(container))[1]:
        for sub in next(os.walk(container / folder))[1]:
            img_hashes = set()
            images = get_image_files(container / folder / sub)
            for img in images:
                img_hash = dhash_z_transformed(img)
                if img_hash in img_hashes:
                    # print(f"Pronađen duplić {img}")
                    img.unlink()
                    deleted.append(img)
                else:
                    img_hashes.add(img_hash)
    return deleted
