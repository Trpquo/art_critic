from fastai.vision.all import Image, PILImage
from fastdownload import download_url

def test_learners(learners, root, preview=False):
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
    for image in range(len(internet)):
        if preview:
            download_url(internet[0], dest=dest/"test.jpg")
            im = Image.open(dest/"test.jpg")
            im.to_thumb(128, 128)
        base = f"test{image+1}.jpg"
        test = dest/base
        print("\n\n#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%")
        print((str(image+1)+" - ")*10)
        download_url(internet[image], dest=test, show_progress=False)

        for key in learners.keys():
            prediction,_,probs = learners[key].predict(PILImage.create(test))
            print(f'Image {base} is {prediction}.')
            print(*zip( ["abstract", "cocrete"], probs ))
