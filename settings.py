from pathlib import Path

root = Path("./")
garage = root / "models"
container = root / "artefacts"

database = "wikiart_composed"
categories = {
    "breath": {
        "abstract": [
            "Hard Edge Painting",
            "Concretism",
            "Suprematism",
            "Abstract Art",
            "Abstract Expressionism",
            "Action painting",
        ],  # "Constructivism",
        "concrete": [
            "Northern Renaissance",
            "Hyper-Realism",
            "Tenebrism",
            "Academicism",
            "Classical Realism",
            "Naturalism",
            "Baroque",
        ],  # "Rococo"
    },
    "depth": {
        "symbolic": [
            "International Gothic",
            "Byzantine",
            "Pop Art",
            "High Renaissance",
            "Classicism",
            "Mannerism \(Late Renaissance\)",
            "Romanesque",
        ],
        "iconic": [
            "Color Field Painting",
            "Abstract Expressionism",
            "Action painting",
            "Lyrical Abstraction",
            "Post-Impressionism",
            "Analytical Realism",
            "Academicism",
            "Rococo",
            "Pointillism",
        ],  # "Art Deco", "Art Nouveau \(Modern\)"
    },
}
sample_size = 500
image_size = 224

# (v. https://github.com/fastai/fastai/tree/master/fastai/vision/models)
# "alexnet", "googlenet", "squeezenet"
# ovi na žalost iz nekog razloga ne idu, a trebali bi biti najbrži (levit) --> "vit_b_16" (najmanji), "vit_b_32", "vit_l_32" (veliki), "vit_h_14" (najveći)
# ne rade ni ovi --> "inception_v3", "densenet121", "swin_v2_b", "efficientnet_b0"
# "resnet18", "resnet34", "resnet50",,
# "xresnet18",  "xresnet18_deeper",  "xresnet34",  "xresnet34_deeper", "xresnet50", "xresnet50_deeper",
# "vgg11_bn", "vgg16_bn", "vgg19_bn" (iz nekog razloga ne mogu veći od 11)
# "convnext_tiny", "convnext_base", "convnext_large" (najbolje radi na 3 epohe, ne će veće od tiny)
iteration_counts = [1, 3, 10]
model_picks = [
    # AlexNet 13 layers
    "alexnet",
    # GoogLeNet 22 layers
    "googlenet",
    # ResNet
    "resnet18",
    "resnet34",
    "resnet50",
    # xresnet
    "xresnet18",
    "xresnet34",
    "xresnet50",
    "xresnet18_deeper",
    "xresnet34_deeper",
    "xresnet50_deeper",
    # vgg
    "vgg11_bn",
    # convnext
    "convnext_tiny",
]
