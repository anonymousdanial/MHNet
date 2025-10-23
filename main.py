import danial.vgg as vgg
import danial.dataloader as dataloader


if __name__ == "__main__":
    test = dataloader.load_image("assets/RR.png")

    print(test.shape)
    vgg = vgg.backbone()
    vgg.eval()
    features = vgg(test, return_all=True)
    for name, feat in features.items():
        print(f"{name}: {feat.shape}")