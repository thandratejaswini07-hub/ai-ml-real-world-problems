"""Optional: small U-Net for semantic segmentation of satellite tiles (needs TensorFlow).

Use with labelled tiles (e.g. from the DeepGlobe or EuroSAT-style datasets):
    model = build_unet((128, 128, 4), num_classes=4)
    model.fit(train_images, train_masks, epochs=20, validation_split=0.2)
"""
from tensorflow.keras import layers, models


def conv_block(x, f):
    x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
    return layers.Conv2D(f, 3, padding="same", activation="relu")(x)


def build_unet(input_shape=(128, 128, 4), num_classes=4):
    inp = layers.Input(input_shape)
    c1 = conv_block(inp, 16); p1 = layers.MaxPooling2D()(c1)
    c2 = conv_block(p1, 32);  p2 = layers.MaxPooling2D()(c2)
    c3 = conv_block(p2, 64);  p3 = layers.MaxPooling2D()(c3)
    b = conv_block(p3, 128)
    u3 = layers.Concatenate()([layers.UpSampling2D()(b), c3]); c4 = conv_block(u3, 64)
    u2 = layers.Concatenate()([layers.UpSampling2D()(c4), c2]); c5 = conv_block(u2, 32)
    u1 = layers.Concatenate()([layers.UpSampling2D()(c5), c1]); c6 = conv_block(u1, 16)
    out = layers.Conv2D(num_classes, 1, activation="softmax")(c6)
    model = models.Model(inp, out)
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


if __name__ == "__main__":
    build_unet().summary()
