import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical


def load_processed_images(data_dir, img_size=(128, 128)):
    X, y = [], []
    class_names = sorted(os.listdir(data_dir))

    for class_idx, class_name in enumerate(class_names):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_path, img_file)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, img_size)
                        img = img.astype('float32') / 255.0
                        X.append(img)
                        y.append(class_idx)

    X = np.expand_dims(np.array(X), -1)
    y = np.array(y)
    return X, y, class_names


def create_data_generators(X, y_encoded, batch_size=32, augment=True):
    y_cat = to_categorical(y_encoded)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_cat, stratify=y_encoded, test_size=0.3, random_state=42)

    y_temp_enc = np.argmax(y_temp, axis=1)

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, stratify=y_temp_enc, test_size=0.5, random_state=42)

    if augment:
        train_aug = ImageDataGenerator(
            rotation_range=10,
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True
        )
    else:
        train_aug = ImageDataGenerator()

    test_aug = ImageDataGenerator()

    train_gen = train_aug.flow(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_gen = test_aug.flow(X_val, y_val, batch_size=batch_size, shuffle=False)
    test_gen = test_aug.flow(X_test, y_test, batch_size=batch_size, shuffle=False)

    return train_gen, val_gen, test_gen


def create_tf_data_pipeline(X, y_encoded, batch_size=32, buffer_size=512, augment=True):
    y_cat = to_categorical(y_encoded)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_cat, stratify=y_encoded, test_size=0.3, random_state=42)

    y_temp_enc = tf.argmax(y_temp, axis=1).numpy()

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, stratify=y_temp_enc, test_size=0.5, random_state=42)

    def augment_fn(image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, 0.9, 1.1)
        return image, label

    def prepare_ds(X, y, training=False):
        ds = tf.data.Dataset.from_tensor_slices((X, y))
        if training and augment:
            ds = ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = prepare_ds(X_train, y_train, training=True)
    val_ds   = prepare_ds(X_val, y_val, training=False)
    test_ds  = prepare_ds(X_test, y_test, training=False)

    return train_ds, val_ds, test_ds


def visualize_batch(generator_or_dataset, class_names, framework='keras'):
    if framework == 'keras':
        images, labels = next(generator_or_dataset)
    else:
        for images, labels in generator_or_dataset.take(1):
            images = images.numpy()
            labels = labels.numpy()

    labels = np.argmax(labels, axis=1) if labels.ndim > 1 else labels

    plt.figure(figsize=(10, 8))
    for i in range(min(9, len(images))):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title(class_names[labels[i]])
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def build_test_generator(batch_size=32, shuffle=False, target_size=(128, 128), color_mode="grayscale"):
    test_dir = "data/test"  # Make sure this folder exists with subfolders per class
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=shuffle,
        color_mode=color_mode
    )
    class_names = list(test_gen.class_indices.keys())
    return test_gen, class_names


def predict_image(file_path, model, class_names, img_size=(128, 128)):
    """
    Predict class of a single palm image.
    """
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not read image. Check file path.")
    
    img = cv2.resize(img, img_size).astype('float32') / 255.0
    img = np.expand_dims(img, axis=(0, -1))  # Shape: (1, 128, 128, 1)

    pred_probs = model.predict(img)
    pred_class_idx = np.argmax(pred_probs)
    pred_class_name = class_names[pred_class_idx]

    return pred_class_idx, pred_class_name, float(np.max(pred_probs))
