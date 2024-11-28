from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_tf_data_pipeline(train_data_dir, test_data_dir, target_size=(224, 224), batch_size=32):
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
    )
    train_data = datagen.flow_from_directory(
        directory=train_data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_data = datagen.flow_from_directory(
        directory=test_data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    return train_data, test_data