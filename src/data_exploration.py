
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Data Loading and Exploration

def load_and_visualize_data(data_path, partition):

    classes = ['meningioma', 'glioma', 'pituitary', 'notumor']
    data = []
    for label in classes:
        folder_path = os.path.join(data_path, label)
        for image_file in os.listdir(folder_path):
            data.append((os.path.join(folder_path, image_file), label))

    df = pd.DataFrame(data, columns=["image_path", "label"])
    df['label'] = pd.Categorical(df['label'], categories=classes)

    # Plot class distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='label', order=classes, hue='label')
    plt.title(partition + ' Data Class Distribution')
    plt.show()

    return df




