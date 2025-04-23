import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# --- 1. Dataset & Základné Parametre ---
dataset_dir = '/kaggle/input/potato-leaves'
if not os.path.exists(dataset_dir):
    print(f"Warning: Dataset directory '{dataset_dir}' not found. Using placeholder data.")
    dummy_classes = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
    dataset_dir = './dummy_potato_data'
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
        for cls in dummy_classes:
            os.makedirs(os.path.join(dataset_dir, cls), exist_ok=True)
            for i in range(50):
                with open(os.path.join(dataset_dir, cls, f'dummy_{i}.txt'), 'w') as f:
                    f.write("dummy image data")
    potato_dir = dataset_dir
else:
    potential_plant_dir = os.path.join(dataset_dir, 'plants')
    if os.path.exists(potential_plant_dir) and os.path.isdir(potential_plant_dir):
         potato_dir = potential_plant_dir
    else:
         print(f"Warning: Specific plant directory '{potential_plant_dir}' not found within dataset. Using base dataset directory '{dataset_dir}'.")
         potato_dir = dataset_dir


IMG_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 40 
CLASSES = []
NUM_CLASSES = 0

# --- Funkcia na načítanie a rozdelenie datasetu ---
def load_and_split_data(base_dir, img_size, batch_size, validation_split=0.2):
    print(f"Loading data from: {base_dir}")
    try:
        dir_contents = os.listdir(base_dir)
        if not any(os.path.isdir(os.path.join(base_dir, i)) for i in dir_contents):
             print(f"Error: No subdirectories (class folders) found in '{base_dir}'.")
             return None, None, None
    except FileNotFoundError:
        print(f"Error: Directory not found: '{base_dir}'")
        return None, None, None
    except Exception as e:
        print(f"Error accessing directory '{base_dir}': {e}")
        return None, None, None

    try:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            base_dir,
            validation_split=validation_split,
            subset="training",
            seed=123,
            image_size=(img_size, img_size),
            batch_size=batch_size,
            label_mode='int'
        )

        val_test_ds = tf.keras.utils.image_dataset_from_directory(
            base_dir,
            validation_split=validation_split,
            subset="validation",
            seed=123,
            image_size=(img_size, img_size),
            batch_size=batch_size,
            label_mode='int'
        )

        val_batches = tf.data.experimental.cardinality(val_test_ds)
        if val_batches == tf.data.experimental.UNKNOWN_CARDINALITY or val_batches < 2:
             print("Warning: Not enough validation batches to split into validation and test sets.")
             print("Using the entire 20% split as validation data. Test set will be empty.")
             test_dataset = val_test_ds.take(0)
             validation_dataset = val_test_ds
        else:
            test_dataset = val_test_ds.take(val_batches // 2)
            validation_dataset = val_test_ds.skip(val_batches // 2)

        print(f"Počet tréningových dávok: {tf.data.experimental.cardinality(train_ds)}")
        print(f"Počet validačných dávok: {tf.data.experimental.cardinality(validation_dataset)}")
        print(f"Počet testovacích dávok: {tf.data.experimental.cardinality(test_dataset)}")

        loaded_class_names = train_ds.class_names
        print(f"Classes found by TensorFlow: {loaded_class_names}")
        global CLASSES, NUM_CLASSES
        CLASSES = loaded_class_names
        NUM_CLASSES = len(CLASSES)
        if NUM_CLASSES == 0:
            print("Error: No classes found in the dataset directory structure.")
            return None, None, None

        return train_ds, validation_dataset, test_dataset
    except Exception as e:
        print(f"Error loading data with image_dataset_from_directory: {e}")
        print("Please ensure the dataset directory structure is correct (e.g., dataset_dir/class_name/image.jpg) and images are valid.")
        return None, None, None

# --- Načítanie dát ---
train_dataset, validation_dataset, test_dataset = load_and_split_data(
    potato_dir, IMG_SIZE, BATCH_SIZE
)

if train_dataset is None or NUM_CLASSES == 0:
    print("Exiting due to data loading issues or no classes found.")
    exit()

# --- 2. Predspracovanie a Augmentácia (pre CNN) ---
resize_and_rescale_cnn = tf.keras.Sequential([
  layers.Resizing(IMG_SIZE, IMG_SIZE),
  layers.Rescaling(1./255)
])

data_augmentation_cnn = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),
])

AUTOTUNE = tf.data.AUTOTUNE

def prepare_cnn(ds, shuffle=False, augment=False):
  if ds is None:
      return None
  ds = ds.map(lambda x, y: (resize_and_rescale_cnn(x), y),
              num_parallel_calls=AUTOTUNE)
  ds = ds.cache()
  if shuffle:
    ds = ds.shuffle(buffer_size=1000)
  if augment:
    ds = ds.map(lambda x, y: (data_augmentation_cnn(x, training=True), y),
                num_parallel_calls=AUTOTUNE)
  return ds.prefetch(buffer_size=AUTOTUNE)

train_ds_cnn = prepare_cnn(train_dataset, shuffle=True, augment=True)
val_ds_cnn = prepare_cnn(validation_dataset)
test_ds_cnn = prepare_cnn(test_dataset)

# --- 3. Definícia CNN Modelu ---
if NUM_CLASSES <= 0:
    print("Error: Cannot define model, number of classes is zero or negative.")
    exit()

cnn_model = tf.keras.Sequential([
  layers.InputLayer(input_shape=(IMG_SIZE, IMG_SIZE, 3)), # Explicitný vstup

  # Vrstvy podľa Tabuľky 1
  layers.Conv2D(32, kernel_size = (3,3), activation='relu'),
  layers.MaxPooling2D(pool_size=(2,2)),
  layers.Conv2D(64, kernel_size = (3,3), activation='relu'),
  layers.MaxPooling2D(pool_size=(2,2)),
  layers.Conv2D(64, kernel_size = (3,3), activation='relu'),
  layers.MaxPooling2D(pool_size=(2,2)),
  layers.Conv2D(64, kernel_size = (3,3), activation='relu'),
  layers.MaxPooling2D(pool_size=(2,2)),
  layers.Conv2D(64, kernel_size = (3,3), activation='relu'),
  layers.MaxPooling2D(pool_size=(2,2)),
  layers.Conv2D(64, kernel_size = (3,3), activation='relu'),
  layers.MaxPooling2D(pool_size=(2,2)),

  layers.Flatten(),
  layers.Dense(64, activation='relu'),
  layers.Dense(NUM_CLASSES, activation='softmax') # Výstupná vrstva
])

print("--- CNN Model Summary ---")
cnn_model.summary()

# --- 4. Tréning CNN Modelu ---
cnn_model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

print("\n--- Tréning CNN modelu... ---")
start_time_cnn = time.time()

validation_data_arg = None
if val_ds_cnn and tf.data.experimental.cardinality(val_ds_cnn).numpy() > 0:
    validation_data_arg = val_ds_cnn

history = cnn_model.fit(
    train_ds_cnn,
    validation_data=validation_data_arg,
    epochs=EPOCHS
)
cnn_training_time = time.time() - start_time_cnn
print(f"Tréning CNN dokončený za {cnn_training_time:.2f} sekúnd.")

# --- 5. Evaluácia CNN Modelu ---
print("\n--- Evaluácia CNN modelu na testovacej sade... ---")
cnn_report = None
cnn_accuracy = 0.0
y_true_cnn = []
y_pred_cnn = []

if test_ds_cnn and tf.data.experimental.cardinality(test_ds_cnn).numpy() > 0:
    cnn_loss, cnn_accuracy = cnn_model.evaluate(test_ds_cnn)
    print(f"CNN Test Loss: {cnn_loss}")
    print(f"CNN Test Accuracy: {cnn_accuracy}")

    y_pred_cnn_probs = cnn_model.predict(test_ds_cnn)
    y_pred_cnn = np.argmax(y_pred_cnn_probs, axis=1)

    for _, labels in test_ds_cnn.unbatch():
        y_true_cnn.append(labels.numpy())
    y_true_cnn = np.array(y_true_cnn)

    print("\nCNN Classification Report:")
    target_names_report = [name.replace('Potato___', '') for name in CLASSES] if CLASSES else None
    if y_true_cnn.size > 0 and y_pred_cnn.size == y_true_cnn.size:
        cnn_report_str = classification_report(y_true_cnn, y_pred_cnn, target_names=target_names_report, zero_division=0)
        print(cnn_report_str)
        cnn_report = classification_report(y_true_cnn, y_pred_cnn, target_names=target_names_report, output_dict=True, zero_division=0)
    else:
        print("Could not generate CNN classification report (empty or mismatched true/predicted labels).")

else:
    print("Test set is empty or invalid, skipping CNN evaluation and report generation.")


# --- 6. Príprava dát pre Tradičné ML Modely ---
print("\n--- Príprava dát pre tradičné ML modely... ---")

def extract_and_flatten(dataset):
    images = []
    labels = []
    if dataset is None or tf.data.experimental.cardinality(dataset).numpy() == 0:
        print("Warning: Input dataset is None or empty. Returning empty arrays.")
        # Adjust the reshape dimensions based on the potentially updated IMG_SIZE
        return np.array([]).reshape(0, IMG_SIZE*IMG_SIZE*3), np.array([])

    for batch_images, batch_labels in dataset:
        batch_images_rescaled = batch_images.numpy().astype(np.float32) / 255.0
        batch_images_flattened = batch_images_rescaled.reshape(batch_images_rescaled.shape[0], -1)
        images.append(batch_images_flattened)
        labels.append(batch_labels.numpy())

    if not images:
         # Adjust the reshape dimensions based on the potentially updated IMG_SIZE
         return np.array([]).reshape(0, IMG_SIZE*IMG_SIZE*3), np.array([])

    images = np.concatenate(images, axis=0)
    labels = np.concatenate(labels, axis=0)
    print(f"Extracted Data shape: {images.shape}, Labels shape: {labels.shape}")
    return images, labels

X_train_flat, y_train = extract_and_flatten(train_dataset)
X_test_flat, y_test = extract_and_flatten(test_dataset)


# --- 7. Tréning a Evaluácia Tradičných ML Modelov ---
traditional_results = {}
traditional_reports = {}
training_times = {}

if X_test_flat.shape[0] == 0 or y_test.shape[0] == 0:
    print("Test set is empty after extraction. Skipping traditional model training and evaluation.")
else:
    print(f"\nTréningové dáta (sploštené): {X_train_flat.shape}")
    print(f"Testovacie dáta (sploštené): {X_test_flat.shape}")
    print("\n--- Tréning a Evaluácia Tradičných ML Modelov ---")

    models = {
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'SVM': SVC(kernel='linear', probability=True, C=0.1, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
        'Logistic Regression': LogisticRegression(max_iter=200, random_state=42, n_jobs=-1),
        'Naive Bayes': GaussianNB(),
        'LDA': LinearDiscriminantAnalysis()
    }

    target_names_report = [name.replace('Potato___', '') for name in CLASSES] if CLASSES else None

    for name, model in models.items():
        print(f"\nTrénuje sa model: {name}")
        start_time = time.time()

        if name in ['KNN', 'SVM', 'Logistic Regression', 'LDA']:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])
        else:
            pipeline = Pipeline([('model', model)])

        try:
            pipeline.fit(X_train_flat, y_train)
            train_time = time.time() - start_time
            training_times[name] = train_time
            print(f"Tréning dokončený za {train_time:.2f} sekúnd.")

            y_pred = pipeline.predict(X_test_flat)
            accuracy = accuracy_score(y_test, y_pred)
            traditional_results[name] = accuracy
            print(f"Test Accuracy ({name}): {accuracy:.4f}")

            print(f"\n{name} Classification Report:")
            report_str = classification_report(y_test, y_pred, target_names=target_names_report, zero_division=0)
            print(report_str)
            traditional_reports[name] = classification_report(y_test, y_pred, target_names=target_names_report, output_dict=True, zero_division=0)

        except Exception as e:
            print(f"Error training or evaluating model {name}: {e}")
            traditional_results[name] = 0.0
            traditional_reports[name] = None
            training_times[name] = None


# --- 8. Porovnanie Celkovej Presnosti ---
print("\n--- Porovnanie Celkovej Výkonnosti Modelov (Accuracy) ---")

all_results = traditional_results.copy()
if cnn_accuracy > 0:
    all_results['CNN'] = cnn_accuracy

sorted_results = dict(sorted(all_results.items(), key=lambda item: item[1], reverse=True))

if not sorted_results:
    print("No model results available to display for overall accuracy comparison.")
else:
    names = list(sorted_results.keys())
    accuracies = list(sorted_results.values())

    plt.figure(figsize=(12, 6))
    bars = plt.bar(names, accuracies, color=['salmon' if name == 'CNN' else 'skyblue' for name in names])
    plt.xlabel("Model")
    plt.ylabel("Overall Test Accuracy")
    plt.title("Porovnanie Celkovej Presnosti Modelov na Testovacej Sade")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=45, ha='right')

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.3f}', va='bottom', ha='center')

    plt.tight_layout()
    plt.show()

    print("\n--- Súhrn Celkovej Presnosti a Času Tréningu ---")
    print("| Model               | Test Accuracy | Training Time (s) |")
    print("|---------------------|---------------|-------------------|")
    cnn_time_str = f"{cnn_training_time:>17.2f}" if 'cnn_training_time' in locals() else "N/A".rjust(19)
    if 'CNN' in sorted_results:
        print(f"| CNN                 | {sorted_results['CNN']:.4f}       | {cnn_time_str} |")
    for name, acc in sorted_results.items():
        if name != 'CNN':
            time_val = training_times.get(name)
            time_str = f"{time_val:>17.2f}" if time_val is not None else "N/A".rjust(19)
            print(f"| {name:<19} | {acc:.4f}       | {time_str} |")

# --- 9. Vizualizácia Výkonnosti po Triedach (F1-Score) ---
print("\n--- Porovnanie Výkonnosti Modelov po Triedach (F1-Score) ---")

f1_model_names = []
f1_class_scores = {class_name: [] for class_name in target_names_report} if target_names_report else {}

if cnn_report and target_names_report:
    f1_model_names.append('CNN')
    for class_name in target_names_report:
        if class_name in cnn_report and 'f1-score' in cnn_report[class_name]:
             f1_class_scores[class_name].append(cnn_report[class_name]['f1-score'])
        else:
             f1_class_scores[class_name].append(0.0)

for name, report in traditional_reports.items():
    if report and target_names_report:
        f1_model_names.append(name)
        for class_name in target_names_report:
             if class_name in report and 'f1-score' in report[class_name]:
                 f1_class_scores[class_name].append(report[class_name]['f1-score'])
             else:
                 f1_class_scores[class_name].append(0.0)

if not f1_model_names or not f1_class_scores or not target_names_report:
    print("Nemôžem vygenerovať graf F1-Score po triedach - chýbajú dáta alebo názvy tried.")
else:
    x_f1 = np.arange(len(f1_model_names))
    width_f1 = 0.25
    num_classes_plot_f1 = len(target_names_report)
    total_width_f1 = width_f1 * num_classes_plot_f1
    start_pos_f1 = - (total_width_f1 / 2) + (width_f1 / 2)

    fig_f1, ax_f1 = plt.subplots(figsize=(14, 7))

    colors_f1 = plt.cm.get_cmap('viridis', num_classes_plot_f1)

    for i, (class_name, scores) in enumerate(f1_class_scores.items()):
        offset = start_pos_f1 + i * width_f1
        if len(scores) == len(f1_model_names):
            rects = ax_f1.bar(x_f1 + offset, scores, width_f1, label=class_name, color=colors_f1(i / num_classes_plot_f1))
            ax_f1.bar_label(rects, padding=3, fmt='%.2f', fontsize=8)
        else:
             print(f"Warning (F1 Plot): Skipping class '{class_name}' due to mismatched data length (expected {len(f1_model_names)}, got {len(scores)}).")

    ax_f1.set_ylabel('F1-Score')
    ax_f1.set_title('Porovnanie F1-Score Modelov pre Každú Triedu')
    ax_f1.set_xticks(x_f1, f1_model_names, rotation=45, ha='right')
    ax_f1.legend(loc='upper left', ncols=1)
    ax_f1.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.show()

# --- 10. Vizualizácia Výkonnosti po Triedach (Recall/Accuracy Proxy) ---
print("\n--- Porovnanie Výkonnosti Modelov po Triedach (Recall ako Proxy Presnosti) ---")

recall_model_names = []
recall_class_scores = {class_name: [] for class_name in target_names_report} if target_names_report else {}

if cnn_report and target_names_report:
    recall_model_names.append('CNN')
    for class_name in target_names_report:
        if class_name in cnn_report and 'recall' in cnn_report[class_name]:
             recall_class_scores[class_name].append(cnn_report[class_name]['recall'])
        else:
             recall_class_scores[class_name].append(0.0)

for name, report in traditional_reports.items():
    if report and target_names_report:
        recall_model_names.append(name)
        for class_name in target_names_report:
             if class_name in report and 'recall' in report[class_name]:
                 recall_class_scores[class_name].append(report[class_name]['recall'])
             else:
                 recall_class_scores[class_name].append(0.0)

if not recall_model_names or not recall_class_scores or not target_names_report:
    print("Nemôžem vygenerovať graf Recall po triedach - chýbajú dáta alebo názvy tried.")
else:
    x_recall = np.arange(len(recall_model_names))
    width_recall = 0.25
    num_classes_plot_recall = len(target_names_report)
    total_width_recall = width_recall * num_classes_plot_recall
    start_pos_recall = - (total_width_recall / 2) + (width_recall / 2)

    fig_recall, ax_recall = plt.subplots(figsize=(14, 7))

    colors_recall = plt.cm.get_cmap('plasma', num_classes_plot_recall)

    for i, (class_name, scores) in enumerate(recall_class_scores.items()):
        offset = start_pos_recall + i * width_recall
        if len(scores) == len(recall_model_names):
            rects = ax_recall.bar(x_recall + offset, scores, width_recall, label=class_name, color=colors_recall(i / num_classes_plot_recall))
            ax_recall.bar_label(rects, padding=3, fmt='%.2f', fontsize=8)
        else:
             print(f"Warning (Recall Plot): Skipping class '{class_name}' due to mismatched data length (expected {len(recall_model_names)}, got {len(scores)}).")

    ax_recall.set_ylabel('Recall (Proxy Presnosti pre Triedu)')
    ax_recall.set_title('Porovnanie Recall Modelov pre Každú Triedu')
    ax_recall.set_xticks(x_recall, recall_model_names, rotation=45, ha='right')
    ax_recall.legend(loc='upper left', ncols=1)
    ax_recall.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.show()


# --- 11. Vizualizácia histórie tréningu CNN ---
print("\n--- Vizualizácia histórie tréningu CNN ---")
if 'history' in locals() and history is not None:
    acc = history.history.get('accuracy')
    val_acc = history.history.get('val_accuracy')
    loss = history.history.get('loss')
    val_loss = history.history.get('val_loss')

    if acc and loss:
        epochs_range = range(len(acc))

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        if val_acc:
             plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('CNN Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        if val_loss:
            plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('CNN Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plt.tight_layout()
        plt.show()
    else:
        print("Skipping CNN history plot - required metrics ('accuracy', 'loss') not found in history object.")
else:
     print("Skipping CNN history plot - 'history' object not found (CNN training might have failed).")


# --- 12. Uloženie CNN modelu ---
print("\n--- Ukladanie CNN modelu ---")
try:
    cnn_model.save('potato_disease_classifier_cnn_model.h5')
    print("CNN model úspešne uložený ako 'potato_disease_classifier_cnn_model.h5'")
except Exception as e:
    print(f"Error saving CNN model: {e}")

