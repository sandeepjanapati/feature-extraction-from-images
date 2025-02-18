import os
import re
import pandas as pd
import numpy as np
from PIL import Image
import pytesseract
import random
import spacy
from spacy.training import offsets_to_biluo_tags # Corrected import
from spacy.training.example import Example
from sklearn.model_selection import KFold
import pint
import cv2
from tqdm import tqdm
import requests
from io import BytesIO

# Configure pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Replace with your actual path if different

# --- 1. Constants ---
allowed_units = {
    "gram", "kilogram", "milligram", "g", "kg", "mg",
    "pound", "lb", "ounce", "oz",
    "liter", "litre", "l", "milliliter", "ml",
    "fluid ounce", "fl oz", "gallon", "gal",
    "volt", "v", "watt", "w",
    "centimetre", "foot", "inch", "metre", "millimetre", "yard", "centimeter",
    "kilovolt", "millivolt", "kilowatt",
    "centilitre", "cubic foot", "cubic inch", "cup", "decilitre",
    "imperial gallon", "microlitre", "pint", "quart", "microgram", "ton", "cubic centimetre", "cubic centimeter"

}


entity_unit_map = {
    'width': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard', "centimeter"},
    'depth': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard', "centimeter"},
    'height': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard', "centimeter"},
    'item_weight': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
    'maximum_weight_recommendation': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
    'voltage': {'kilovolt', 'millivolt', 'volt'},
    'wattage': {'kilowatt', 'watt'},
    'item_volume': {'centilitre', 'cubic foot', 'cubic inch', 'cup', 'decilitre', 'fluid ounce', 'gallon',
                   'imperial gallon', 'litre', 'microlitre', 'millilitre', 'pint', 'quart', 'cubic centimetre', 'cubic centimeter'}
}

# --- 2. Utility Functions ---
def parse_string(s):
    s_stripped = "" if s is None or str(s) == 'nan' else s.strip()
    if not s_stripped:
        return None, None
    pattern = re.compile(r'^-?\d+(\.\d+)?\s+[a-zA-Z\s]+$')
    if not pattern.match(s_stripped):
        return None, None
    parts = s_stripped.split(maxsplit=1)
    try:
        number = float(parts[0])
    except ValueError:
        return None, None
    unit = parts[1]

    if unit.replace('ter', 'tre') in allowed_units:
        unit = unit.replace('ter', 'tre')
    elif unit.replace('feet', 'foot') in allowed_units:
        unit = unit.replace('feet', 'foot')

    if unit not in allowed_units:
        return None, None
    return number, unit


def download_image(url):
    try:
        response = requests.get(url, stream=True, timeout=5)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image from {url}: {e}")
        return None


def enhanced_ocr_preprocessing(image):
    if image is None:
        return None

    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

    # Noise Reduction
    gray = cv2.medianBlur(gray, 3)

    # Contrast Enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return thresh


def perform_ocr(image):
    if image is None:
        return ""
    return pytesseract.image_to_string(image, lang='eng', config='--psm 6')


def evaluate_prediction(row):
    if not row['entity_value'] and not row['predicted_value']:
        return 'TN'
    elif row['entity_value'] and row['predicted_value']:
        return 'TP' if row['entity_value'] == row['predicted_value'] else 'FP'
    elif row['entity_value'] and not row['predicted_value']:
        return 'FN'
    else:
        return 'FP'

# --- Configuration ---
DEVICE = 'cpu'
DOWNLOAD_DIR = "downloaded_images"
BATCH_SIZE = 8  # Reduced batch size
NUM_EPOCHS = 20  # Increased number of epochs
LEARNING_RATE = 0.005


# --- 3. Data Cleaning and Preprocessing ---
try:
    df = pd.read_csv("train.csv")
except FileNotFoundError:
    print("Error: train.csv not found in the current directory.")
    exit()

# Initialize spaCy NLP model here so it can be used in the function.
nlp = spacy.blank("en")


def clean_entity_value(value):
    if pd.isna(value):
        return None, None

    try:
        num, unit = parse_string(value)
        if num is None:
            return None, None
        if unit not in allowed_units:
            return None, None
        return num, unit
    except (ValueError, AttributeError):
        return None, None


cleaned_train = []
for idx, row in tqdm(df.iterrows(), desc="Cleaning Training Data"):
    num, unit = clean_entity_value(row['entity_value'])
    if num is not None:
        cleaned_train.append({
            'index': idx,
            'image_link': row['image_link'],
            'group_id': row['group_id'],
            'entity_name': row['entity_name'],
            'entity_value_num': num,
            'entity_value_unit': unit,
            'entity_value': row['entity_value']
        })

cleaned_train_df = pd.DataFrame(cleaned_train)


# --- 4. Image Download and OCR ---
if not os.path.exists(DOWNLOAD_DIR):
    os.makedirs(DOWNLOAD_DIR)

cleaned_train_df['ocr_text'] = ""

for index, row in tqdm(cleaned_train_df.iterrows(), total=len(cleaned_train_df), desc="Performing OCR"):
    img_path = os.path.join(DOWNLOAD_DIR, os.path.basename(row['image_link']))
    if not os.path.exists(img_path):
        image = download_image(row['image_link'])
        if image is None:
            cleaned_train_df.loc[index, 'ocr_text'] = ""
            continue
        try:
            image.save(img_path)
        except OSError as e:
            print(f"Could not save {row['image_link']} image: {e}")
            continue


    try:
        image = Image.open(img_path)
        preprocessed_image = enhanced_ocr_preprocessing(image)
        if preprocessed_image is None:
            print(f"Preprocessing failed for image {row['image_link']}")
            cleaned_train_df.loc[index, 'ocr_text'] = ""
            continue

        text = perform_ocr(preprocessed_image)
        cleaned_train_df.loc[index, 'ocr_text'] = text
    except Exception as e:
        print(f"Error processing image {row['image_link']}: {e}")
        cleaned_train_df.loc[index, 'ocr_text'] = ""


# --- 5. Generate NER Training Data ---
ner_training_data = []

for _, row in tqdm(cleaned_train_df.iterrows(), total=len(cleaned_train_df), desc="Generating NER Training Data"):
    text = row['ocr_text']
    entity_value = row['entity_value']

    try:
        # Find all occurrences of entity_value in text (case-sensitive)
        # Use a lookahead assertion to handle overlapping matches
        starts = [m.start() for m in re.finditer(f"(?={re.escape(entity_value)})", text)]


        if starts:
            entities = []
            for start in starts:
                end = start + len(entity_value)
                entities.append((start, end, "VALUE"))

                unit_match = re.search(r"(\d+\.?\d*)\s*([a-zA-Z]+(?:\s*[a-zA-Z]+)?)", entity_value)
                if unit_match:
                    unit = unit_match.group(2)

                    # Handle case sensitivity during unit lookup
                    unit_start = text.find(unit, end)  # Case-sensitive search for unit after value
                    if unit_start == -1:  # if unit not found, try lowercase versions
                      unit_start = text.lower().find(unit.lower(), end)

                    if unit_start != -1:
                        unit_end = unit_start + len(unit)
                        entities.append((unit_start, unit_end, "UNIT"))




            try:
                biluo_tags = offsets_to_biluo_tags(nlp.make_doc(text), entities)
            except Exception as e:
                print(f"Error converting to BILUO tags: {e}")
                continue



            aligned_entities = []
            start, end, label = None, None, None
            for i, tag in enumerate(biluo_tags):
                if tag.startswith("U-"):
                    aligned_entities.append((i, i + 1, tag[2:]))
                elif tag.startswith("B-"):
                    start = i
                    label = tag[2:]
                elif tag.startswith("L-"):
                    end = i + 1
                    aligned_entities.append((start, end, label))
                    start, end, label = None, None, None

            ner_training_data.append((text, {"entities": aligned_entities}))

    except Exception as e:
        print(f"Error generating training data: {e}")




# --- 6. NER Model Training ---
def train_ner_model(train_data, nlp):
    ner = nlp.add_pipe("ner")

    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):  # only train NER
      optimizer = nlp.begin_training()
      for itn in range(NUM_EPOCHS):
          random.shuffle(train_data)
          losses = {}
          for batch in spacy.util.minibatch(train_data, size=BATCH_SIZE):
              for text, annotations in batch:
                  doc = nlp.make_doc(text)
                  example = Example.from_dict(doc, annotations)
                  nlp.update([example], drop=0.5, sgd=optimizer, losses=losses)
          print("Losses", losses)
    return nlp

ner_model = train_ner_model(ner_training_data, nlp)



# --- 7. Entity Extraction with NER and Unit Handling ---
def extract_entity_value_with_ner(text, entity_name, nlp_model, ureg):
    doc = nlp_model(text)
    extracted_entities = {}
    for ent in doc.ents:
        extracted_entities[ent.label_] = ent.text

    value = extracted_entities.get("VALUE", None)
    unit = extracted_entities.get("UNIT", None)

    if value and unit:
        try:
            quantity = ureg(f"{value} {unit}")

            if entity_name == "item_weight":
                if quantity.units == ureg.gram:
                    if quantity.magnitude >= 1000:
                        quantity = quantity.to(ureg.kilogram)
                    elif quantity.magnitude < 1:
                        quantity = quantity.to(ureg.milligram)
                quantity.ito(ureg.gram)  # Standardized Unit

            elif entity_name == "item_volume":
                quantity.ito(ureg.milliliter)  # standardize on milliliter
            elif entity_name in ('width', 'depth', 'height'):
                try:
                    quantity.ito(ureg.centimetre)
                except pint.DimensionalityError as e:
                    print(f"Dimensionality Error for {entity_name}: {e}")
                    return ""
            elif entity_name in ('maximum_weight_recommendation'):
                quantity.ito(ureg.kilogram)
            elif entity_name == 'voltage':
                quantity.ito(ureg.volt)
            elif entity_name == 'wattage':
                quantity.ito(ureg.watt)
            else:
                print("Unknown entity:", entity_name)
                return ""

            return f"{quantity.magnitude:.2f} {quantity.units}"

        except pint.errors.UndefinedUnitError:
            print("Error: Undefined unit:", unit)
            return ""
        except Exception as e:
            print("Pint Error:", e)
            return ""

    return ""



# --- 8. Cross-validation and Prediction ---
def format_entity_value(entity_value):
    try:
        num, unit = parse_string(entity_value)
        if num is None or unit is None:
            return ""
        return f"{num:.2f} {unit}"
    except (ValueError, TypeError):
        return ""


kf = KFold(n_splits=5, shuffle=True, random_state=42)
f1_scores = []
ureg = pint.UnitRegistry()


def process_image_with_ner(row, nlp_model, ureg):
    try:
        image = download_image(row['image_link'])
        if image is None:
            return ""

        preprocessed_image = enhanced_ocr_preprocessing(image)
        if preprocessed_image is None:
            return ""

        text = perform_ocr(preprocessed_image)
        extracted_value = extract_entity_value_with_ner(text, row['entity_name'], nlp_model, ureg)
        return extracted_value
    except Exception as e:
        print(f"Error processing image: {row['image_link']}")
        print(f"Error message: {e}")
        return ""



if __name__ == "__main__":
    for fold, (train_index, val_index) in enumerate(kf.split(cleaned_train_df)):
        print(f"Fold {fold + 1}")

        train_fold = cleaned_train_df.iloc[train_index]
        val_fold = cleaned_train_df.iloc[val_index].copy()

        val_fold['predicted_value'] = val_fold.apply(lambda row: process_image_with_ner(row, ner_model, ureg), axis=1)
        val_fold['entity_value'] = val_fold['entity_value'].apply(format_entity_value)

        val_fold['evaluation'] = val_fold.apply(evaluate_prediction, axis=1)
        tp = (val_fold['evaluation'] == 'TP').sum()
        fp = (val_fold['evaluation'] == 'FP').sum()
        fn = (val_fold['evaluation'] == 'FN').sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        f1_scores.append(f1)

    avg_f1 = np.mean(f1_scores)
    print(f"Average F1 Score across folds: {avg_f1:.4f}")

    # --- Test Set Prediction ---
    try:
        test_df = pd.read_csv("test.csv")
    except FileNotFoundError:
        print("Error: test.csv not found in the current directory.")
        exit()

    test_df['predicted_value'] = test_df.apply(lambda row: process_image_with_ner(row, ner_model, ureg), axis=1)
    test_predictions = test_df[['index', 'predicted_value']].rename(columns={'predicted_value': 'prediction'})
    test_predictions.to_csv('test_predictions.csv', index=False)