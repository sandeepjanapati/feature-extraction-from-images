# Feature Extraction from Images

This project addresses the challenge of automatically extracting structured product information (such as weight, volume, voltage, etc.) directly from product images. In e-commerce and other domains, product descriptions are often incomplete or missing crucial details. This project offers a practical solution to enrich product data by leveraging the information contained within images, leading to improved product catalogs, enhanced search functionality, and more accurate product comparisons. This project was inspired from the Amazon ML Challenge 2024.


## Problem Statement

Many online product listings rely heavily on images, yet these images often contain valuable information not captured in the accompanying textual descriptions.  This project aims to bridge this gap by automatically extracting structured entity values (e.g., "100 grams", "220 volts") from product images.

## Data Description

The project utilizes a dataset comprising:

*   **Training Data:** 2,63,859 images with corresponding structured labels, including:
    *   `image_link`: URL of the product image.
    *   `group_id`: Product category code.
    *   `entity_name`: Name of the feature being extracted (e.g., "item_weight").
    *   `entity_value`:  Ground truth value of the feature (e.g., "100 grams").
*   **Test Data:** 1,31,187 images with the following information:
    *   `index`: Unique identifier for each image.
    *   `image_link`: URL of the product image.
    *   `group_id`: Product category code.
    *   `entity_name`:  Name of the feature being extracted.


## Solution Overview

The implemented solution uses a combination of image processing, Optical Character Recognition (OCR), and Named Entity Recognition (NER) to extract product features from images. The key steps include:

1.  **Image Preprocessing:** Images are preprocessed to enhance OCR accuracy. Techniques employed include noise reduction, contrast enhancement, and thresholding to mitigate variations in image quality and lighting conditions.

2.  **OCR with Tesseract:** Tesseract OCR is utilized to extract text from the preprocessed images. This extracted text serves as the input to the NER model.

3.  **Named Entity Recognition with spaCy:** A custom NER model, trained using spaCy, identifies and extracts entity values and their associated units from the OCR text. The training data for the NER model is generated automatically by matching structured labels with the OCR output.  Fuzzy string matching techniques are used to address potential discrepancies between the labels and the extracted OCR text.

4.  **Unit Handling and Conversion:** The `pint` library is employed for unit standardization and conversion. Extracted values are transformed to a predefined set of allowed units (defined in `constants.py`), ensuring consistency and facilitating accurate comparisons.

## Implementation Details

*   **Programming Language:** Python
*   **Libraries:** pandas, scikit-learn, spaCy, pytesseract, Pillow, requests, opencv-python, pint
*   **NER Model:** Custom spaCy NER model trained using automatically generated annotations.
*   **Error Handling:** Includes error handling for invalid or inconsistent values and download issues.

## Evaluation

The model's performance is rigorously assessed using 5-fold cross-validation. The primary evaluation metric is the F1-score, providing a balance between precision and recall.


## File Structure

feature-extraction-from-images/ 
├── README.md # This file 
├── main.py # Main Python script 
├── constants.py # Allowed units and entity mappings 
├── utils.py # Utility functions (e.g., image download, preprocessing) 
├── train.csv # Training data 
├── test.csv # Test data
├── downloaded_images/ # Downloaded images



## Dependencies

*   pandas
*   scikit-learn
*   spacy
*   pytesseract
*   Pillow
*   requests
*   opencv-python
*   pint

Install dependencies using: `pip install -r requirements.txt` (Create a `requirements.txt` file listing all dependencies)


## Getting Started

1.  Clone the repository: `git clone https://github.com/sandeepjanapati/feature-extraction-from-images.git`
2.  Install project dependencies (see Dependencies section).
3.  Install Tesseract OCR and configure `pytesseract` by following the platform-specific instructions in the `pytesseract` documentation.
4.  Run the script: `python main.py`


## Challenges

*   **OCR Inaccuracy:**  OCR performance can be significantly affected by variations in image quality, fonts, and text orientations. Inaccurate OCR output directly impacts the downstream NER model's performance.
*   **Aligning Ground Truth with OCR:** Precisely aligning the ground truth entity values with potentially noisy and differently formatted OCR output is challenging.  Fuzzy matching techniques help, but can be imperfect.

## Future Work

*   **Object Detection for ROI Extraction:**  Object detection models (like Faster R-CNN or YOLO) can be integrated to identify and isolate regions of interest (ROIs) within the images, focusing OCR efforts on the relevant parts and filtering out irrelevant background noise or text.
*   **Advanced OCR Preprocessing:**  Exploring more advanced image preprocessing and OCR techniques (e.g., image deskewing, binarization methods) could further improve the accuracy of text extraction.
*   **Ensemble Methods:**  Combining predictions from multiple NER models or employing ensemble techniques can improve robustness and increase the overall F1-score.

## License

MIT License