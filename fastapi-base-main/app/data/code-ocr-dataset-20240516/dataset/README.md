## Documentation for Handwritten Code OCR Dataset

### Overview

This dataset consists of 55 images of handwritten code, their corresponding ground truth text, and OCR outputs from four different services: Google Cloud Vision (GCV), AWS Textract, Microsoft Azure OCR, and Mathpix.

### Contents

The dataset is provided in a ZIP file with the following structure:

```
dataset/
│
├── images/
│   ├── 0.jpg
│   ├── 1.jpg
│   ├── 2.jpg
│   ├── ...
│   └── 54.jpg
│
├── rawdata.csv
└── apipayload.json
```

- **images/**: Contains 55 images of handwritten code, named sequentially from `0.jpg` to `54.jpg`.
- **rawdata.csv**: A CSV file containing the ground truth text for each image and the OCR outputs from Google Cloud Vision, AWS Textract, Microsoft Azure OCR, and Mathpix, along with the edit distances between the ground truth and each OCR output.
- **apipayload.json**: A JSON file containing the raw API payloads returned by each OCR service for every image.

### CSV File Structure (`rawdata.csv`)

The CSV file contains the following columns:

- **Ground Truth**: The actual handwritten code transcribed manually.
- **GCV**: OCR output from Google Cloud Vision.
- **ED GCV**: Edit distance between the ground truth and the Google Cloud Vision output.
- **AWS**: OCR output from AWS Textract.
- **ED AWS**: Edit distance between the ground truth and the AWS Textract output.
- **Azure**: OCR output from Microsoft Azure OCR.
- **ED Azure**: Edit distance between the ground truth and the Azure OCR output.
- **MP**: OCR output from Mathpix.
- **ED MP**: Edit distance between the ground truth and the Mathpix output.

### JSON File Structure (`apipayload.json`)

The JSON file contains an array of objects, each corresponding to one image in the dataset. Each object has the following structure:

```json
{
  "image_id": 0,
  "GCV": { ... },
  "AWS": { ... },
  "Azure": { ... },
  "MP": { ... }
}
```

- **image_id**: The ID of the image (0 to 54).
- **GCV**: The raw API payload returned by Google Cloud Vision.
- **AWS**: The raw API payload returned by AWS Textract.
- **Azure**: The raw API payload returned by Microsoft Azure OCR.
- **MP**: The raw API payload returned by Mathpix.

### Dataset Structure

The dataset consists of two parts:

| **Dataset**                 | **Num Photos** | **Correct** | **Annotated** |
|-----------------------------|----------------|-------------|---------------|
| Correct Student Dataset     | 44             | 100%        | ✓             |
| Logical Error Dataset       | 11             | 0%          | ✓             |

- **Correct Student Dataset**: Contains 44 images of handwritten code that are logically correct. Each image is fully annotated with the expected OCR output.
- **Logical Error Dataset**: Contains 11 images of handwritten code that contain logical errors. Each image is annotated with the expected OCR output.

#### Details of Logical Error Dataset

The Logical Error Dataset includes specific descriptions of the errors present in the handwritten code. This information can be useful for understanding the types of mistakes made and for training OCR systems to handle such errors effectively.

| **Image ID** | **Description of Error** |
|--------------|---------------------------|
| 29           | In the `identify leap year` function, the student used incorrect logic: `if (Year % 4 == 0) or (Year % 100 == 0) or (Year % 400 == 0)`. Correct logic: `if (Year % 4 == 0) and (Year % 100 != 0) or (Year % 400 == 0)`. |
| 45           | Initialized the variable `max = 0` inside of the for loop, causing it to reset at every iteration. |
| 46           | Off by one error in string indexing: should be `str[len(str)-i-1]` instead of `str[len(str)-i]`. |
| 47           | Recursive factorial function lacks a base case and does not handle `n = 0`. |
| 48           | Fibonacci sequence indexing error: should be `sequence[i] + sequence[i+1]`, not `sequence[i+1] + sequence[i+2]`. Incorrect loop condition: should be `while len(sequence) < n` instead of `<= n`. |
| 49           | KeyError raised if `freq[item]` is called and `item` does not exist in the dictionary. Solution: check if `item` is in the dictionary before incrementing. |
| 50           | The range in the second loop should start from `i+1` to avoid repeating elements. |
| 51           | Indentation error in assigning `final_list` and unnecessary variable `final_loop` resetting inside the loop. |
| 52           | Incorrect operator: used division (`/`) instead of modulo (`%`) for checking even or odd. |
| 53           | Non-alphabetic characters are incorrectly counted as lower case due to an overly broad `else` condition. |
| 54           | Error in multiplication logic: multiplies every number by 0. Initialize `total` as `1` for correct operation. |

### Usage Instructions

1. **Loading the Images**: The images are stored in the `images/` directory. Each image file is named according to its corresponding ID in the CSV and JSON files.
   
   ```python
   from PIL import Image
   
   image_path = 'images/0.jpg'
   image = Image.open(image_path)
   image.show()
   ```

2. **Loading the CSV Data**: You can use pandas to load and explore the CSV data.

   ```python
   import pandas as pd

   csv_path = 'rawdata.csv'
   data = pd.read_csv(csv_path)
   print(data.head())
   ```

3. **Loading the JSON Data**: Use the json module to load the JSON file and access the raw API payloads.

   ```python
   import json

   with open('apipayload.json', 'r') as file:
       api_payloads = json.load(file)

   print(api_payloads[0])
   ```

4. **Filtering Correct and Erroneous Data**: You can filter the dataset to work with only the correct data or only the erroneous data. Below are code snippets to help you do this easily.

   ```python
   # Load the CSV data
   import pandas as pd

   csv_path = 'rawdata.csv'
   data = pd.read_csv(csv_path)

   # Image IDs for correct and erroneous datasets
   correct_ids = list(range(29)) + list(range(30, 45))
   error_ids = [29] + list(range(45, 55))

   # Filtering the correct data
   correct_data = data.iloc[correct_ids]

   # Filtering the erroneous data
   error_data = data.iloc[error_ids]

   # Example: Displaying the first few rows of each dataset
   print("Correct Data:")
   print(correct_data.head())
   
   print("\nErroneous Data:")
   print(error_data.head())
   ```

5. **Comparing OCR Outputs**: You can compare the OCR outputs from different services using the edit distances provided in the CSV file. A lower edit distance indicates a closer match to the ground truth.

   ```python
   sample_data = data.iloc[0]
   print("Ground Truth:", sample_data['Ground Truth'])
   print("GCV Output:", sample_data['GCV'])
   print("AWS Output:", sample_data['AWS'])
   print("Azure Output:", sample_data['Azure'])
   print("MP Output:", sample_data['MP'])
   ```

### Notes

- The edit distances are calculated using the Levenshtein distance metric, which measures the minimum number of single-character edits required to transform one string into another.
- The raw API payloads in the JSON file can be large and may contain additional metadata and details that are not included in the CSV file.
- This dataset can be useful for benchmarking OCR systems, studying the performance of OCR on handwritten code, and developing improved OCR algorithms.

### Citation

Please cite the paper if you use the dataset in your research.

<!-- ```
@dataset{your_name_2024_handwritten_code_ocr,
  author       = {Your Name},
  title        = {Handwritten Code OCR Dataset},
  year         = 2024,
  publisher    = {Your Institution},
  url          = {link_to_dataset}
}
``` -->
