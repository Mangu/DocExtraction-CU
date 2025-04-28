# ISDA Document Parsing

## Overview
This repository contains the `Extraction` Jupyter Notebooks which is used for document extraction. The notebook leverages Python code for reading documents, extracting relevant information, and processing the extracted data for further analysis or usage.

## Notebooks

### Content Understanding Extraction
- **Notebook**: [content-understanding-extraction.ipynb](https://github.com/Mangu/DocExtraction-CU/blob/main/content-understanding-extraction.ipynb)
- **Description**: This notebook involves the creation of a custom analyzer using agreement schemas. It details the setup of Azure AI Services configurations and demonstrates how to extract specific information such as parties and collateral details from ISDA agreements.

### Document Extraction with Azure AI Document Intelligence and Azure OpenAI GPT-4o (Text Only)
- **Notebook**: [document-intellegence-extraction-gpt.ipynb](https://github.com/Mangu/DocExtraction-CU/blob/main/document-extraction-gpt.ipynb)
- **Description**: This notebook demonstrates how to extract structured data from documents using Azure AI Document Intelligence and Azure OpenAI GPT models. The process involves analyzing a document to extract its structure as Markdown, constructing prompts for extracting structured data, and using the GPT-4o model to generate structured outputs.
  
### Usage
1. **Clone the Repository:**
    ```sh
    git clone https://github.com/Mangu/DocExtraction-CU.git
    cd DocExtraction-CU
    ```

2. **Install Dependencies:**
    Ensure you have the necessary Python packages installed. You can install the required packages using:
    ```sh
    pip install -r requirements.txt
    ```

3. **Run the Notebook:**
    Open the `Extraction.ipynb` notebook in Jupyter Notebook or Jupyter Lab:
    ```sh
    jupyter notebook Extraction.ipynb
    ```

4. **Execute the Notebook:**
    Follow the instructions within the notebook to execute the code cells step by step. 

### Contributions
Contributions to improve the document extraction techniques or extend the functionalities are welcome. Please follow the standard GitHub contribution guidelines.

### License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

### Contact
For any questions or issues, please open an issue in the repository or contact the maintainers.
