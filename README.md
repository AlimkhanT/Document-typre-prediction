# Image Classification Flask Application

This Flask-based web application allows users to upload `.jpg` image files and classify them into one of several predefined document categories using two machine learning models. The application uses ResNet-50-based models for document classification and displays the results to the user.

## Features
- Upload `.jpg` images through a simple form.
- Get predictions from two models: baseline model and enhanced model.
- Display results dynamically on the webpage.
- Built with Flask and PyTorch.

## Installation

### Prerequisites

Make sure you have the following installed:

- Python 3.7+ (for compatibility with libraries)
- `pip` (Python package manager)

### Steps to Run the Application

1. Clone the repository or download the project files to your local machine.

    ```bash
    git clone <repository_url>
    cd capstone-main
    ```

2. Create and activate a virtual environment (optional but recommended).

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install required dependencies.

    ```bash
    pip install -r requirements.txt
    ```

4. Download the pretrained models (`baseline_model.pth` and `enhanced_model.pth`) and place them in the `backend` directory. These models are used for classification.

5. Run the Flask application.

    ```bash
    python backend/app.py
    ```

6. Open your browser and navigate to `http://127.0.0.1:5000` to interact with the app.

## Usage

1. Once the app is running, go to the homepage (`http://127.0.0.1:5000`).
2. Upload a `.jpg` image using the file upload form.
3. After the image is uploaded, the app will display the classification results from both the baseline and enhanced models.
4. The app predicts the document category (e.g., "scientific_report", "resume", etc.).

## File Structure

/backend
    app.py              # Flask app with logic to handle image uploads and model predictions
    baseline_model.pth  # Pretrained baseline model (ResNet-50)
    enhanced_model.pth  # Pretrained enhanced model (ResNet-50 with additional layers)
/templates
    index.html          # HTML for the main page
/static
    styles.css          # CSS for styling the page
requirements.txt        # List of Python dependencies
README.md              # This file


## Models

- **Baseline Model**: A standard ResNet-50 model trained on document images.
- **Enhanced Model**: A modified ResNet-50 model with additional layers like Batch Normalization and Dropout for improved performance.

## Requirements

- Flask 2.x+
- torch 1.8.0+
- torchvision 0.9.0+
- Pillow 8.x+
- (Additional libraries will be listed in the `requirements.txt` file)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- ResNet-50 architecture is provided by PyTorch.
- The models were trained on document classification tasks.
