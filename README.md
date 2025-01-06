# Smart Helmet with Depth Map Visualization

This project aims to provide depth estimation and vehicle detection using monocular images and YOLO model for bounding box detection. The system computes region depths and determines safe and warning zones for improved vehicle safety.

## Requirements

### System Requirements

- Python 3.8 or above
- CUDA-capable GPU (for faster performance, optional)

### Python Libraries

The necessary libraries can be installed using the `requirements.txt` file.

## Setup Guide

### 1. Clone the repository

```bash
git clone https://github.com/QuantuM410/smart-helmet-2
cd smart-helmet-2
```

### 2. Create and activate a virtual environment

If you're using `venv2`, you can skip this step. Otherwise, create a new virtual environment:

```bash
python -m venv venv2
source venv2/bin/activate  # For Linux/macOS
venv2\Scripts\activate     # For Windows
```

### 3. Install dependencies

Run the following command to install all the required libraries:

```bash
pip install -r requirements.txt
```

### 4. Download YOLOv8 Model Weights

Download the `yolov8n.pt` model weights file and place it in the root directory of the project (the file should already be present in your repository if you're using the structure shared).

### 5. Set up the environment

Ensure that you have the correct versions of any required software, such as CUDA and cuDNN, if using a GPU.

### 6. Running the Application

After setting up the environment, you can start the application by running the following command:

```bash
python app.py
```

This will launch the application, which will begin processing video input and visualize the depth estimation and bounding box outputs.

## File Structure

- `app.py`: The main file that runs the Flask application.
- `models/`: Contains the `yolo.py` for object detection and `depth.py` for depth estimation.
- `static/`: Directory for storing images and other media files used in the app.
- `templates/`: Contains the `index.html` template for the web interface.
- `utils/`: Contains utility scripts like `proximity_analyzer.py` and `video_feed.py`.
- `venv2/`: Your virtual environment directory (not tracked by git).
- `yolov8n.pt`: The pretrained YOLOv8 model weights.

## Troubleshooting

- Ensure that your GPU drivers and CUDA installation are compatible if you're using a GPU.
- If you encounter issues with missing dependencies, verify that all required libraries are installed using `pip list`.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
