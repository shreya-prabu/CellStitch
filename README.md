CellStitch Application

Overview:
CellStitch is a web-based application developed using Dash and Python 
that leverages the Cellpose deep learning model for cellular segmentation. 
The app allows users to upload images, specify custom image dimensions and 
flow thresholds, and run the Cellpose algorithm to generate segmentation masks, 
which can then be downloaded.

Features:
Image Upload: Supports uploading of TIFF and other common image formats.
Customizable Parameters: Users can input the desired image shape (X, Y, Z dimensions) and specify a flow threshold for the Cellpose algorithm.
Cellpose Integration: Utilizes the powerful Cellpose model for accurate and efficient cellular segmentation.
Real-time Feedback: Provides real-time feedback during image processing to keep users informed of progress.
Downloadable Results: Segmentation masks are generated and made available for download.

Ensure you have the following installed:
1. Python 3.x
2. pip (Python package installer)

Installation:
1. Clone the repository:
   git clone https://github.com/shreya-prabu/cellstitch.git
   cd cellstitch
2. Install the required python packages:
   pip install -r requirements.txt
3. Verify that the required packages have been installed:
   pip install dash cellpose torch tifffile pillow numpy flask

File Structure:
1. app.py: The main application file.
2. requirements.txt: Contains the list of required Python packages.
3. README.md: This file, which provides an overview and instructions.

Usage:
1. Run the application:
   python app.py
2. Access the app:
   Open your web browser and go to http://127.0.0.1:8050/.
3. Using the app:
   Upload Image: Drag and drop or select the image file you want to process.
   Specify Image Shape: Enter the desired image dimensions in the X, Y, Z format (e.g., 20,224,224).
   Set Flow Threshold: Enter the flow threshold value (between 0.0 and 20.0, default is 1.0).
   Run the Algorithm: Click the "Run CellStitch" button to start processing.
   Download Result: Once processing is complete, click the provided link to download the segmentation masks.
