
About the project


This project explores how faces can be represented and analyzed using both classical statistical methods and modern deep learning techniques.
The work begins with Eigenfaces, one of the earliest approaches in face recognition. Eigenfaces treat face images as high-dimensional data and look for a compact representation that preserves the most meaningful variations across individuals. This approach allows us to study what information is truly essential for representing faces.
We also explore a deep learning–based pipeline using FaceNet. By comparing these two approaches, the project highlights the comparision of face recognition techniques that use variance-based linear models and identity-driven nonlinear embeddings pipeline.


The data for faces is in att_faces directory and we also use non face images to make the classification more robust. Images without faces is in non_faces directory.


```text
├── att_faces/
│   ├── s1/
│   ├── s2/
│   └── ...
├── non_faces/
│   ├── 0001.png
│   ├── 0002.png
│   └── ...
```

# Visualization and Analysis of Eigenfaces, Reconstruction and its analysis, Recognition/Classification via LDA

The notebook 'eigenfaces.ipynb' consists of the code for visualization and analysis of eigenfaces, reconstruction, and recognition/classification via LDA.
The key cells of this notebook are:

1.) Cell 2 imports face and non-face images.
2.) Cell 3 performs mean centering and visualizes the "mean face".
3.) Cell 4 performs SVD to get the eigenfaces and their corresponding eigenvalues.
4.) Cell 5 visualizes the first 10 eigenfaces and Cell 6 plots Variance Explained against the number of principal components.
5.) Cell 7 does reconstruction on 80 randomly picked face images (with numpy seed 7) and plots MSE for reconstruction against number of principal components.
    It also visualizes reconstruction for a random image for 1, 10, 100, 111, and 400 principal components used.
6.) Cell 8 does the train and test split and cell 9 consists of code for Linear Discriminant Analysis (LDA). Read the report for more details on the splits. 
7.) Cell 10 and 11 are codes for recognition and classification respectively, with numpy seed 7 for reproducibility. We vary number of principal components
    used from 1 to 50, and perform 10 random-split trials for each, and calculate average accuracy and standard deviation. 

The reader should be able to run the cells in this notebook and reproduce the results reported on the project report. 

# Face Recognition with MTCNN and FaceNet

## Overview
Here we implement  a  face recognition pipeline using:
- **MTCNN** for face detection and alignment
- **FaceNet (InceptionResnetV1)** for face embedding extraction
- **kNN** for identity classification


The pipeline is evaluated on the **AT&T Faces dataset**.


# Environment setup

Create a virtual environment using the command:

```python -m venv qf_final_project```

Activate the virtual environment:

```source qf_final_project/bin/activate```

Install the requirements:

```pip install -r requirements.txt```

Run the classifier

```python facenet_classifier.py```

This will run the pipeline and the results/images associated with the pipeline is available in ```figures``` folder
