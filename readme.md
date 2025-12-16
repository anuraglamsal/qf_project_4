
About the project


This project explores how faces can be represented and analyzed using both classical statistical methods and modern deep learning techniques.
The work begins with Eigenfaces, one of the earliest approaches in face recognition. Eigenfaces treat face images as high-dimensional data and look for a compact representation that preserves the most meaningful variations across individuals. This approach allows us to study what information is truly essential for representing faces.
We also explore a deep learning–based pipeline using FaceNet. By comparing these two approaches, the project highlights the comparision of face recognition techniques that use variance-based linear models and identity-driven nonlinear embeddings pipeline.


The data  for faces is in att_faces directory and we also use non face images to make the classification more robust. Images without faces is in non_faces directory.


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