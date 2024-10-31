# Dog Breed Classifier

This project develops a dog breed classifier based on Convolutional Neural Networks (CNNs) using Tensorflow and Keras. It takes a photo of the dog and classifies it to a certain dog breed with supervized learning. The project is trained on a part of the Dog Breed Identification dataset, which can be found on the data sharing website Kaggle.

## Dataset

The data used in this project comes from the Kaggle competition **[Dog Breed Identification](https://www.kaggle.com/competitions/dog-breed-identification/data)**. You will need to download the dataset from Kaggle, place it in the `data` directory, and structure it as described below.

## Folder Structure

Here's an overview of the folder's structure:

```
DogBreedClassifier/
├── data/
│   ├── train/                    # Training images
│   ├── test/                     # Test images
│   ├── labels.csv                # CSV file containing image filenames and breed labels
├── dog_breed_classifier.py       # Script for training the model
├── test_dog_breed_classifier.py  # Script for testing and predicting breeds of test images
├── dog_breed_classifier_model.h5 # Saved model after training
├── class_indices.json            # JSON file storing class index mappings
└── README.md                     # Project documentation
```


## Getting Started

1. **Download the Dataset**:
   - Download the dataset from the [Dog Breed Identification competition on Kaggle](https://www.kaggle.com/competitions/dog-breed-identification/data).
   - Place the images in the `data/train` folder and `data/test` folder as structured above.
   - Ensure `labels.csv` is in the `data` folder.

2. **Train the Model**:
   - Run the following command to train the model:
   ```bash
   python dog_breed_classifier.py
   ```
   - This will:
     - Load and preprocess the data from `data/train`.
     - Train a CNN model to classify dog breeds.
     - Save the trained model as `dog_breed_classifier_model.h5`.
     - Generate a `class_indices.json` file that maps class indices to breed names.

3. **Test the Model**:
   - After training, you can test the model on images in the `data/test` folder:
   ```bash
   python test_dog_breed_classifier.py
   ```
   - This script will:
     - Load the trained model (`dog_breed_classifier_model.h5`) and the `class_indices.json` file.
     - Predict the breed of each image in the `data/test` folder.
     - Save the predictions in `predictions.txt` with each line showing the image name and predicted breed.

    


## Credits

- The dataset is provided by the [Kaggle Dog Breed Identification competition](https://www.kaggle.com/competitions/dog-breed-identification/data).
