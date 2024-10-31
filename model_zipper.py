import tarfile

with tarfile.open('dog_breed_classifier_model.tar.bz2', "w:bz2") as tar:
    tar.add('dog_breed_classifier_model.h5')

# This file is for compressing the model file.
