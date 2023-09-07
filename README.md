# Handwritten Bangla Character Recognition using CNN in PyTorch

### About the dataset

The [Original Dataset](https://data.mendeley.com/datasets/hf6sf8zrkc/2) contains Handwritten Bangla Characters and Digits, a total of **84** classes where,
- 11 are Vowels
- 37 are consonants
- Rest are some of the Complex characters (consisting of 2 or more graphemes)

### Instructions for Training the model

- Open CMD/Terminal and clone the repository using the command : `git clone git@github.com:ayan-cs/handwritten-character-recognition-banglalekha`
- Download the BanglaLekha Numerals dataset from the given link above and extract inside the repository folder. It is recommended not to make any change to the dataset folder.
- Preprocess the data by executing the `Data_Preparation.ipynb` notebook. This should create *Train* and *Validation* splits inside the parent dataset folder. (If you want to reverse the split and re-split the dataset again, a code snippet is available inside the notebook)
- Configure `train_config.yaml` file.
- Run the script on CMD/Terminal : `python main.py train`
- The trained model will be available inside **Checkpoints** folder and the plots will be saved inside **Plots & Outputs** folder.

### Instructions for Inference/Prediction

- Open CMD/Terminal and clone the repository using the command : `git clone git@github.com:ayan-cs/handwritten-character-recognition-banglalekha`
- Make sure the data is preprocessed.
- Configure `inference_config.yaml` file. For demo, one trained ResNet-34 model has been provided.
- Open CMD/Terminal, run the command : `python main.py inference`
- The outputs will be available inside the **Plots & Outputs** folder.