# NLP-word-vectors-subjective-objective-detection
Note: This is a course project of MIE324 at University of Toronto (year 2018). If you are a student of this course and find this project is similar to one of the projects you got, please contact me and I will make this repository private. Thank you!

This project builds and trains 3 models (MLP, CNN, and RNN) to detect whether a sentence is subjective or objective. To try it out, download this repository and run subjective_bot.py. It will load the pre-trained models and output their predictions on any sentence you enter. 

### Data (data.tsv)
The data used for training comes form portions of movie review from Rotten Tomatoes (assumed to be subjective) and summaries of plot from IMDB (assumed to be objective).

### split_data.py 
perform the strtified train validation test split 

### model.py
build 3 models 

### main.py
train the 3 models

### subjective_bot.py
load the pre-trained models and predict whether the sentence entered by the user is subjective or objective



