# Music Recommendation Using Deep Learning

Music Recommendation using latent feature vectors obtained from a network trained on the Free Music Archive dataset.


## Acknowledgments

* Project is inspired by **Sander Dieleman's** blog, [*Recommending music on Spotify with Deep Learning*](http://benanne.github.io/2014/08/05/spotify-cnns.html), and a paper that he co-published, [*Deep content-based music recommendation*](https://proceedings.neurips.cc/paper/2013/file/b3ba8f1bee1238a2f37603d90b58898d-Paper.pdf)
* The initial code is found online from **Vikram Shenoy** - [Vikram Shenoy](https://github.com/VikramShenoy97)
* [Free Music Archive Dataset](https://github.com/mdeff/fma)



## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Install the required pacakges with the following line:

```
pip install -r requirements.txt
```

### Dataset
The fma_small dataset consists of 8000 mp3 files from the [Free Music Archive](https://github.com/mdeff/fma).
To use this dataset, go to the link and download **fma_metadata.zip** and
**fma_small.zip** from the website.

Each file in fma_small is a 30 second clip of music. The dataset is balanced and has 8 genres 
( Hip-Hop, International, Electronic, Folk, Experimental, Rock, Pop, and Instrumental).

The dataset is stored in the folder **Dataset** as *fma_small*. 
When you run it, make sure the folders containing the audio files are in *fma_small*,
and *tracks.csv* (contains genre of the songs) is in the folder *fma_metadata*. Both *fma* folders should be in **Dataset**.



## Training

Run the script *train.py* in the terminal as follows.
```
Python train.py
```
If the folders *Train_Spectogram_Images* and *Train_Sliced_Images* exist, then no further
data processing will be done. If you want to create new images from the audio to feed
into the neural network, make sure to delete these folders before you run this line.

### Model and History

The trained network is then saved as *Model.h5* and it's history is saved as 
*training_history.csv* in the *Saved_Model* folder. There is a temporary model 
in the folders if you don't want to retrain the model.

### Training Performance

```
Final Training Accuracy = 72.24%
Final Validation Accuracy = 64.44%
Temporary Model Training Accuracy = 72.13%
Temporary Model Validation Accuracy = 58.33% (64.89% in previous epoch)
```


## Recommendation

Add a folder *DLMusicTest_30* under **Dataset** to run this. 
The test mp3 audio files should be in this folder. Similar to training,
if *Test_Sliced_Images* and *Test_Spectogram_Images* exist, then no processing
will be done on the test data as they already exist. If you want
to try a new set of test data, make sure to delete the folders first.

Run the script *recommendation.py* in the terminal as follows.
```
python recommendation.py
```

This will give you a list of songs.


Enter an anchor song for which you want similar recommendations (Choose one from your list).
```
Enter a Song Name:
SampleSongName
```
### Results
The code generates two recommendations for the song.

