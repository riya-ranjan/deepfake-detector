# deepfake-detector

## This is a deepfake detection project that attempts multimodal deepfake detection. 
We attempt a multimodal audio-visual approach to detecting deepfakes, particularly for
detecting fake pornography. We use a CNN-LSTM architecture to parallely process audio 
and video features before predicting whether or not a video is a deep fake. 

For training and testing purposes, we use the LAV-DF dataset as presented by 
[Cai et. al](https://arxiv.org/pdf/2204.06228v2) That dataset can be downloaded
[here](https://drive.google.com/file/d/1-OQ-NDtdEyqHNLaZU1Lt9Upk5wVqfYJw/view?usp=sharing).

Our proposed architecture is included in /models, which includes our model as well as data
preprocessing strategy. Experimental results are included in /experiments.
