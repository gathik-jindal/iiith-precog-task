# Breaking CAPTCHAs

lets cook!

## Introduction

This project has given me a start to ML that I've always wanted, but in a very unexpected way. Its been like punching in the dark, looking up videos on what machine learning is, to understanding neural networks, to trying to integrate the different layers of it. Thanks to a course I had on signal processing a lot of the math was easy to understand, like convolution, maxPooling, batchNormalization, etc.

On a side note one thing I have decided during the course of doing the tasks and reading the other tasks is that over the summer I'm going to complete every theme, including the paper readings.

## Learning Curve

The learning curve throughout this project has certainly been very steep. I yesterday didn't know how to even run a pre-trained model and now I am here making one. If I had more time I'd love to document how I'd play around with this model. I would change some datasets, plot graphs against ratio of easy and hard set and see if accuracy was any better. I'd change image sizes, text thickness and font size.

Besides, generation was the hardest to understand and implement. I couldn't get the `forward` function to work, and due to time constraints (I had to do paper reading task) I couldn't complete it. But I would love to see it in action.

### Acknowledgements

I should really thank Youtube and ChatGPT (:P) for my learning here. I did refer to various texts and most notably [this](https://github.com/TheAILearner/A-CRNN-model-for-Text-Recognition-in-Keras) github repo that really got me started. I finally knew the right things that I had to learn and understand.

I really find this unconventional sort of learning of going in creator mode and making huge mistakes and learning as I go, googling things very very fun. But without my understanding in linear algebra, signal processing this task would've been very daunting. My fundamentals in maths have been very essential for which I am very thankful to my college for.

## Methodology

There's not much for me here. I simply did what was obvious. For CAPTCHA dataset generation I used the `pillow` which I knew from before and it fits my use case perfectly. Some improvement I should I have done was made my ML model call the generation of this dataset. That way it'd be easier for me to play around with different numbers and metrics.

Classification is inspired from the github repo and blog I linked here and in my `README.md`. It perfectly fit my use case. I wanted to play around with adding / removing layers and varying datasets and seeing how accuracy changes, but sadly time was not on my side.

Generation is also inspired from there, but due to how I process my image I had to make changes to the numbers (the transform a 128x32 image while I transform a 250x100 image). This makes a big difference because in the end are able to remove height since it reduces to one dimension, but in my case it remains to 5. I had to play around with `squeeze` but I had to stop so I could finish my reading task in time.