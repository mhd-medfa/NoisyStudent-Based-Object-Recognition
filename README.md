# NoisyStudent-Based-Object-Recognition

This repository contains my solution of the [ML 2020 Competition](https://competitions.codalab.org/competitions/27549), which designed for both Robotics and Data Science Master programs at [Innopolis University](https://innopolis.university/). This solution was ranked 3rd overall. My account's username is [diehard](https://competitions.codalab.org/competitions/27549#results).

Details about the solution:
[video link](https://www.youtube.com/watch?v=pYVceiqfntc)

The main problem that's supposed to solve is object recognition for 9 objects:  {airplane, car, bird, cat, deer, dog, horse, ship, truck}. For two different domains data belong. This means there are two different probability distributions from which the images were sampled. We will denote them by Ps(x, y) and Pt(x, y). Look at the Figures below to preview samples from both domains:

Images from Domain Ps(x, y) "labelled dataset":
![](https://i.ibb.co/dPZ0379/xs.png)

Images from Domain Pt(x, y) "unlabelled dataset":
![](https://i.ibb.co/bQY4RBy/xt.png)

## How To Run

To run this solution:

**Conda**
1. `$ conda create --name <envname> --file requirements.txt`
2. `$ source activate <envname>`
3. `$ python run.py`

**Docker**
1. `$ docker build -t ml_competition .`
2. `$ docker run --name ml_competition -d ml_competition`

## References

Noisy Student research paper: https://arxiv.org/abs/1911.04252

Rand Augment: https://github.com/heartInsert/randaugment/blob/master/Rand_Augment.py

Image classification via fine-tuning with EfficientNet: https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/

A good article (in Japanese) with implementation for the noisy student research paper using Resnet50: https://qiita.com/rabbitcaptain/items/a15591ca49dc428223ca

Licensed under [MIT License](LICENSE).
