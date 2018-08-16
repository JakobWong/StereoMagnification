# StereoMagification
An implementation of [Stereo Magnification: Learning View Synthesis using Multiplane Images](https://arxiv.org/abs/1805.09817) in tensorflow

## Differnces from Original Paper

- The original network performs planar transformation on the predicted color images and then alpha-composit them. In current version, The planar transformation part has been ignored.

## Reference

I organized my code according to that of this project:[Adversarial Video Generation](https://github.com/dyelax/Adversarial_Video_Generation), which makes it easy to maintain. Thanks to [dyelax](https://github.com/dyelax).
