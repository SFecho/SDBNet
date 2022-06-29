# SDBNet

Code repo for the paper "Deep non-blind deblurring network for saturated blurry images" 




Download the training and testing dataset from the [[url]](https://www.kaggle.com/datasets/sfecho/sdbnetdataset).



## Prerequisites

- Linux (Ubuntu 20.04)
- Python 3.7
- Pytorch 1.8.1
- torchvision 0.9
- cuda 11.1
- tensorboardX
- imageio
- hydra
- omegaconf
- tqdm
- glob

## Get Started
### Training
First, Updating the config/default-training.yaml and using the follow command to training the CML-AE:

```yaml
defaults:
  - hydra/job_logging: custom
  - hydra/hydra_logging: colorlog
  - job: train/deblurmasked
```

```shell
python trainer.py
```

Second, Updating the config/default-training.yaml and using the follow command to pre-training the CEN:

```yaml
defaults:
  - hydra/job_logging: custom
  - hydra/hydra_logging: colorlog
  - job: train/deblurmaskdistill
```

```shell
python trainer.py
```

Finally, Updating the config/default-training.yaml and using the follow command to training the SDBNet:

```yaml
defaults:
  - hydra/job_logging: custom
  - hydra/hydra_logging: colorlog
  - job: train/deblurframe2
```

```shell
python trainer.py
```

### Testing
Updating the config/default-testing.yaml and using the follow command to testing the SDBNet:
```yaml
defaults:
  - hydra/job_logging: custom
  - hydra/hydra_logging: colorlog
  - job: test/deblurframe2
```

```shell
python tester.py
```


## License

This project is open sourced under MIT license.
