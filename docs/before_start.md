# Getting Started

we visualize our training details via wandb (https://wandb.ai/site).

## visualization

1) you'll need to login
   ```shell 
   $ wandb login
   ```
2) you'll need to copy & paste you API key in terminal
   ```shell
   $ https://wandb.ai/authorize
   ```
   or add the key to the "code/config/config.py" with
   ```shell
   C.wandb_key = ""
   ```

## training

our code is trained using one nvidia A6000, but our code also supports distributed data parallel mode in pytorch. We
set batch_size=8 for all the experiments, with learning rate 7.5e-6 and 700 * 700 resolution.

### checkpoints

we follow [Meta-OoD](https://github.com/robin-chan/meta-ood) and use the deeplabv3+ checkpoint
in [here](https://github.com/NVIDIA/semantic-segmentation/tree/sdcnet). you'll need to put it in "ckpts/pretrained_ckpts" directory, and
**please note that downloading the checkpoint before running the code is necessary for our approach.**

for training, simply execute

```shell 
$ python rpl_corocl.code/main.py 
```

## inference

please download our checkpoint
from [here](https://drive.google.com/drive/folders/1rVaBRdOpS2JkAo-ZRO64jSZU0VbdZsDn?usp=sharing) and specify the
checkpoint path for **rpl_corocl_weight_path** in config.py.

```shell
python rpl_corocl.code/test.py
```
