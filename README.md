

ffmpeg is used in sublinux, draw images is used in windows, training is used in ubuntu docker container.

## Train & Test

Export current directory to `PYTHONPATH`:

```bash
export PYTHONPATH=`pwd`
```

- demo

```
python3 demo.py -c yamls/ps_gat.yaml -m cache/ps_gat/100/models/checkpoint_epoch_200.pth
```

- train

```
python3 train.py -c yamls/ps_gat.yaml
```

- test

```
python3 test.py -c yamls/ps_gat.yaml -m cache/ps_gat/100/models/checkpoint_epoch_200.pth
```


## dist train 

```shell
#!/usr/bin/env bash

set -x
NGPUS=$1
PY_ARGS=${@:2}

python3 -m torch.distributed.launch --nproc_per_node=${NGPUS} train.py --launcher pytorch ${PY_ARGS} 
```

## dist test

```shell
#!/usr/bin/env bash

set -x
NGPUS=$1
PY_ARGS=${@:2}

python3 -m torch.distributed.launch --nproc_per_node=${NGPUS} test.py --launcher pytorch ${PY_ARGS}

```
