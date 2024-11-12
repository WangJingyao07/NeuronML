# Awesome Neuromodulated Meta-Learning

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re) 
![Static Badge](https://img.shields.io/badge/Meta_Learning-Flexible_Network_Structure-blue)
![Static Badge](https://img.shields.io/badge/to_be_continue-orange)
![Static Badge](https://img.shields.io/badge/TPAMI_Preprint-yellow)
![pv](https://pageview.vercel.app/?github_user=WangJingyao07/NeuronML)
![Repo Clones](https://img.shields.io/badge/Clones-2-blue)
![Stars](https://img.shields.io/github/stars/WangJingyao07/NeuronML)


**Official code for "Exploring Flexible Structure in Meta-Learning"** [PDF](https://doi.org/10.48550/arXiv.2411.06746)

🥇🌈This repository contains not only the CODE of our NeuronML but also several self-make Application cases of Neuroscience. 

*Note:* The CODE is the Pytorch version of MAML (original Tensorflow version is [CODE-maml](https://github.com/cbfinn/maml))


## Create Environment

For easier use and to avoid any conflicts with existing Python setup, it is recommended to use [`virtualenv`](https://docs.python-guide.org/dev/virtualenvs/) to work in a virtual environment. Now, let's start:

**Step 1:** Install [`virtualenv`](https://docs.python-guide.org/dev/virtualenvs/)

```bash
pip install --upgrade virtualenv
```

**Step 2:** Create a virtual environment, activate it:

```bash
virtualenv venv
source venv/bin/activate
```

**Step 3:** Install the requirements in [`requirements.txt`](requirements.txt).

```bash
pip install -r requirements.txt
```


## Data Availability

All data sets used in this work are open source. The download and deployment ways are as follows:
​
* miniImageNet, Omniglot, and tieredImageNet will be downloaded automatically upon runnning the scripts, with the help of [pytorch-meta](https://github.com/tristandeleu/pytorch-meta).

* For ['meta-dataset'](https://github.com/google-research/meta-dataset/blob/e95c50658e4260b2ede08ede1129827b08477f1a/prepare_all_datasets.sh), follow the following steps: Download ILSVRC2012 (by creating an account [here](https://image-net.org/challenges/LSVRC/2012/index.php) and downloading `ILSVRC2012.tar`) and Cu_birds2012 (downloading from `http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz`) separately. Then, Run `sbatch scripts/download_meta_dataset/install_meta_dataset_parallel.sh` to download and prune all datasets in a parallel fashion. All the ten datasets should be copied in a single directory.

* For the few-shot-regression setting, Sinusoid, Sinusoid & Line, and Harmonic dataset are toy examples and require no downloads. Just follow the implementation in the paper.

* For the reinforcement learning environment:  Khazad Dum and [MuJoCo](https://github.com/google-deepmind/mujoco)

Now, you have completed all the settings, just directly train and test as you want :)


## Train

We offer two ways to run our code (Take [`MAML`](scripts/MAML) with [`meta-dataset`](scripts/MAML/Train/train_maml_metadataset_all_samplers.sh) as an example), **which will be provided after a few days (until the arxiv been open-sourced)*



## Citation
If you find our work and codes useful, please consider citing our paper and star our repository (🥰🎉Thanks!!!):
```
@misc{wang2024neuromodulatedmetalearning,
      title={Neuromodulated Meta-Learning}, 
      author={Jingyao Wang and Huijie Guo and Wenwen Qiang and Jiangmeng Li and Changwen Zheng and Hui Xiong and Gang Hua},
      year={2024},
      eprint={2411.06746},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2411.06746}, 
}
```



