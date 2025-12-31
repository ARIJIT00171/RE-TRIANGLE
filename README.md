# RE-TRIANGLE: Does TRIANGLE Enable Multimodal Alignment Beyond Cosine Similarity?

![Animation](https://github.com/ARIJIT00171/RE-TRIANGLE/blob/722c96284f424a0abc5d38489580c99035f2f5a0/intro_gif.gif
)

Welcome to the official repository for our reproducibility research of "**TRIANGLE**", a similarity measure that is directly computed in the higher-dimensional space spanned by the modality embeddings. The code is implemented in Python using PyTorch and Distributed Processing to provide an efficient and fast implementation of the algorithm.

The code in this repository has been used to research the reproducibility of the experiments conducted in the original 'TRIANGLE' paper. Furthermore, the repository contains our new additions. Our contributions include:
<ul>
  <li><b>Cosine Regularization</b>: We have added the code for TRIANGLE loss with cosine regularization, something which was mentioned in the paper but not implemented.</li>
  <br>
  <li><b>Multiple Method Support</b>: We have refactored the code from scratch and included multiple methods (VAST, TRIANGLE and TRIANGLE with cosine regularization) in the same repository for easy execution.</li>
  <br>
  <li><b>Ranking Metrics</b>: We have added the logic for the generation of trec files and computation of additional ranking metrics like Reciprocal Rank and nDCG.</li>
  <br>
  <li><b>Chunked Area Calculation</b>: We have added a chunk based version of area_computation that is GPU efficient and can handle large tensors without OOM error.</li>
  <br>
  <li><b>Explainability of Triangle</b>: We have added a colab notebook that explains Triangle by visualizing tri-modality features in 3 dimension.</li>
</ul>


## Table of Content
- [Installation Guide](#installation-guide)
- [Datasets](#datasets)
- [Download basic encoder's pretrained checkpoints](#download-basic-encoders-pretrained-checkpoints)
- [How to Run](#how-to-run)
  - [Zero Shot Audio and Video Retrieval](#zero-shot-audio-and-video-retrieval)
  - [Training From Scratch](#training-from-scratch)
  - [TRIANGLE Explainability](#triangle-explainability)


## Installation Guide
<ol type=1>
<li> Download the repository as a .zip or clone the repository using:
<br>git clone git@github.com:ARIJIT00171/RE-TRIANGLE.git
</li>
<br>
<li> Run the remaining steps <b>only if</b> running the code on local machine:
<ol type='a'>
<li> Install the correct version of the used packages from the .yml file using the following command:
<b>conda env create -f triangle_env.yaml</b>
</li>
<li> Upon installation of the environment, it can be (de)activated using:
<br>
<b>conda activate triangle</b>
<br>
<b>conda deactivate triangle</b>
</li>

<li> The environment can be deleted using:
<b>conda remove -n triangle --all</b>
</li>

<li> Additional packages can be installed using pip:
<b>pip install [package_name]</b>
</li>
</ol>
</li>
<br>
<li> Run the remaining steps <b>only if</b> running the code on snellius:
<ol type='a'>
<li> Create the environment using the following slurm job:

```
#!/bin/bash

#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --job-name=InstallEnvironment
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=04:00:00
#SBATCH --output=slurm_output_install_env_%A.out

module purge
module load 2024
module load Anaconda3/2024.06-1

cd $HOME/Environment/
conda env create -f triangle_env.yaml
```
</li>
</ol>
</li>
</ol>

## Datasets
The following are the datasets that were used in the experiments:
<br>
<ol type=1>
<li>
<b>MSR-VTT</b>, a large-scale benchmark dataset of ~10,000 web videos with multiple natural-language captions per video, widely used for video–text retrieval and captioning tasks.
</li>
<br>
<li>
<b>DiDeMo</b>, a dataset used to train models to find the exact moment in a video that matches a text description.
</li>
<br>
<li>
<b>ActivityNet</b>, a large video dataset for understanding and detecting human activities in long, untrimmed videos.
</li>
<br>
<li>
<b>VATEX</b>, a multilingual video dataset with English and Chinese captions for video understanding and captioning tasks.
</li>
<br>
<li>
<b>AudioCaps</b>, a dataset of short videos paired with natural-language captions that describe the sounds in the video.
</li>
<br>
<li>
<b>YouCook2</b>, a dataset of cooking videos with step-by-step text descriptions aligned to video segments.
</li>
</ol>
<br>
Download the datasets and store them locally. Each dataset has a videos directory and annotations json file. To extract and create audio files from videos, run adhoc_scripts/extract_audio.py as follows:

```
python adhoc_scripts/extract_audio.py --video_root <Directory containing dataset videos> --audio_root <Directory to write extracted audio>
```

Once the datasets have been downloaded properly, edit the config files inside the config/triangle/finetune_cfg directory to replace the preset dataset paths with your local system paths (where the datasets reside on your machine after downloading).

## Download basic encoder's pretrained checkpoints

Make a dir named pretrained_weights under the main work dir.

<ol type=1>
<li>
Download evaclip weight:

```
wget -P pretrained_weights/clip/ https://huggingface.co/QuanSun/EVA-CLIP/resolve/main/EVA01_CLIP_g_14_psz14_s11B.pt
```

</li>
<br>
<li>
Download beats weight from <a>https://github.com/microsoft/unilm/tree/master/beats</a>
</li>
<br>
<li>
Download bert weight:
<br>

```
from transformers import BertForMaskedLM, BertTokenizer
bert = BertForMaskedLM.from_pretrained('bert-base-uncased')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert.save_pretrained('pretrained_weights/bert/bert-base-uncased')
bert_tokenizer.save_pretrained('pretrained_weights/bert/bert-base-uncased')
```

</li>
<br>
<li>
Download Pretrained Triangle checkpoint (on vast27m) from <a href="https://drive.google.com/file/d/1T-wuY-CzUp_PF8UuhKDqXEAL86obUpzj/view?usp=sharing">link</a>
</li>
</ol>
<br>

The processed pretrained_weights path should be as follows:

```
    ├── pretrained_weights
    │   ├── beats
    │   │   └── BEATs_iter3_plus_AS2M.pt
    │   ├── bert
    │   │      └── bert-base-uncased
    │   ├── clip
    │   │    └── EVA01_CLIP_g_14_psz14_s11B.pt
    │   ├── triangle_pretraining
    │   │    └── ckpt
    │   │      └── model_step_200.pt
    │   │    └── log
    │   │      └── hps.json
    │   │      └── log.txt
```

## How to Run
Now that the environment has been correctly installed and datasets have been downloaded (along with audio extraction), it is time to run the code.

For all experiment runs, please make sure to add the following to your job file (to disable online wandb logs that require API key):

```
export WANDB_MODE=offline
```

This will create wandb log files locally. These can later be uploaded to wandb using an API key.

### Zero Shot Audio and Video Retrieval

The audio/video retrieval experiment can be divided into two subtasks: 
<br>
1) Text-to-Data Retrieval (T2D): Given a natural language query, find the most relevant audio/video from a set of candidate videos.
2) Data-to-Text Retrieval (D2T): Given a candidate audio/video, find the most relevant natural language query.

Our code gives you the evaluation scores for both the directions. To run the zero shot audio/video retrieval for a dataset, run the following command in job file:

```
torchrun \
    --nnodes 1 \
    --node_rank 0 \
    --nproc_per_node=4 \ # Number of GPUs for distributed run
    --master_port=29501 \
    distributed_run.py \
      --model_type triangle \ # model_type can be vast, triangle or triangle_cos
      --mode testing \
      --pretrain_dir <path to directory containing pretrained triangle checkpoint and weights for encoders used)> \
      --config <config path of the dataset (inside config/triangle/finetune_cfg directory)> \
      --output_dir <Path to directory where results will be stored>
```

The outputs will be saved in the output_dir folder (passed as command line argument and created automatically by the code).
This directory will also contain the generated trec files (created during evaluation). To obtain the new ranking metrics from these, run the adhoc_scripts/ir_measures_eval.py file using:
<br>

```
python adhoc_scripts/ir_measures_eval.py \
    <path to directory containing trec_runs directory (with generated trec files)> \
    --output <path to directory where result csv will be stored>
```

### Training From Scratch

The Training From Scratch experiment aims to perform a deeper study on the ability of TRIANGLE to better model the latent space by letting TRIANGLE losses learn from scratch on the MSR-VTT dataset for the multimodal text-to-audio/video (T2AV) and audio/video-to-text (AV2T) tasks.  
<br>
To run the training from scratch experiment for MSR-VTT dataset, run the following command in job file:

```
torchrun \
    --nnodes 1 \
    --node_rank 0 \
    --nproc_per_node=4 \ # Number of GPUs for distributed run
    --master_port=29501 \
    distributed_run.py \
      --model_type triangle \ # model_type can be vast, triangle or triangle_cos
      --mode training \
      --config <config path for MSR-VTT dataset (inside config/triangle/finetune_cfg directory)> \
      --output_dir <Path to directory where results will be stored>
```

The outputs will be saved in the output_dir folder (passed as command line argument and created automatically by the code).
This directory will also contain the generated trec files (created during evaluation). To obtain the new ranking metrics from these, run the adhoc_scripts/ir_measures_eval.py file using:
<br>

```
python adhoc_scripts/ir_measures_eval.py \
    <path to directory containing trec_runs directory (with generated trec files)> \
    --output <path to directory where result csv will be stored>
```

### TRIANGLE Explainability

To run the TRIANGLE Explainability experiment, do the following: 
<br>
1) Run adhoc_scripts/explain_sample_yc.py file to generate and extract sample features to plot. This can be done by running the following command in job file:
<br>

```
python adhoc_scripts/explain_sample_yc.py \
    --pretrain_dir <path to directory containing pretrained triangle checkpoint and weights for encoders used)> \
    --video_dir <path to directory containing videos for the dataset> \
    --audio_dir <path to directory containing audios for the dataset> \
    --json_path <path to directory containing captions/descriptions for the dataset> \
    --num_samples 8 # Number of samples to extract
```

2) After the above script has run, it will create a folder named "saved_features_sample". Compress it to create a zip file "saved_features_sample.zip".
3) Open the adhoc_scripts/Explain_Triangle.ipynb notebook in Colab and Run the notebook. The first cell will prompt you to upload a file. Upload the "saved_features_sample.zip" created in previous step.
4) After the remaining cells finish running, you can see the visualization results.
