# GPO
Code for "Robust Prompt Optimization for Large Language Models Against Distribution Shifts" in EMNLP 2023. See the paper at [this link](https://arxiv.org/abs/2305.13954). 

**Nov 27 2023**: Updated GPO codes and data. Our code is based on [APE](https://github.com/keirp/automatic_prompt_engineer/tree/5f8058c041ba271599539069f594f93f92278a1e). We sincerely thank the APE authors!

![image](https://github.com/li-moxin/GPO/blob/main/intro.png)

*Robust Prompt Optimization Problem*

![image](https://github.com/li-moxin/GPO/blob/main/framework.png)

*GPO framework*

## Setup
After cloning the project, run the following steps to create the environment. 

```
cd GPO
conda create -n gpo python=3.8
pip install -e .
cd experiments
mkdir results
```

## Running the code
Firstly, remember to fill in your openai api key in `experiments/run_gpo.py`. 

For example, to run GPO on Yelp and Flipkart,
```
python run_gpo.py yelp flipkart 6 6 1 36 0 0.6
```
The parameters are source_group, target_group, num_subsamples, num_demos, num_prompts_per_subsample, eval_num, seed, conf_threshold. 

- `source_group:` see all the data file saved at `experiments/data/instruction_induction/raw/induce/`
- `target_group:` see all the data file saved at `experiments/data/instruction_induction/raw/execute/`
- `num_subsamples, num_demos, num_prompts_per_subsample, eval_num:` come from APE. If not assigned, the code will automatically assign the default parameters in the paper. 
- `seed:` 0~4, for five times average. 
- `conf_threshold:` the consistency threshold. If not assigned, the code will automatically assign the default value in the paper. 

The logs are saved at `experiments/results`. 

## Data
The data we used are stored under `experiments/data/instruction_induction/raw/`. In our experiments, the training and validation instances are *randomly sampled* from the entire original dataset. In this repo, we only keep the data used in our experiments under the this folder, where unused data are saved as None for saving space. 

## Contact
Kindly contact `limoxin@u.nus.edu` for any issue, thank you!

## Reference
If you find our work useful, kindly use the following citation. Thank you!
```
@inproceedings{
li2023robust,
title={Robust Prompt Optimization for Large Language Models Against Distribution Shifts},
author={Moxin Li, Wenjie Wang, Fuli Feng, Yixin Cao, Jizhi Zhang, Tat-Seng Chua},
booktitle={The 2023 Conference on Empirical Methods in Natural Language Processing},
year={2023},
url={https://openreview.net/forum?id=svUOik2Xu1}
}
```
