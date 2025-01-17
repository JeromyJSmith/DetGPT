# DetGPT: Detect What You Need via Reasoning

<p align="center" width="100%">
<img src="assets/detgpt.png" alt="DetGPT" style="width: 100%; min-width: 300px; display: block; margin: auto; background-color: transparent;">
</p>


[![Demo](https://img.shields.io/badge/Website-Demo-ff69b4.svg)](https://355b6070043d1f8ae9.gradio.live)
[![Project](https://img.shields.io/badge/Project-Page-20B2AA.svg)](https://detgpt.github.io/)
[![Code License](https://img.shields.io/badge/License-BSD--3--Clause-green)](https://github.com/OptimalScale/DetGPT/blob/master/LICENSE.md)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Embark](https://img.shields.io/badge/Discord-DetGPT-%237289da.svg?logo=discord)](https://discord.gg/u9VJNpzhvA)
[![slack badge](https://img.shields.io/badge/Slack-Join-blueviolet?logo=slack&amp)](https://join.slack.com/t/lmflow/shared_invite/zt-1s6egx12s-THlwHuCjF6~JGKmx7JoJPA)
[![WeChat badge](https://img.shields.io/badge/WeChat-Join-brightgreen?logo=wechat&amp)](https://i.328888.xyz/2023/05/08/i19P4Q.jpeg)

## News
* [2023-05-09] We have launched our [project website](https://detgpt.github.io).
* [2023-05-08] The first version of DetGPT is available now! Try our [demo](https://5c6cdefdefee31f62b.gradio.live/).


## Online Demo
Due to high website traffic, we have created multiple online services. If one link is not working, please use another one. Thank you for your support!

[Demo1](https://a03e18d54fcb7ceb54.gradio.live/)

[Demo2](https://585a0e597e250a662a.gradio.live/)

[Demo3](https://4b04d09170b66bd1b2.gradio.live/)

[comment]: <> ([Demo4]&#40;https://b66150ee453d74dfeb.gradio.live/&#41;)


## Examples

  |   |
:-------------------------:
![ex1](assets/ex1.jpeg) | 
![ex5](assets/ex6.png)  |
![ex3](assets/ex4.png)  |  


## Features
- DetGPT locates target objects, not just describing images.
- DetGPT understands complex instructions, like "Find blood pressure-reducing foods in the image."
- DetGPT accurately localizes target objects via LLM reasoning. - For example, it can identify bananas as a potassium-rich food to alleviate high blood pressure.
- DetGPT provides answers beyond human common sense, like identifying unfamiliar fruits rich in potassium.



## Setup

**1. Installation**
```bash
git clone https://github.com/OptimalScale/DetGPT.git
cd DetGPT
conda create -n detgpt python=3.9 -y
conda activate detgpt
pip install -e .
```

**2. Install GroundingDino**
```bash
python -m pip install -e GroundingDINO
```

**2. Download the pretrained checkpoint**

Our model is based on pretrained language model checkpoints.
In our experiments, we use [Robin](https://github.com/OptimalScale/LMFlow#model-zoo) from [LMFlow team](https://github.com/OptimalScale/LMFlow), and [Vicuna](https://lmsys.org/blog/2023-03-30-vicuna/) and find they perform competitively well.
You can run following script to download the Robin checkpoint:
```
cd output_models
bash download.sh all
cd -
```
Merge the robin lora model with the original llama model and save the merged
model to `output_models/robin-7b`, where the corresponding model path is
specified in this config file
[here](detgpt/configs/models/detgpt_robin_7b.yaml#L16).

To obtain the original llama model, one may refer to this
[doc](https://optimalscale.github.io/LMFlow/examples/checkpoints.html). To
merge a lora model with a base model, one may refer to
[PEFT](https://github.com/huggingface/peft) or use the
[merge script](https://github.com/OptimalScale/LMFlow#53-reproduce-the-result)
provided by LMFlow.

## Training

The code will be released soon.

## Deploy Demo Locally
Run the demo by executing the following command. Replace 'path/to/pretrained_linear_weights' in the config file to the real path.  We currently release linear weights based on [Vicuna-13B-v1.1](https://github.com/lm-sys/FastChat#vicuna-weights) and will release other weights later. The demo runs on 2 GPUs by default, one for the language model and another for GroundingDino.

```
CUDA_VISIBLE_DEVICES=0,1 python demo_detgpt.py --cfg-path configs/detgpt_tasktune_13b_coco.yaml
```


## Acknowledgement
The project is built on top of the amazing open-vocabulary detector [GroundingDino](https://github.com/IDEA-Research/GroundingDINO) and multimodal conversation model [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4), which is based on [BLIP2](https://huggingface.co/docs/transformers/main/model_doc/blip-2) and [Lavis](https://github.com/salesforce/LAVIS). 
Thanks for these great work!


If you're using DetGPT in your research or applications, please cite using this BibTeX:
```bibtex
 @misc{detgpt2023,
    title = {DetGPT: Detect What You Need via Reasoning},
    url = {to be finished},
    author = {Pi, Renjie and Gao, Jiahui and Diao, Shizhe and Pan, Rui and Dong, Hanze and Zhang, Jipeng and Yao, Lewei and Kong, Lingpeng and Zhang, Tong},
    month = {May},
    year = {2023}
}
```


## License
This repository is released under [BSD 3-Clause License](LICENSE.md).
