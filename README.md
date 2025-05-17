<h1>VLSBench: Unveiling Information Leakage in Multimodal Safety</h1>


<span style="color:red">📢 <strong><i>We are currently organizing the code for VLSBench. If you are interested in our work, please star ⭐ our project.</i></strong></span>


<a href='https://arxiv.org/abs/2411.19939'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> <a href='https://ai45lab.github.io/VLSBench'><img src='https://img.shields.io/badge/Project-Page-green'></a> <a href='https://huggingface.co/datasets/Foreshhh/vlsbench'><img src='https://img.shields.io/badge/🤗-Dataset-blue'></a>
</a>
 



<h2 id="updates">🔥 Updates</h2>

📆[2025-05-16] 🎈 Our paper has beed accepeted to ACL2025 Main Conference, see you in Vienna 🎈

📆[2025-03-10] 🎈 Update `data.json` in [HF](https://huggingface.co/datasets/Foreshhh/vlsbench) with `safety reason` and `image description` for more efficient and reliable evaluaiton.

📆[2024-12-16] 🎈 Thanks to @paperweekly to share our work: [Chinese Blog](https://mp.weixin.qq.com/s/OZ_H4bWi6ZYk2K_S-620QA) 🎈

📆[2024-12-16] 🎈 We release the model checkpoints we used in paper: [Qwen2-VL-VLGuard](https://huggingface.co/Foreshhh/Qwen2-VL-7B-VLGuard) [Qwen2-VL-SafeRLHF](https://huggingface.co/Foreshhh/Qwen2-VL-7B-SafeRLHF) 🎈

📆[2024-11-26] 🎈 Our paper, code and dataset are released! 🎈


<h2 id="SALAD-Bench">🎉 Introduction</h2>

<div align="center">
<img src="./static/images/teaser.png" width="900" alt="Intro_img">
</div>

Safety concerns of Multimodal large language models (MLLMs) have gradually become an important problem in various applications. Surprisingly, previous works indicate a counter-intuitive phenomenon that using textual unlearning to align MLLMs achieves comparable safety performances with MLLMs trained with image-text pairs. To explain such a counter-intuitive phenomenon, we discover a **visual safety information leakage (VSIL)** problem in existing multimodal safety benchmarks, i.e., the potentially risky and sensitive content in the image has been revealed in the textual query. In this way, MLLMs can easily refuse these sensitive text-image queries according to textual queries.
However, image-text pairs without VSIL are common in real-world scenarios and are overlooked by existing multimodal safety benchmarks. To this end, we construct multimodal **visual leakless safety benchmark (VLSBench)** preventing visual safety leakage from image to textual query with 2.4k image-text pairs. Experimental results indicate that VLSBench poses a significant challenge to both open-source and close-source MLLMs, including LLaVA, Qwen2-VL, Llama3.2-Vision, and GPT-4o. This study demonstrates that textual alignment is enough for multimodal safety scenarios with VSIL, while multimodal alignment is a more promising solution for multimodal safety scenarios without VSIL.


<h2 id="SALAD-Bench">📍 Problem of VSIL </h2>

<div align="center">
<img src="./static/images/alignment_table.png" width="900" alt="Intro_img">
</div>

The VSIL problem leads the shortcut alignment methods: textual alignment to the multimodal safety challenge.

<h2 id="dataset">⚙️ Dataset</h2>

You can download our [dataset](https://huggingface.co/datasets/Foreshhh/vlsbench) from Huggingface, also, json file can be quickly accessed [here](./data/data.json)

<div align="center">
<img src="./static/images/category_paper.png" width="900" alt="Intro_img">
</div>

You can check some examples here

<div align="center">
<img src="./static/images/examples.png" width="900" alt="Intro_img">
</div>

<h2 id="quick-start">🚀 Usage</h2>

Our code support several archs

- openai: api based model with openai format: If you are using openai apis, please remember to specify the model_name and your customized api_key and api_base in [load_openai.py](./models/load_openai.py)
- llava: the origin llava implementation
- llava_hf: the hugginface implementation of llava
- llava_next: the hugginface implementation of llava 1.6 and above
- qwen2vl: for Qwen2-VL
- mllama: for Llama3.2-Vision

First, download the [dataset](https://huggingface.co/datasets/Foreshhh/vlsbench) in huggingface and specify the downloaded dir as the `ROOT_DIR`.

And specify the evaluation api key used in [here](https://github.com/hxhcreate/VLSBench/blob/main/eval_utils.py#L59).

Then, execute the following script:

```bash
python eval.py --arch llava --data_root $ROOT_DIR --output_dir ./outputs
```

<h2 id="citation">📑 Citation</h2>

```bibtex
@article{hu2024vlsbench,
      title={VLSBench: Unveiling Visual Leakage in Multimodal Safety}, 
      author={Xuhao Hu and Dongrui Liu and Hao Li and Xuanjing Huang and Jing Shao},
      journal={arXiv preprint arXiv:2411.19939},
      year={2024}
}
```

<hr>

