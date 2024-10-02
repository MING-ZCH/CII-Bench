# CII-Bench
**Can MLLMs Understand the Deep Implication Behind Chinese Images?**
<p align="center">
<a href="https://cii-bench.github.io/"><img src="https://img.shields.io/badge/Homepage-blue" alt="Homepage"></a>
<a href="https://github.com/MING-ZCH/CII-Bench"><img src="https://img.shields.io/badge/Code-24292e" alt="Code"></a>
<a href="https://huggingface.co/datasets/"><img src="https://img.shields.io/badge/Dataset-yellow" alt="Dataset"></a>
</p> 

## 🔥News
<!-- - [Oct. 2024]: Our paper has released on arXiv, check it out! -->
## Introduction
As the capabilities of Multimodal Large Language Models (MLLMs) continue to improve, the need for higher-order capability evaluation of MLLMs is increasing. However, there is a lack of work evaluating MLLM for higher-order perception and understanding of Chinese visual content.
To fill the gap, we introduce the **C**hinese **I**mage **I**mplication understanding **Bench**mark, **CII-Bench**, which aims to assess the higher-order perception and understanding capabilities of MLLMs for Chinese images. 
CII-Bench stands out in several ways compared to existing benchmarks. Firstly, to ensure the authenticity of the Chinese context, images in CII-Bench are sourced from the Chinese Internet and manually reviewed, with corresponding answers also manually crafted. Additionally, CII-Bench incorporates images that represent Chinese traditional culture, such as famous Chinese traditional paintings, which can deeply reflect the model's understanding of Chinese traditional culture. \
Through extensive experiments on CII-Bench across multiple MLLMs, we have made significant findings: \
Initially, a substantial gap is observed between the performance of MLLMs and humans on CII-Bench. The highest accuracy of MLLMs attains 64.4\%, where as human accuracy averages 78.2\%, peaking at an impressive 81.0\%. Subsequently, MLLMs perform worse on Chinese traditional culture images, suggesting limitations in their ability to understand high-level semantics and lack a deep knowledge base of Chinese traditional culture. Finally, it is observed that most models exhibit enhanced accuracy when image emotion hints are incorporated into the prompts. \
We believe that CII-Bench will enable MLLMs to gain a better understanding of Chinese semantics and Chinese-specific images, advancing the journey towards expert artificial general intelligence (AGI).

## 🏆 Mini-Leaderboard


## Citation

If you find our work helpful in your research, please cite the following paper:
