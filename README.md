# About 

**unofficial version** // 
**this is for school/personal project**

//Also my first project !//

Gimanee is Health Care website with Thai-language AI chatbot that answers health questionsand give basic health advices. Trained on 35,000+ Q&A pairs between doctors and patients collected from verified health websites etc Moh-prompt, THO, panthip health websites, hugging face, it helps users access reliable health info easily via a offline website.

<img width="1907" height="937" alt="Image" src="https://github.com/user-attachments/assets/43b4a8c9-087c-470a-b7bf-d652be48ff75" />

## Getting Started

These instructions will give you a copy of the project up and running on
your local machine for development and testing purposes. See deployment
for notes on deploying the project on a live system.

### Apps

Requirements for the software and other tools to build, test and push 
- [Python 3.10.0](https://www.python.org/downloads/)
- [CUDA 12.1 (optional)](https://developer.nvidia.com/cuda-toolkit)

### Installing

A step by step series of examples that tell you how to get a development
environment running

Create venv in GIMANEE

    python -m venv <environment_name>

Install library

    pip install numpy sentencepiece random flask torch model transformers datasets tiktoken wandb tqdm

## Running the tests

First of all, Download <ckpt.pt> from our github put it in syrup, open the website (from GIMANEE.html (in syrup)), open terminal tab (use cmd) > path to syrup
then, run command

    python syrup.py

## Credit to 

[nanoGPT](https://github.com/karpathy/nanoGPT)

## Authors

  - **Soroj Sungkhanun**


7/27/2025 last updated
