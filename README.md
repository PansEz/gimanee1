# About 

**unofficial version**
**this is for school/personal project**

Gimanee Health Care Chatbot is a Thai-language AI chatbot that answers basic health questions. Trained on 35,000+ Q&A pairs, it helps users access reliable health info easily via a offline website.


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

First of all, Download <ckpt.pt> from our github put it in syrup, open the website (from .html (in syrup)), open terminal tab (use cmd) > path to syrup
then, run command

    python syrup.py

## Credit to 

[nanoGPT](https://github.com/karpathy/nanoGPT)

## Authors

  - **Soroj Sungkhanun**
    

  - Hat tip to anyone whose code is used
  - Inspiration
  - etc
