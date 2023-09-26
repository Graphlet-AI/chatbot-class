# Graphlet AI - Chatbot Class

This course covers generative AI, large language models (LLMs), vector search, retrieval aided generation (RAG), LLM fine-tuning, and more. The course is designed for software engineers, data scientists, and machine learning engineers who want to learn how to build AI-powered chatbots.

<center><img src="images/Graphlet-AI-Banner-with-Hypergraph-and-Womans-Head.jpg" /></center>

## Course Essentials

### Skill Prerequisites

- Basic knowledge of Python
- Basic CS and math skills

### Course Outline

- Introduction to Generative AI
- Introduction to Large Language Models (LLMs)
- Introduction to Vector Search
- Introduction to Retrieval Aided Generation (RAG)
- Introduction to LLM Fine-Tuning

### Course Projects

- Build a generative chatbot
- Build a retrieval-based chatbot
- Build a generative chatbot with retrieval-aided generation (RAG)
- Build a generative chatbot with LLM fine-tuning

### Course Materials

- [Course Slides](https://bit.ly/graphlet_chatbot_slides)
- [Course Code Repository](https://github.com/Graphlet-AI/chatbot-class) (this repository)

## Docker Setup

I provide a Docker image for this course that uses [Jupyter Notebooks](https://jupyter.org/). Docker allows you to run the class's code in an environment precisely matching the one in which the code was developed and tested. You can also use the Docker image to run the course code in VSCode or another editor (see below).

In addition to 

### Install Docker

[Install docker](https://docs.docker.com/engine/install/) and then check the [Get Started](https://www.docker.com/get-started/) page if you aren't familiar.

There are several docker containers used in this course:

- `jupyter`: Jupyter Notebook server where we will interactively write and run code.
- `chroma`: Chroma vector database server where we will store and query vector embeddings of documents for RAG.
- `neo4j`: Neo4j graph database server where we will store and query graph data for prompt engineering and fine-tuning LLMs.
- `opensearch`: OpenSearch server where we will store and query documents for RAG.

### Docker Compose

Bring up the course environment with the following command:

```bash
docker compose up -d
```

Find the Jupyter Notebook url via this command:

```bash
docker logs jupyter -f --tail 100
```

Look for the url with `127.0.0.1` and open it. You should see the Jupyter Notebook home page.

NOTE: Insert an image of Jupyter home page for this course.

### Docker and VSCode

NOTE: add instructions.

## Code-Level Environment Setup

We use a Docker image to run the course, but you can also setup the environment so the code will work in VSCode or another editor. We provide a development tools setup using `black`, `flake8`, `isort`, `mypy` and `pre-commit` for you to modify and use as you see fit.

### Install Anaconda Python

We use Anaconda Python, Python version 3.10.0, for this course. You can download Anaconda Python from [here](https://www.anaconda.com/products/individual). Once you have installed Anaconda Python, you can create a new environment for this course by running the following command:

```bash
conda create -n chatbot-class python=3.10 -y
```

When you create a new environment or start a new shell, you will need to activate the `chatbot-class` conda environment with the following command:

```bash
conda activate chatbot-class
```

Now you are running Python 3.10 in the `chatbot-class` environment. To use this Python in VSCode, hit SHIFT-CMD-P (on Mac) and select `Python: Select Interpreter`. Then select the `chatbot-class` environment's Python.

To deactivate this environment, run:

```bash
conda deactivate
```

#### Other Virtual Environments

Note: I don't support other environments, but you can actually use any Python 3.10 if you are smart enough to make that work. :) You will need to manage your own virtual environments. Python 3's [`venv`](https://docs.python.org/3/library/venv.html) are easy to use.

To create a `venv` for the project, run:

```bash
python3 -m venv chatbot-class
```

To activate this venv run:

```bash
source chatbot-class/bin/activate
```

To deactivate this environment, run:

```bash
deactivate
```

### Install Poetry for Dependency Management

We use [Poetry](https://python-poetry.org/) for dependency management, as it makes things fairly painless. 

Verify [Poetry installation instructions here](https://python-poetry.org/docs/#installation) so you know the URL `https://install.python-poetry.org` is legitimate to execute in `python3`.

Then install Poetry with the following command:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

It is less "clean" in terms of environmental isolation, but alternatively you can install poetry via `pip`:

```bash
pip install poetry
```

### Install Dependencies via Poetry

```bash
poetry install
```
