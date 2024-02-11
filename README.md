# INSTALL

Hardware RTX 3080, 16GB GPU RAM

```
pyenv local 3.10
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Setup env variable
```
vi venv/bin/activate
```
add
```
export PYTHONENV=/home/username/test_llm
```

# RAG - Retrieval Augmented Generation
Repo to test out functionality of LangChain v0.1.6 as at Feb 11, 2024