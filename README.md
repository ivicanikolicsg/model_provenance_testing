# Model provenance testing of LLMs
This is the code for the paper "Model provenance testing for Large Language Models".
<!-- which provides algorithms for testing if one LLM (called tested or child LLM) is obtained by fine-tuning of another LLM (called parent LLM). The repository contains the most general algorithm, which detects if tested LLM has a parent among the provided set of parents. -->

The project consists of:
* The two benchmarks (Bench-A,Bench-B) of LLMs evaluated in the paper (see folder 'data')
* Python implementation of the tester

# Instructions
To run the tester you need at least to specify two files: one for the parent models and one for the tested models. 
```
python tester.py --prompt_id -1 --file_parents <parent_file> --file_candidates <tested_models_file>
```
The tester first caches the outputs of prompts of all parents (because they are reused accross all tested models) and stores them 
