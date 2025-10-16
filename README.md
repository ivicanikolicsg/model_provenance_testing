# Model provenance testing of LLMs
This is the code for the paper "Model provenance testing for Large Language Models".
<!-- which provides algorithms for testing if one LLM (called tested or child LLM) is obtained by fine-tuning of another LLM (called parent LLM). The repository contains the most general algorithm, which detects if tested LLM has a parent among the provided set of parents. -->

The project consists of:
* The two benchmarks (Bench-A,Bench-B) of LLMs evaluated in the paper (see folder `data`)
* Python implementation of the tester

The tester will download all required LLMs from HuggingFace.  You can test LLMs from the two benchmarks, or your own (see below). 

To produce the full evaluation of the two benchmarks, it requires around 1TB of space and around 2 days (mostly spent downloading LLMs from HuggingFace).

# Instructions
To run the tester you need at least to specify two files: one for the parent models and one for the tested models. 
```
python tester.py --prompt_id -1 --file_parents <parent_file> --file_candidates <tested_models_file>
```
The tester first caches the outputs of all parent LLMs accross different set of prompts (because they are reused accross all tested models) and then stores them. All future testers can use these cached outputs, by specifying the prompt_id, i.e. instead of `--prompt_id -1` which means recompute outputs, you can use `--prompt_id <prompt_cache_id>`, where `<prompt_cache_id>` increase sequentially `0,1,...`, check the folder `cached_prompts` for available ids.

The file `<parent_file>` contains one HuggingFace model per line, e.g.
```
openai-community/gpt2
EleutherAI/pythia-70m
microsoft/DialoGPT-medium
...
```
The file `<tested_models_file>` contains lines of tested models, which can be specified with ground truth parent model (format <tested_model>,<parent_model>), or without, 

i.e. either with parent (provide None, if tested model does not have parent)
```
BEE-spoke-data/zephyr-220m-sft-full,BEE-spoke-data/smol_llama-220M-openhermes
codeparrot/codeparrot-small,None
...
```
of without parent
```
BEE-spoke-data/zephyr-220m-sft-full
codeparrot/codeparrot-small
...
```
In the latter case, the tester will just output the parent guess, wherease in the former case, in addition it will provide statistics (percentage, recall, ...) about the correct guesses against the provided parents. 

# Evaluating Bench-A, Bench-B
To run the two benchmarks you can use the provided files of parent and candidate models, i.e. 
```
python tester.py --prompt_id -1 --file_parents data/benchmark_A_parents.txt --file_candidates benchmark_A_candidates.txt
python tester.py --prompt_id -1 --file_parents data/benchmark_B_parents.txt --file_candidates benchmark_B_candidates.txt
```

# Advanced options
The tester supports some options (e.g. adjusting number of prompts, device used for inference, ...), check `--help` for extensive list.
