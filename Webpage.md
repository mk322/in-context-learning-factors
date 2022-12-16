# 499G1/590G1 Final Project Report

## Abstract
Large pretrained language model has gained more and more attention in recent years. Not only because of their representation power and superior performance with traditional fine-tuning on various dataset, its ability of generalizing to new task with "in-context learning" opened another path of Natural Language Processing (NLP) research.

There has been various techniques proposed to improve the performance under prompting setup. Those techniques are generally referred as "prompt engineering". Motivated by a recent paper [2]

## Introduction
As demonstrated first in the GPT3 paper published by OpenAI [1], the large language model (here we are talking about the model that is in the scale of a few hundred million parameters to a few hundred billion parameters) that is pretrained on the general auto-regressive language modeling objective can achieve state-of-the-art performance on various downstream tasks **without** updating any parameters. The techniques to achieve such result, referred as "in-context learning" or "prompting", attracted lots of research attention due to its simplicity, sample efficientcy, and memory efficiency.

In-context learning is usually discussed under the context of few-shot learning or zero-shot learning. Given a downstream task, in the zero-shot setup, we will prepend a string of task description to the input $x$; while in the few-shot setup, we will prepend a list of ($x_i$, $y_i$) pair to the input $x$. An example of in-context learning in both setup is illustrated in Figure 1.

In this project, we are interesting in the result claimed in [2], which states that the input-output pairing is actually **not** important. i.e.: the prompts could contains wrong examples and still achieve high performance. What actually matters is the distribution of the input text (the $x_i$ part in the prompt) and the label space (does the prompt contain all labels and only all the labels we can possibly have in the dataset). This is an interesting finding since usually we would think the input-label pairing is important as it demonstrates to the model how to do the task.

## Related Works
The related works, which are usually referred as "prompt engineering", aims to find a general way to design the prompt for different downstream tasks. The main idea for our project came from the paper “Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?” [2], where researchers studied what aspects of the demonstrations help the model learn and improve its performance on the final task. Specifically, we want to perform some experiments on (1) input distribution, (2) output distribution, (3) input-output mapping, and (4) formatting to study how the input distributions and output distributions affect the performance of the model for in-context learning, following the idea
from [2]. [3] provides a great review on literatures in this field. Besides designing better prompt to improve the performance, understanding what is important and what contributes to the in-context "learning" behavior is another importnat topic. Papers like [2] and [4] did ablation on various perspestive of prompt designs and conclude that some parts of the prompt are more important than others. The code we adopted came from [github repo](https://github.com/Alrope123/rethinking-demonstrations).

## Methodology

## Experiments
##### Models 
We experimented with 4 models in total. We provide 2 dense, decoder-only language models, GPT-J-6B and GPT-2-Large. We employ each LM with the direct and channel inference approaches, following Min et al. (2021a). 

##### Datasets
In addition to the datasets the authors used in their experiments, we experimented with 7 new datasets, including coLA, poem sentiment analysis, glue-wnli, climate_fever, glue-rte, superglue-cb, sick on sentiment analysis and natural language inference (full list and references provided in Appendix A). All datasets are text classification. We use these datasets because they (1) are true low-resource datasets with less than 10K train- ing examples, (2) include well-studied benchmarks from GLUE (Wang et al., 2018) and Super- GLUE (Wang et al., 2019a), and (3) cover diverse domains including science, linguistics and more.

##### Other Details
We use k = 16 examples as demonstrations by default for all experiments in the paper unless otherwise specified. Examples are sampled at uniform from the training data. We choose a set of k training examples using 5 different random seeds and run experiments 5 times. For each dataset and seed, we used 100-200 test examples to calculate the average for our F1 score. We report Macro-F1 scores for classification tasks. We compute the per-dataset average over seeds and then report the macro-average over datasets. We use minimal templates in forming an input sequence from an example. We refer to the original paper’s Appendix B for more details. 

## Results
##### Input-Label Mapping
![]
##### Input Distribution

##### Label Space

##### Prompt Format
Based on the figure, we find that models trained with instructive templates generally perform better than the model trained with irrelevant and misleading templates. This is valid for all the models and datasets in our experiments. There is generally a more significant difference among templates on the direct method than the channel method. For example, shifting from an instructive template to an irrelevant template, there is a 4% drop with the direct gpt2-large model and a 0.5% drop with the channel gpt2-large model. There isn’t a consistent relation between the performance of models trained with instructive templates and models trained without templates. (Concatenate premise with hypothesis without other words). On the direct gpt2-large model, the macro-F1 score is 34.09 for the instructive template and 36.35% for the null template, while the macro-F1 score is 47.47% for the instructive template and 45.12% for the null template on the channel gpt2-large model. 




## Reference
1. Brown, Tom, et al. "Language models are few-shot learners." Advances in neural information processing systems 33 (2020): 1877-1901.
2. Min, Sewon, et al. "Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?." arXiv preprint arXiv:2202.12837 (2022).
3. Liu, Pengfei, et al. "Pre-train, prompt, and predict: A systematic survey of prompting methods in natural language processing." arXiv preprint arXiv:2107.13586 (2021).
4. Webson, Albert, and Ellie Pavlick. "Do Prompt-Based Models Really Understand the Meaning of their Prompts?." arXiv preprint arXiv:2109.01247 (2021).
