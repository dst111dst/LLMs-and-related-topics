# I.LLMs-and-related-topics
Store recent topics about LLMs in the first part.

## LLM and related topics

## 1.Use and Adapt LLMs

### Prompting for few-shot learning

1. [Making Pre-trained Language Models Better Few-shot Learners](https://arxiv.org/pdf/2012.15723.pdf) ([blog post](https://gaotianyu.xyz/prompting/))
2. [How Many Data Points is a Prompt Worth?](https://arxiv.org/pdf/2103.08493.pdf)

### Prompting as parameter-efficient fine-tuning


(from prefix prompting)

1. [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/pdf/2101.00190.pdf)
2. [The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/pdf/2104.08691.pdf)

### In-context learning

1. [Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?](https://arxiv.org/pdf/2202.12837.pdf)
2. [An Explanation of In-context Learning as Implicit Bayesian Inference](https://arxiv.org/pdf/2111.02080.pdf) (or [blog post](http://ai.stanford.edu/blog/understanding-incontext/) )

### Calibration of prompting LLMs

1. [Calibrate Before Use: Improving Few-Shot Performance of Language Models](https://arxiv.org/pdf/2102.09690.pdf)
2. [Surface Form Competition: Why the Highest Probability Answer Isn’t Always Right](https://arxiv.org/pdf/2104.08315.pdf)

### Reasoning

1. [Chain of Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/pdf/2201.11903.pdf)
2. [Large Language Models are Zero-Shot Reasoners](https://arxiv.org/pdf/2205.11916.pdf)
3. [Towards Reasoning in Large Language Models: A Survey](https://arxiv.org/abs/2212.10403)

### Knowledge

1. [Language Models as Knowledge Bases?](https://arxiv.org/pdf/1909.01066.pdf)
2. [How Much Knowledge Can You Pack Into the Parameters of a Language Model?](https://arxiv.org/pdf/2002.08910.pdf)
3. [Knowledge Injection into Encoder-Decoder Language Models](https://arxiv.org/pdf/2302.09170.pdf) 2023
4. [Prompting Large Language Models with Answer Heuristics for Knowledge-based Visual Question Answering](https://arxiv.org/pdf/2303.01903) 2023

##  2.Data, Model Scaling and Risks

### Data

1. [Documenting Large Webtext Corpora: A Case Study on the Colossal Clean Crawled Corpus](https://arxiv.org/pdf/2104.08758.pdf)

### Scaling

1. [Training Compute-Optimal Large Language Models](https://arxiv.org/pdf/2203.15556.pdf)

### Privacy

1. [Extracting Training Data from Large Language Models](https://arxiv.org/pdf/2012.07805.pdf)

### Bias & Toxicity : evaluation

1. [RealToxicityPrompts: Evaluating Neural Toxic Degeneration in Language Models](https://arxiv.org/pdf/2009.11462.pdf)
2. [OPT paper, Section 4](https://arxiv.org/pdf/2205.01068.pdf)

### Bias & Toxicity : mitigation

1. [Self-Diagnosis and Self-Debiasing: A Proposal for Reducing Corpus-Based Bias in NLP](https://arxiv.org/pdf/2103.00453.pdf)


## 3.Beyond Current LLMs: Models and Applications


### Sparse models

1. [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://jmlr.org/papers/volume23/21-0998/21-0998.pdf)

### Retrieval-based LMs

1. [Improving language models by retrieving from trillions of tokens](https://arxiv.org/pdf/2112.04426.pdf)
2. [Can Retriever-Augmented Language Models Reason? The Blame Game Between the Retriever and the Language Model](https://arxiv.org/pdf/2212.09146.pdf) 2022 
3. [Retrieval Enhanced Model for Commonsense Generation](https://arxiv.org/pdf/2105.11174) ACL, 2021
4. [Recent Advances in Retrieval-Augmented Text Generation](https://lemaoliu.github.io/retrieval-generation-tutorial/)
   90-Minute Tutorial @ IJCAI Conference 2022.
5. [Building Scalable, Explainable, and Adaptive NLP Models with Retrieval](https://ai.stanford.edu/blog/retrieval-based-NLP/) 2021


### Training LMs with human feedback

1. [Training language models to follow instructions with human feedback](https://arxiv.org/pdf/2203.02155.pdf)

### Code LMs

1. [Evaluating Large Language Models Trained on Code](https://arxiv.org/pdf/2107.03374.pdf)

### Multimodal LMs

1. [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/pdf/2204.14198.pdf)
2. [Prompting Large Language Models with Answer Heuristics for Knowledge-based Visual Question Answering](https://arxiv.org/pdf/2303.01903)  2023
3. [Language Models with Image Descriptors are Strong Few-Shot Video-Language Learners](https://arxiv.org/abs/2205.10747) NeurIPS, 2022
4. [From Visual Prompt Learning to Zero-Shot Transfer: Mapping Is All You Need](https://arxiv.org/pdf/2303.05266) 2023


### AI Alignment

1. [A General Language Assistant as a Laboratory for Alignment](https://arxiv.org/pdf/2112.00861.pdf)
2. [Alignment of Language Agents](https://arxiv.org/pdf/2103.14659.pdf)
3. [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/pdf/2204.05862.pdf)


# II.From LLM to VisualLM

Related topics: tuning language model for multimodal tasks, use the augmentation methods in LLMs to improve VLMs, use the prompts or ICL to improve VLMs, analyzing the similar abilities in VLMs (e.g. zero-shot/ few-shot learning ability )

## Blogs and surveys of VLMs

 [A Dive into Vision-Language Models](https://huggingface.co/blog/vision_language_pretraining) HuggingFace, 2023

- Since 2021, we’ve seen an increased interest in models that combine vision and language modalities (also called **joint vision-language models**), such as [OpenAI’s CLIP](https://openai.com/blog/clip/). Joint vision-language models have shown particularly impressive capabilities in very challenging tasks such as **image captioning, text-guided image generation and manipulation, and visual question-answering**. This field continues to evolve, and so does its effectiveness in improving z**ero-shot generalization** leading to various practical use cases.

- In this blog post, we'll introduce **joint vision-language models** focusing on how they're trained. We'll also show how you can leverage..

- A vision-language model typically consists of 3 key elements: an image encoder, a text encoder, and a strategy to fuse information from the two encoders. These key elements are tightly coupled together as the loss functions are designed around both the model architecture and the learning strategy. While vision-language model research is hardly a new research area, the design of such models has changed tremendously over the years. Whereas earlier research adopted hand-crafted image descriptors and pre-trained word vectors or the frequency-based TF-IDF features, the latest research predominantly adopts i**mage and text encoders with transformer architectures** to **separately or jointly learn image and text features**. These models are pre-trained with strategic pre-training objectives that enable various downstream tasks.

## Multimodal Pre-train Language Model

1. [Flamingo: A Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198) NeurIPS, 2022 (visual + lm)

2. [Language Quantized AutoEncoders: Towards Unsupervised Text-Image Alignment](https://arxiv.org/abs/2302.00902) 2023

...

## Retrieval Augmentation in Visual LMs

1. [Retrieval-Augmented Multimodal Language Modeling](https://arxiv.org/pdf/2211.12561.pdf) ACL, 2022
2. [Re-Imagen: Retrieval-Augmented Text-to-Image Generator](https://arxiv.org/pdf/2209.14491.pdf) 2022
3. [SMALLCAP: Lightweight Image Captioning Prompted with Retrieval Augmentation](https://arxiv.org/abs/2209.15323) 2022
4. [Efficient Image-Text Retrieval via Keyword-Guided Pre-Screening](https://arxiv.org/pdf/2303.07740.pdf) 2023
5. [Re-ViLM: Retrieval-Augmented Visual Language Model for Zero and Few-Shot Image Captioning](https://arxiv.org/pdf/2302.04858.pdf) 2023

## Visual Augmentation in Visual LMs
1. [Visually-Augmented Language Modeling](https://openreview.net/pdf?id=8IN-qLkl215) ICLR, 2023

## SLM Augmentation in Visual LMs
1. [Prompting Large Language Models with Answer Heuristics for Knowledge-based Visual Question Answering](https://arxiv.org/pdf/2303.01903.pdf) 2023

