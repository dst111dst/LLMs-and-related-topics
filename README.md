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


## Paper List (Since 2023)

## 0.Not included

Not included in my current topics but stil valuable:

[1] [Paraphrasing evades detectors of AI-generated text, but retrieval is an effective defense](https://arxiv.org/pdf/2303.13408) 2023 

- Topic: AI-Generator detection. Refer to: DetectGPT./ GPTZERO./ A watermark of Large Language Model
- I just suppose that retriever could serve as a tool in many aspects. :)  And for prompting/tuning/data augmentation and so on.

> To detect the deployment of large language models for malicious use cases (e.g., fake content creation or academic plagiarism), several approaches have recently been proposed for identifying AI-generated text via watermarks or statistical irregularities. How robust are these detection algorithms to paraphrases of AI-generated text? To **<u>stress test</u>** these detectors【what is stress test??】, we first train **an 11B parameter paraphrase generation model (DIPPER)** that can **paraphrase paragraphs**, optionally leveraging surrounding text (e.g., user-written prompts) as context. DIPPER also uses scalar knobs to control the amount of lexical diversity and reordering in the paraphrases. <u>Paraphrasing text generated by three large language models</u> (including GPT3.5-davinci-003) <u>with DIPPER</u> successfully evades several detectors, including **watermarking, GPTZero, DetectGPT, and OpenAI's text classifier**. For example, DIPPER drops the detection accuracy of DetectGPT from 70.3% to 4.6% (at a constant false positive rate of 1%), without appreciably modifying the input semantics. To **increase the robustness of AI-generated text detection** to **paraphrase attacks**, we introduce a simple defense that <u>**relies on retrieving semantically-similar generations**</u> and **must be maintained by a language model API provider**. Given a candidate text, our algorithm **searches a database of sequences previously generated by the API,** looking for <u>*sequences that match the candidate text within a certain threshold*</u>. We empirically verify our defense using a database of 15M generations from a fine-tuned T5-XXL model and find that it can detect 80% to 97% of paraphrased generations across different settings, while only classifying 1% of human-written sequences as AI-generated. We will open source our code, model and data for future research.

Simple method but more values in its empirical findings.

[2] [ZeroNLG: Aligning and Autoencoding Domains for Zero-Shot Multimodal and Multilingual Natural Language Generation](https://arxiv.org/pdf/2303.06458) 2023

- Topic: Multimodal alignment;Multimodal representation learning; zero-shot learning
- Codes will be released soon in the [repo](https://github.com/yangbang18/ZeroNLG).

> Natural Language Generation (NLG) accepts input data in the form of images, videos, or text and generates corresponding natural language text as output. Existing NLG methods mainly adopt a supervised approach and rely heavily on coupled data-to-text pairs. However, for many targeted scenarios and for non-English languages, sufficient quantities of labeled data are often not available. To relax the dependency on labeled data of downstream tasks, we propose an intuitive and effective zero-shot learning framework, ZeroNLG, which can deal with multiple NLG tasks, including image-to-text (image captioning), video-to-text (video captioning), and text-to-text (neural machine translation), across English, Chinese, German, and French within a unified framework. ZeroNLG does not require any labeled downstream pairs for training. During training, ZeroNLG (i) projects different domains (across modalities and languages) to corresponding coordinates in a shared common latent space; (ii) bridges different domains by aligning their corresponding coordinates in this space; and (iii) builds an unsupervised multilingual auto-encoder to learn to generate text by reconstructing the input text given its coordinate in shared latent space. Consequently, during inference, based on the data-to-text pipeline, ZeroNLG can generate target sentences across different languages given the coordinate of input data in the common space. Within this unified framework, given visual (imaging or video) data as input, ZeroNLG can perform zero-shot visual captioning; given textual sentences as input, ZeroNLG can perform zero-shot machine translation. We present the results of extensive experiments on twelve NLG tasks, showing that, without using any labeled downstream pairs for training, ZeroNLG generates high-quality and believable outputs and significantly outperforms existing zero-shot methods.

methodology overview:  Fig. 1. During training, ZeroNLG first (i) projects different data **across modalities and languages** to **corresponding coordinates** in **a shared common latent space**; (ii) aligns their coordinates to bridge different domains; Here Si and Sj refer to **the text in non-English text,** e.g. Chinese and Germany; (iii) performs **unsupervised auto-encoding** to learn to **generate/reconstruct text given the coordinate of input text in this space**. During inference, ZeroNLG **encodes the input data acquiring its coordinate in this space**, which can be directly used to perform zero-shot data-to-text generation (i.e., visual captioning and machine translation) without the need for downstream labeled pairs.

[3] [Visually-Prompted Language Model for Fine-Grained Scene Graph Generation in an Open World](https://arxiv.org/abs/2303.13233) 2023

> Scene Graph Generation (SGG) aims to extract <subject, predicate, object> relationships in images for vision understanding. Although recent works have made steady progress on SGG, they still suffer long-tail distribution issues that tail-predicates are more costly to train and hard to distinguish due to a small amount of annotated data compared to frequent predicates. Existing re-balancing strategies try to haddle it via prior rules but are still confined to pre-defined conditions, which are not scalable for various models and datasets. In this paper, we propose a Cross-modal prediCate boosting (CaCao) framework, where a visually-prompted language model is learned to generate diverse fine-grained predicates in a low-resource way. The proposed CaCao can be applied in a plug-and-play fashion and automatically strengthen existing SGG to tackle the long-tailed problem. Based on that, we further introduce a novel Entangled cross-modal prompt approach for open-world predicate scene graph generation (Epic), where models can generalize to unseen predicates in a zero-shot manner. Comprehensive experiments on three benchmark datasets show that CaCao consistently boosts the performance of multiple scene graph generation models in a model-agnostic way. Moreover, our Epic achieves competitive performance on open-world predicate prediction.

- Topic: Visual Genome
- Main focus: enrich the **tail predicates** of scene graphs in a l**ow-cost and easily scalable** way.

[4] [A Survey of Large Language Models](https://arxiv.org/pdf/2303.18223.pdf) 2023 Wayne Xin Zhao, ..., Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, ..., Junjie Zhang,  ..., Ji-Rong Wen

> Language is essentially a complex, intricate system of human expressions governed by grammatical rules. It poses a significant challenge to develop capable AI algorithms for comprehending and grasping a language. As a major approach, language modeling has been widely studied for language understanding and generation in the past two decades, evolving from statistical language models to neural language models. Recently, pre-trained language models (PLMs) have been proposed by pre-training Transformer models over large-scale corpora, showing strong capabilities in solving various NLP tasks. Since researchers have found that model scaling can lead to performance improvement, they further study the scaling effect by increasing the model size to an even larger size. Interestingly, when the parameter scale exceeds a certain level, these enlarged language models not only achieve a significant performance improvement but also show some special abilities that are not present in small-scale language models. To discriminate the difference in parameter scale, the research community has coined the term large language models (LLM) for the PLMs of significant size. Recently, the research on LLMs has been largely advanced by both academia and industry, and a remarkable progress is the launch of ChatGPT, which has attracted widespread attention from society. The technical evolution of LLMs has been making an important impact on the entire AI community, which would revolutionize the way how we develop and use AI algorithms. In this survey, we review the recent advances of LLMs by introducing the background, key findings, and mainstream techniques. In particular, we focus on four major aspects of LLMs, namely pre-training, adaptation tuning, utilization, and capacity evaluation. Besides, we also summarize the available resources for developing LLMs and discuss the remaining issues for future directions.



[5] [MM-REACT: Prompting ChatGPT for Multimodal Reasoning and Action](https://arxiv.org/abs/2303.11381) Microsoft, 2023

> We propose MM-REACT, a system paradigm that integrates ChatGPT with a pool of vision experts to achieve multimodal reasoning and action. In this paper, we define and explore a comprehensive list of advanced vision tasks that are intriguing to solve, but may exceed the capabilities of existing vision and vision-language models. To achieve such advanced visual intelligence, MM-REACT introduces a **textual prompt design** that can represent text descriptions, textualized **spatial coordinates**【what‘s this??】, and **aligned file names for dense visual signals such as images and videos** 【how to align？？】. MM-REACT's prompt design allows language models to accept, associate, and process multimodal information, thereby facilitating the synergetic combination of ChatGPT and various vision experts. Zero-shot experiments demonstrate MM-REACT's effectiveness in addressing the specified capabilities of interests and its wide application in different scenarios that require advanced visual understanding. Furthermore, we discuss and compare MM-REACT's system paradigm with an alternative approach that extends language models for multimodal scenarios through joint finetuning. Code, demo, video, and visualization are available at [this https URL](https://multimodal-react.github.io/)
>
> 一个直接的问题：和visual chatgpt 有什么区别？

Methodology Overview:

> To this end, we present MM-REACT, a system paradigm that composes numerous vision experts with ChatGPT for multimodal reasoning and action. To enable images and videos as inputs, we **use their file path as the input to ChatGPT.** The file path **functions as a placeholder**,【但这个文件路径名难道不会包含有语义信息吗？就像visual chatgpt里的api名称一样？】 allowing ChatGPT to treat it as a black box. Whenever a specific property such as celebrity names or box coordinates is required, ChatGPT is expected to seek help from a specific vision expert to identify the desired information. To **inject the knowledge of vision experts' usages into ChatGPT**, we **add instructions to ChatGPT prompts** about each expert's capability, input argument type, and output type, along with a few in-context examples for each expert. Additionally, a special watchword is instructed such that we can use regex expression matching to invoke the expert accordingly. 为此，我们提出了MM-REACT，一个将众多视觉专家与ChatGPT结合起来进行多模态推理和行动的系统范式。为了使图像和视频能够作为输入，我们使用其文件路径作为ChatGPT的输入。文件路径作为一个占位符，允许ChatGPT将其作为一个黑盒子。每当需要一个特定的属性，如名人的名字或盒子的坐标，ChatGPT就会向特定的视觉专家寻求帮助，以确定所需的信息。为了将视觉专家的使用知识注入ChatGPT，我们在ChatGPT的提示中加入了关于每个专家的能力、输入参数类型和输出类型的说明，以及每个专家的一些内文例子。此外，还指示了一个特殊的观察词，这样我们就可以使用重合表达式匹配来调用相应的专家。
>
> We show MM-REACT's representative visual understanding capabilities in Figure 1. For example, MM- REACT could associate information from multiple uploaded receipts and calculate the total travel cost (“Multi-Image Reasoning”), recognize and answer questions about the “morel mushrooms” (“Open-World Concept Under- standing”), and condense a long video into representative thumbnails (“Video Summarization and Event Localiza- tion”).These visual intelligence features are similar to those found in recent models, such as multimodal GPT-4 [23] and PaLM-E [10], but achieved via prompting instead of additional multimodal training. The MM-REACT system may provide extra flexibility in module upgrades, and may be effective in certain visual understanding tasks by better utilizing existing specialized vision experts, such as celebrity recognition and dense captioning.



[6] [AGREE: Aligning Cross-Modal Entities for Image-Text Retrieval Upon Vision-Language Pre-trained Models](https://dl.acm.org/doi/10.1145/3539597.3570481) WSDM, 2022

Task: image-text retrieval; cross modal retrieval

> Image-text retrieval is a challenging cross-modal task that arouses much attention. While the traditional methods cannot break down the barriers between different modalities, Vision-Language Pre-trained (VLP) models greatly improve image-text retrieval performance based on massive image-text pairs. Nonetheless, the VLP-based methods are still prone to produce retrieval results that cannot be cross-modal aligned with entities. Recent efforts try to fix this problem at the pre-training stage, which is not only expensive but also unpractical due to the unavailable of full datasets. In this paper, we novelly propose a lightweight and practical approach to **align cross-modal entities** for image-text retrieval upon VLP models only at the **fine-tuning and re-ranking stages**. We employ **external knowledge** and tools to **construct extra fine-grained image-text pairs**, and then **emphasize cross-modal entity alignment** through **<u>contrastive learning and entity-level mask modeling in fine-tuning</u>**. Besides, **<u>two re-ranking strategies</u>** are proposed, including one specially designed for **zero-shot scenarios.** Extensive experiments with several VLP models on multiple Chinese and English datasets show that our approach achieves state-of-the-art results in nearly all settings.

supplementray [video](https://dl.acm.org/doi/10.1145/3539597.3570481) from WSDM 2023

[7] [Scaling Vision with Sparse Mixture of Experts](https://arxiv.org/abs/2106.05974) 2021

topic: model scaling; vision language model pre-training

> Sparsely-gated Mixture of Experts networks (MoEs) have demonstrated excellent scalability in Natural Language Processing. In Computer Vision, however, almost all performant networks are "dense", that is, every input is processed by every parameter. We present a Vision MoE (V-MoE), a sparse version of the Vision Transformer, that is scalable and competitive with the largest dense networks. When applied to image recognition, V-MoE matches the performance of state-of-the-art networks, while requiring as little as half of the compute at inference time. Further, we propose an extension to the routing algorithm that can prioritize subsets of each input across the entire batch, leading to adaptive per-image compute. This allows V-MoE to trade-off performance and compute smoothly at test-time. Finally, we demonstrate the potential of V-MoE to scale vision models, and train a 15B parameter model that attains 90.35% on ImageNet.

[8] DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature 2023

- Related tool: [GPT-4 And ChatGPT detector by ZeroGPT: detect OpenAI text](https://www.zerogpt.com/) - ZeroGPT the most Advanced and Reliable Chat GPT and GPT-4 detector tool

> The fluency and factual knowledge of large language models (LLMs) heightens the need for corresponding systems to detect whether a piece of text is machine-written. For example, students may use LLMs to complete written assignments, leaving instructors unable to accurately assess student learning. In this paper, we first demonstrate that text sampled from an LLM tends to occupy negative curvature regions of the model's log probability function. Leveraging this observation, we then define **a new curvature-based criterion for judging** if a passage is generated from a given LLM. This approach, which we call DetectGPT, does not require training a separate classifier, collecting a dataset of real or generated passages, or explicitly watermarking generated text. It uses only log probabilities computed by the model of interest and random perturbations of the passage from another generic pre-trained language model (e.g, T5). We find DetectGPT is more discriminative than existing zero-shot methods for model sample detection, notably improving detection of fake news articles generated by 20B parameter GPT-NeoX from 0.81 AUROC for the strongest zero-shot baseline to 0.95 AUROC for DetectGPT. See this https URL for code, data, and other project information.
>
> 大型语言模型（LLMs）的流畅性和事实知识使得相应的系统更需要检测一段文本是否是机器写的。例如，学生可能会使用LLMs来完成书面作业，使教员无法准确评估学生的学习。在本文中，我们首先证明，从LLM中取样的文本往往占据模型的对数概率函数的负曲率区域。利用这一观察结果，我们定义了一个新的基于曲率的标准来判断一个段落是否是由一个给定的LLM生成的。这种方法，我们称之为DetectGPT，不需要训练单独的分类器，不需要收集真实或生成的段落的数据集，也不需要明确对生成的文本进行水印。它只使用感兴趣的模型计算的对数概率和另一个通用的预训练语言模型（如T5）的段落的随机扰动。我们发现DetectGPT在模型样本检测方面比现有的零散方法更具辨别力，特别是提高了对20B参数GPT-NeoX生成的假新闻文章的检测，从最强的零散基线的0.81AUROC提高到DetectGPT的0.95AUROC。有关代码、数据和其他项目信息，请参见这个https URL。

Task: detect AI-generated tasks

[9] [ChatGPT as a Factual Inconsistency Evaluator for Abstractive Text Summarization](https://arxiv.org/pdf/2303.15621)

>  The performance of abstractive text summarization has been greatly boosted by pre-trained language models recently. The main concern of existing **abstractive summarization methods** is the *<u>factual inconsistency problem</u>* of their generated summary. To alleviate the problem, many efforts have focused on developing effective factuality evaluation metrics based on natural language inference and question answering et al. However, they have limitations of high computational complexity and relying on annotated data. Most recently, large language models such as ChatGPT have shown strong ability in not only natural language understanding but also natural language inference. In this paper, we study the **factual inconsistency evaluation ability** of ChatGPT under **the zero-shot setting** by evaluating it on the **coarse-grained and fine-grained factuality evaluation tasks** including **binary natural language inference (NLI), summary ranking, and consistency rating**. Experimental results show that ChatGPT outperforms previous SOTA evaluation metrics on 6/9 datasets across three tasks, demonstrating its great potential for assessing factual inconsistency in the zero-shot setting. The results also highlight the importance of **prompt design** and the need for future efforts to address ChatGPT's limitations on e**valuation bias, wrong reasoning, and hallucination.**

LLM as Evaluators



[9] [A Survey of Large Language Models](https://arxiv.org/pdf/2303.18223.pdf) 2023.04

Since researchers have found that model scaling can lead to performance improvement, they further study the scaling effect by increasing the model size to an even larger size. Interestingly, when the parameter scale exceeds a certain level, these enlarged language models not only achieve a significant performance improvement but also show some **<u>special abilities (e.g., in-context learning) that are not present in small-scale language models (e.g., BERT).</u>** To discriminate the difference in parameter scale,the research community has coined the term large language models (LLM) for the PLMs of significant size (e.g., containing tens or hundreds of billions of parameters). Considering this rapid technical progress, in this survey, we review the recent advances of LLMs by introducing the <u>background, key findings, and mainstreamn techniques.</u> In particular, we focus on four major aspects of LLMs, namely **pre-training, adaptation tuning, utilization, and capacity evaluation**. Besides, we also summarize **the available resources for developing LLMs** and discuss the remaining issues for future directions. This survey provides an up-to-date review of the literature on LLMs, which can be a useful resource for both researchers and engineers.



[10] [Vision-Language Models for Vision Tasks: A Survey]() 2023.04

Most visual recognition studies rely heavily on crowd-labelled data in deep neural networks (DNNs) training, and they usually train a DNN for each single visual recognition task, leading to a laborious and time-consuming visual recognition paradigm. To address the two challenges, Vision-Language Models (VLMs) have been intensively investigated recently, which learns rich vision-language correlation from **web-scale image-text pairs** that are almost infinitely available on the Internet and enables **zero-shot predictions** on various visual recognition tasks with **a single VLM**. This paper provides a systematic review of visual language models for various visual recognition tasks, including: (1) the background that introduces the development of visual recognition paradigms; (2) the foundations of VLM that summarize the widely-**adopted network architectures,pre-training objectives, and downstream tasks**; (3) the widely-adopted **datasets in VLM pre-training and evaluation**s; (4) the review and categorization of existing VLM **pre-training**
methods,VLM **transfer learning methods**, and VLM **knowledge distillation** methods; (5) the **benchmarking**, analysis and discussion of the reviewed methods;(6) several research challenges and potential research directions that could be pursued in the future VLM studies for visual recognition. A project associated with this survey has been created at https://github.com/jingyi0000/VLM_survey.



[11] [Self-Supervised Multimodal Learning: A Survey]() 2023.04

However, the **heavy dependence on data** paired with expensive human annotations impedes scaling up models. Meanwhile, given the availability of **large-scale unannotated data** in the wild, self-supervised learning has become an attractive strategy to alleviate the annotation bottleneck. Building on these two directions,self-supervised multimodal learning (SSML) provides ways to leverage supervision from raw multimodal data. In this survey, we provide a comprehensive review of the state-of-the-art in SSML, which we categorize along three orthogonal axes: **objective functions, data alignment, and model architectures**. These axes correspond to the inherent characteristics of self-supervised learning methods and multimodal data.
Specifically,we classify **training objectives into instance discrimination, clustering, and masked prediction categories**. We also discuss multimodal input data pairing and **alignment strategies during training**. Finally, we review model architectures including the **design of encoders,fusion modules, and decoders**,which are essential components of SSML methods. We review **downstream multimodal application tasks,** reporting the concrete performance of the state-of-the-art image-text models and multimodal video models,and also review real-world applications of SSML algorithms in diverse fields such as healthcare, remote sensing,and machine translation.
Finally, we discuss challenges and future directions for SSML. A collection of related resources can be found at:
https://github.com/ys-zong/awesome-self-supervised-multimodal-learning.



[12] [UKP-SQuARE v3: A Platform for Multi-Agent QA Research](https://arxiv.org/pdf/2303.18120.pdf) 2023.04

The continuous development of Question Answering (QA) datasets has drawn the research community's attention toward multi-domain models. A popular approach is to use multi-dataset models, which are models trained on multiple datasets to learn their regularities and prevent overfitting to a single dataset. However, with the proliferation of QA models in online repositories such as GitHub or Hugging Face, an alternative is becoming viable. Recent works have demonstrated that combining expert agents can yield large performance gains over multi-dataset models. To ease research in multi-agent models, we extend UKP-SQuARE, an online platform for QA research, to support three families of multi-agent systems: i) agent selection, ii) early-fusion of agents, and iii) late-fusion of agents. We conduct experiments to evaluate their inference speed and discuss the performance vs. speed trade-off compared to multi-dataset models. UKP-SQuARE is open-source and publicly available at http://square.ukp-lab.de.



[13] [QUADRo: Dataset and Models for QUestion-Answer Database Retrieval](https://arxiv.org/pdf/2304.01003.pdf) 2023.04

An effective paradigm for building Automated Question Answering systems is the **re-use of previously answered questions**, e.g., for FAQs or forum applications. Given a database (DB) of question/answer (q/a) pairs, it is possible to answer a target question by scanning the DB for similar questions. In this paper, we scale this approach to open domain, making it competitive with other standard methods, e.g., unstructured document or graph based. For this purpose, we (i) build a large scale DB of 6.3M q/a pairs, using public questions, (ii) design a new system based on neural IR and a q/a pair reranker, and (iii) construct training and test data to perform comparative experiments with our models. We demonstrate that Transformer-based models using (q,a) pairs outperform models only based on question representation, for both neural search and reranking. Additionally, we show that our DB-based approach is competitive with Web-based methods, i.e., a QA system built on top the BING search engine, demonstrating the challenge of finding relevant information. Finally, we make our data and models available for future research.



[14] [Fairness-guided Few-shot Prompting for Large Language Models](https://arxiv.org/pdf/2303.13217.pdf) 2023.04



[15] [LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention](https://arxiv.org/pdf/2303.16199.pdf) 2023.03

We present LLaMA-Adapter, a lightweight adaption method to efficiently fine-tune LLaMA into an instruction-following model. Using 52K self-instruct demonstrations, LLaMA-Adapter only introduces 1.2M learnable parameters upon the frozen LLaMA 7B model, and costs less than one hour for fine-tuning on 8 A100 GPUs. Specifically, we adopt a set of learnable adaption prompts, and prepend them to the input text tokens at higher transformer layers. Then, a zero-init attention mechanism with zero gating is proposed, which adaptively injects the new instructional cues into LLaMA, while effectively preserves its pre-trained knowledge. With efficient training, LLaMA-Adapter generates high-quality responses, comparable to Alpaca with fully fine-tuned 7B parameters. Furthermore, our approach can be simply extended to multi-modal input, e.g., images, for image-conditioned LLaMA, which achieves superior reasoning capacity on ScienceQA. We release our code at [this https URL](https://github.com/ZrrSkywalker/LLaMA-Adapter).

 

[16] [Explicit Planning Helps Language Models in Logical Reasoning](https://arxiv.org/pdf/2303.15714.pdf) 2023.03

Language models have been shown to perform remarkably well on a wide range of natural language processing tasks. In this paper, we propose a novel system that uses language models to perform **multi-step logical reasoning**. Our system incorporates **explicit planning into its inference procedure**, thus able to make more informed reasoning decisions at each step by looking ahead into their future effects. In our experiments, our full system significantly outperforms other competing systems. On a **multiple-choice question answering task**, our system performs competitively compared to GPT-3-davinci despite having only around 1.5B parameters. We conduct several ablation studies to demonstrate that explicit planning plays a crucial role in the system's performance.



[17] [Context-faithful Prompting for Large Language Models](https://arxiv.org/pdf/2303.11315.pdf) 2023.03

Large language models (LLMs) encode parametric knowledge about world facts and have shown remarkable performance in knowledge-driven NLP tasks. However, their reliance on parametric knowledge may cause them to overlook contextual cues, leading to incorrect predictions in context-sensitive NLP tasks (e.g., knowledge acquisition tasks). In this paper, we seek to assess and enhance LLMs' contextual faithfulness in two aspects: **knowledge conflict and prediction with abstention**. We demonstrate that LLMs' faithfulness can be significantly improved using c**arefully designed prompting strategies.** In particular, we identify **opinion-based prompts** and **counterfactual demonstrations** as the most effective methods. Opinion-based prompts **reframe the context as a narrator's statement and inquire about the narrator's opinions**, while **counterfactual demonstrations use instances containing false facts to improve faithfulness in knowledge conflict situations**. Neither technique requires additional training. We conduct experiments on three datasets of two standard NLP tasks, **<u>machine reading comprehension and ==relation extraction==</u>**, and the results demonstrate significant improvement in faithfulness to contexts.



[==18==] [Large Language Model Is Not a Good Few-shot Information Extractor, but a Good Reranker for Hard Samples](https://arxiv.org/pdf/2303.08559.pdf) 2023.03

Review the details of this paper, espacially the prompt template it used.



[19] [Towards Integration of Discriminability and Robustness for Document-Level Relation Extraction](https://arxiv.org/pdf/2304.00824.pdf) 2023.04

Document-level relation extraction (DocRE) predicts relations for entity pairs that rely on long-range context-dependent reasoning in a document. As a typical multi-label classification problem, DocRE faces the challenge of effectively distinguishing a **small set of positive relations from the majority of negative ones**. This challenge becomes even more difficult to overcome when there exists a significant number of annotation errors in the dataset. In this work, we aim to achieve better integration of both the discriminability and robustness for the DocRE problem. Specifically, we first design an effective loss function to endow high discriminability to both probabilistic outputs and internal representations. We innovatively customize entropy minimization and supervised contrastive learning for the challenging multi-label and long-tailed learning problems. To ameliorate the impact of label errors, we equipped our method with a novel negative label sampling strategy to strengthen the model robustness. In addition, we introduce two new data regimes to mimic more realistic scenarios with annotation errors and evaluate our sampling strategy. Experimental results verify the effectiveness of each component and show that our method achieves new state-of-the-art results on the DocRED dataset, its recently cleaned version, Re-DocRED, and the proposed data regimes.

( the biased distribution of datasets. focus on the loss function)



[20] [Pre-training Transformers for Knowledge Graph Completion](https://arxiv.org/pdf/2303.15682.pdf) 2023.03

Learning transferable representation of knowledge graphs (KGs) is challenging due to the heterogeneous, multi-relational nature of graph structures. Inspired by Transformer-based pretrained language models' success on learning transferable representation for texts, we introduce a novel inductive **KG representation model (iHT) for KG completion by large-scale pre-training**. iHT consists of **a entity encoder** (e.g., BERT) and **a neighbor-aware relational scoring function** both parameterized by Transformers. We first pre-train iHT on a large KG dataset, Wikidata5M. Our approach achieves new state-of-the-art results on matched evaluations, with a relative improvement of more than 25% in mean reciprocal rank over previous SOTA models. When further fine-tuned on smaller KGs with either entity and relational shifts, pre-trained iHT representations are shown to be transferable, significantly improving the performance on FB15K-237 and WN18RR.

[21] 



## 1.Hallucination

[1] Tracing and Removing Data Errors in Natural Language Generation Datasets

>  Recent work has identified noisy and misan- notated data as a core cause of hallucinations and unfaithful outputs in Natural Language Generation (NLG) tasks. Consequently, iden- tifying and removing these examples is a key open challenge in creating reliable NLG systems. In this work, we introduce a frame- work to identify and remove low-quality training instances that lead to undesirable outputs, such as faithfulness errors in text summarization. We show that existing approaches for error tracing, such as gradient-based influence measures, do not perform reliably for detecting faithfulness errors in summarization. We overcome the drawbacks of existing error tracing methods through a new, **contrast-based estimate that compares undesired generations to human-corrected outputs**. Our proposed method can achieve a mean average precision of 0.91 across synthetic tasks with known ground truth and can **achieve a two-fold reduction in hallucinations on a real entity hallucination evaluation on the NYT dataset**.
>
>  从消除低质量数据的角度来减少hallucination。同时对language model for generating datasets的工作也有关，NYT dataset就是NER的。
>
>  没太看懂Synthetic Hallucinations和Extrinsic hallucinations in the NYT dataset有什么区别。
>
>  但是它用来generate的LLM是BART。

[2] [Check Your Facts and Try Again: Improving Large Language Models with External Knowledge and Automated Feedback](https://arxiv.org/pdf/2302.12813.pdf) Microsoft Research, 3rd March.

> Large language models (LLMs), such as ChatGPT, are able to generate human-like, fluent responses for many downstream tasks, e.g., task-oriented dialog and question answering. However, applying LLMs to real-world, mission-critical applications remains challenging mainly due to their tendency to generate hallucinations and their inability to use external knowledge. This paper proposes **a LLM-Augmenter system**, which **augments a black-box LLM with a set of plug-and-play modules**. Our system makes the LLM generate responses **grounded in external knowledge**, e.g., stored in task-specific databases. It also **iteratively revises** LLM prompts to improve model responses **using feedback generated by utility functions**, e.g., **the factuality score of a LLM-generated response**. The effectiveness of LLM-Augmenter is empirically validated **on two types of scenarios, task-oriented dialog and open-domain question answering**. LLM-Augmenter significantly reduces ChatGPT's hallucinations **without sacrificing the fluency and informativeness of its responses**.【这个fluency和informativeness是怎么判断的？】 We make the source code and models publicly available.
>
> 可以借鉴的：iteratively revises the prompts to improve responses using feedback generated by utility functions.相当于用我们这里的check module打分之后不行->regenerate。【这一篇和2022年Google的revise很像。。】
>
> 关键在于：怎么整合的外部知识？怎么automatically generate feedback？怎么revise？如果feedback是错的怎么办。。

Methodology overview: LLM-AUGMENTER **first retrieves evidence from external knowledge** (e.g., Web or task-specific datasets) and, **if necessary** 【如何判断是不是necessary？】, further consolidates evidence by **linking retrieved raw evidence with related contex**t (e.g., information of the entity "2013 Los Angeles Galaxy”) and **performing reasoning to form evidence chains** (e.g., table-passage in the figure). Then, LLM- AUGMENTER **queries a fixed LLM** (i.e., ChatGPT in our study) **using a prompt that contains the consolidated evidence** for ChatGPT to generate a candidate response grounded in external knowledge (evidence). LLM-AUGMENTER then **verifies the candidate response** e.g., <u>*by checking whether it hallucinates evidence*</u>. If so,LLM-AUGMENTER **generates a feedback message** (e.g.,about the team “C.S.D. Municipal”). The **message is used to revise the prompt to query ChatGPT again**. The process **iterates until a candidate response passes the verification** and is sent to the user.

从这里来看，这个LLM-AUGMENTER的作用包括：retrieve evidence from external knowledge; 判断是否necessary；如果necessary，还要consolidate eveidence by linking retrieved raw evidence with related context; 然后用reasoning能力来提供一个evidence chain；再query这个fixed LM，通过一个包含了consolidated evidence的prompt; 然后还要verify是不是halucinates evidence；最后再revise prompt并且query again。

In addition to proposing LLM-AUGMENTER, to be detailed in Section 2, we make the following contributions. We perform an empirical study to validate the effectiveness of LLM-AUGMENTER using two tasks, **information seeking dialog** (Section 3) and **open-domain Wiki question answering (Wiki QA)** (Section 4). The study shows that LLM-AUGMENTER significantly reduces Chat-GPT's hallucinations **without sacrificing the flu-ency and informativeness** of its generated responses. For example, on the dialog task of customer service, **<u>human evaluation</u>** shows LLM- AUGMENTER improve ChatGPT by 32.3% in Usefulness (measuring the groundedness or hallucination of model responses) and 12.9% in Humanness (measuring the fluency and informativeness of model responses). The Wiki QA task is extremely challenging to ChatGPT in that answering these questions often requires **multi-hop reasoning** to **piece together information of various modalities**【是怎么融入多模态的信息的？】 scattered across different documents. Our results show that although **the closed-book ChatGPT** performs poorly and often hallucinates, LLM- AUGMENTER substantially improves the factuality score of the answers (absolute +10% in F1) by grounding ChatGPT's responses in consolidated external knowledge and automated feedback.

居然还用了hidden Markov chain...但是这些reward要怎么算呢？？？

总体上为一个iterative framework打了一个挺好的样，是和self check GPT不一样的思路，一个是iteratively revise，一个是sampling based hallucination detection。目标不一样，

[3] [Interpretable Visual Question Answering Referring to Outside Knowledge](https://arxiv.org/pdf/2303.04388.pdf) 2023

<img src="/Users/tt/Library/Application Support/typora-user-images/截屏2023-04-02 下午5.34.46.png" alt="截屏2023-04-02 下午5.34.46" style="zoom:50%;" />

Fig. 2: An overview of our model. The model answers the input question Qn and generates a corresponding explanation with human-friendly natural language sentence Wn based on the input image In. We newly introduce the image caption set Cn and outside knowledge set Kn into the proposed model to refer to various information during the generation process. 该模型回答输入的问题Qn，并根据输入的图像In生成相应的带有人类友好自然语言句子Wn的解释。我们在模型中新引入了图像标题集Cn和外部知识集Kn，以便在生成过程中参考各种信息。

[4] [A Survey on Automated Fact-Checking](https://arxiv.org/abs/2108.11896) ACL,2022

In this survey, we present a comprehensive and up-to-date survey of automated fact-checking, unifying various definitions developed in previous research into a common framework. We begin by defining the three stages of our fact-checking framework—claim detection, evidence retrieval, and claim verification, the latter consisting of verdict prediction and justification production. We then give an overview of the existing datasets and modeling strategies, taxonomizing these and contextualizing them with respect to our framework. We finally discuss key research challenges that have been addressed, and give directions for challenges that we believe should be tackled by future research. We accompany the survey with a repository,[1](javascript:;) which lists the resources mentioned in our survey.


Figure 2 shows a NLP framework for automated fact-checking consisting of three stages: (i) *claim detection* to **identify claims that require verification**; (ii) *evidence retrieval* to f**ind sources supporting** or refuting the claim; and (iii) *claim verification* to **assess the veracity of the claim** based on **the retrieved evidence**. Evidence retrieval and claim verification are sometimes tackled as a single task referred to as *factual verification*, while claim detection is often tackled separately. Claim verification can be decomposed into two parts that can be tackled separately or jointly: *verdict prediction*, where claims are assigned truthfulness labels, and *justification production*, where explanations for verdicts must be produced.

reference [here](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00454/109469/A-Survey-on-Automated-Fact-Checking)

[5] [Plausible May Not Be Faithful: Probing Object Hallucination in Vision-Language Pre-training](https://arxiv.org/pdf/2210.07688) EACL, 2023

> Large-scale vision-language pre-trained (VLP) models are prone to hallucinate non-existent visual objects when generating text based on visual information. In this paper, we systematically study the object hallucination problem from three aspects. First, we examine **recent state-of-the-art VLP models, showing that they still hallucinate frequently**,【看一下是怎么研究的，以及在什么任务上研究的（是否包括VQA任务）】 and models achieving better scores on standard metrics (e.g., CIDEr) could be more unfaithful【如何衡量faithful？】. Second, we investigate **how different types of image encoding in VLP** influence hallucination, including **region-based, grid-based, and patch-based**. Surprisingly, we find that patch-based features perform the best and **smaller patch resolution** yields a non-trivial reduction in object hallucination. Third, we decouple various VLP objectives and demonstrate that **token-level image-text alignment** and **controlled generation** are crucial to reducing hallucination. Based on that, we propose a simple yet effective **VLP loss** named ObjMLM to **further mitigate object hallucination**. Results show that it reduces object hallucination by up to 17.4% when tested on two benchmarks (COCO Caption for in-domain and NoCaps for out-of-domain evaluation).
>
> 所以好的对齐在本质上能够减少hallucination

One major type of hallucination in VLP is known as object hallucination (Rohrbach et al., 2018), where models **generate texts containing non-existent or inaccurate objects from the input images**. ...=>所以这篇工作本质上解决的type1 hallucination。它是从pre-train的角度来考虑的，这是有基于alignment的representation learning的原理作为依据来支撑的。

Finally, we propose a simple yet effective new vision-language pre-training loss, namely objectmasked language modeling (ObjMLM), to further mitigate object hallucination by enhancing the alignment and restriction between text tokens and visual objects during generation. [Code and evaluation setups](https://github.com/wenliangdai/VLP-Object-Hallucination) are released.

Overall,our contributions are three-fold:
1)This is the first paper that **systematically studies state-of-the-art VLP models on the object hallucination problem**, proving that it is still far from resolved and previous methods that improve standard metrics may reflect in worse hallucination.
2)We thoroughly investigate the influence of **different VLP losses and image encoding methods** on object hallucination. Our findings could be valuable for future work to build more responsible VLP systems.
3)We present **a new pre-training objective ObjMLM** to mitigate object hallucination.Experimental results show that it reduces object hallucination by 17.4% without introducing extra training data.


so the task is image captioning. => **what is the difference of hallucination between image captioning and VQA?**

[6] [Models See Hallucinations: Evaluating the Factuality in Video Captioning](https://arxiv.org/abs/2303.02961) 2023

> Video captioning aims to describe events in a video with natural language. In recent years, many works have focused on improving captioning models' performance. However, like other text generation tasks, it risks introducing factual errors not supported by the input video. These factual errors can seriously affect the quality of the generated text, sometimes making it completely unusable. Although **factual consistency** has received much research attention in text-to-text tasks (e.g., **<u>summarization</u>**), it is **less studied in the context of vision-based text generation**. In this work, we conduct **a detailed human evaluation** of the **factuality** in video captioning and **collect two annotated factuality datasets**. We find that 57.0% of the model-generated sentences **have factual errors,** indicating it is a severe problem in this field. However, existing evaluation metrics are mainly based on **n-gram matching** and show little correlation with human factuality annotation. We further propose **a weakly-supervised, model-based factuality metric FactVC**, which outperforms previous metrics on factuality evaluation of video captioning. The datasets and metrics will be released to promote future research for video captioning.

Task: video captioning

Main contribution: Conduct a detailed human evaluation of the facuality; collect two annotated factuality datasets; propose a weakly-supervised, model-based factuality metirc FactVC

captioning models（和我们想探究的还是挺不一样的）

[7] [A Survey of Hallucination in Natural Language Generation](https://arxiv.org/abs/2202.03629) Association for Computing Machinery, 2022

Section 5 - hallucination mitigation methods:

section 12 - hallucination in vision-language generation:


section 12.1 object hallucination in image captioning



definition, mitigation(solutions), metrics(evaluate the VLM's reliability regarding to hallucination)

section 12.2  Hallucination in Other VL Tasks



not including the mitigation and metrics yet.

section 12.3 future direction 

For future research on the hallucination problem in VL, we summarize three promising directions. Firstly, hallucination in VL is still in the early stage. There is a **lack of empirical and theoretical analyses in many tasks,** such as **visual storytelling, visual commonsense reasoning, video captioning**, etc.【不过video captioning也有了，就在上面的 [6]】 Secondly, **more effective evaluation metrics** are needed. For example,although CHAIR can automatically evaluate the degree of object hallucination in image captioning, it requires a pre-defined list of object categories, which does not generalize well. Furthermore, for the hallucination types discussed in Section 12.2, currently **there is no automatic metric** 【这意味着，对于VQA的任务，还没有有关evaluate hallucination的方式。。所以我们可能需要考虑OpenQA那边的任务，而我看到这里的时候，就想到可以结合Microsoft最近的工作，[Submitted on 24 Feb 2023 (v1), last revised 8 Mar 2023 (this version, v3)]
Check Your Facts and Try Again: Improving Large Language Models with External Knowledge and Automated Feedback】. Therefore, we cannot perform quantitative evaluations for them. Thirdly, we believe how to perform controlled generation [28,154] with visual grounding is a promising direction to mitigate hallucination in VL.

所以我们需要考虑一下可行性的问题，VQA这边是否缺少相关的公开的visualLM？经常在image captioning中用到的CLIP这种模型也没办法用在vqa的generation里。

[8] [SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models](https://arxiv.org/abs/2303.08896) 2023

> Generative Large Language Models (LLMs) such as GPT-3 are capable of generating highly fluent responses to a wide variety of user prompts. However, LLMs are known to hallucinate facts and make non-factual statements which can undermine trust in their output. Existing fact-checking approaches either require access to token-level output probability distribution (which may not be available for systems such as ChatGPT) or external databases that are interfaced via separate, often complex, modules. In this work, we propose "SelfCheckGPT", a simple sampling-based approach that can be used to fact-check black-box models in a zero-resource fashion, i.e. without an external database. SelfCheckGPT leverages the simple idea that if a LLM has knowledge of a given concept, sampled responses are likely to be similar and contain consistent facts. However, for hallucinated facts, stochastically sampled responses are likely to diverge and contradict one another. We investigate this approach by using GPT-3 to generate passages about individuals from the WikiBio dataset, and manually annotate the factuality of the generated passages. We demonstrate that SelfCheckGPT can: i) detect non-factual and factual sentences; and ii) rank passages in terms of factuality. We compare our approach to several existing baselines and show that in sentence hallucination detection, our approach has AUC-PR scores comparable to grey-box methods, while SelfCheckGPT is best at passage factuality assessment.

[9] [Errors are Useful Prompts: Instruction Guided Task Programming with Verifier-Assisted Iterative Prompting](https://arxiv.org/pdf/2303.14100.pdf) 2023

Generating low-level robot task plans from high-level natural language instructions remains a challenging problem. Although large language models have shown promising results in generating plans, the accuracy of the output remains unverified. Furthermore, the lack of domain-specific language data poses a limitation on the applicability of these models. In this paper, we propose CLAIRIFY, a novel approach that combines automatic iterative prompting with program verification to ensure programs written in data-scarce domain-specific language are syntactically valid and incorporate environment constraints. Our approach provides effective guidance to the language model on generating structured-like task plans by incorporating any errors as feedback, while the verifier ensures the syntactic accuracy of the generated plans. We demonstrate the effectiveness of CLAIRIFY in planning chemistry experiments by achieving state-of-the-art results. We also show that the generated plans can be executed on a real robot by integrating them with a task and motion planner.

In this work, we propose to address the verification and data-scarcity challenges. We introduce CLAIRIFY1, a framework that translates natural language into a domain-specific structured task plan usingan automated iterative verification technique to ensure the plan is syntactically valid in the target DSL (Figure 1) by providing the LLM a description of the target language. Our model also takes into account environment constraints if provided. The generated structured- language-like output is evaluated by our verifier, which checks for syntax correctness and for meeting environment constraints. The syntax and constraint errors are then fed back into the LLM generator to generate a new output. This iterative interaction between the generator and the verifier leads to grounded syntactically correct target language plans.

[10] [Language Models as Zero-Shot Planners: Extracting Actionable Knowledge for Embodied Agents](https://arxiv.org/pdf/2201.07207) 2022

Can world knowledge learned by large language models (LLMs) be used to act in interactive environments? In this paper, we investigate the possibility of grounding high-level tasks, expressed in natural language (e.g. "make breakfast"), to a chosen set of actionable steps (e.g. "open fridge"). While prior work focused on learning from explicit step-by-step examples of how to act, we surprisingly find that if pre-trained LMs are large enough and prompted appropriately, they can effectively decompose high-level tasks into mid-level plans without any further training. However, the plans produced naively by LLMs often cannot map precisely to admissible actions. We propose a procedure that conditions on existing demonstrations and semantically translates the plans to admissible actions. Our evaluation in the recent VirtualHome environment shows that the resulting method substantially improves executability over the LLM baseline. The conducted human evaluation reveals a trade-off between executability and correctness but shows a promising sign towards extracting actionable knowledge from language models. Website at [this https URL](https://huangwl18.github.io/language-planner)

[11] [IQA: Visual Question Answering in Interactive Environments](https://prior.allenai.org/projects/iqa#paper) *CVPR, 2018*

We introduce Interactive Question Answering (IQA), the task of answering questions that require an autonomous agent to interact with a dynamic visual environment. IQA presents the agent with a scene and a question, like: “Are there any apples in the fridge?” The agent must navigate around the scene, acquire visual understanding of scene elements, interact with objects (e.g. open refrigerators) and plan for a series of actions conditioned on the question. Popular reinforcement learning approaches with a single controller perform poorly on IQA owing to the large and diverse state space. We propose the Hierarchical Interactive Memory Network (HIMN), consisting of a factorized set of controllers, allowing the system to operate at multiple levels of temporal abstraction. To evaluate HIMN, we introduce IQUAD V1, a new dataset built upon AI2-THOR, a simulated photo-realistic environment of configurable indoor scenes with interactive objects. IQUAD V1 has 75,000 questions, each paired with a unique scene configuration. Our experiments show that our proposed model outperforms popular single controller based methods on IQUAD V1.

> 作者在文章中介绍了IQA（交互式回答问题），这是一项回答问题的任务，与以往我所见到的论文不同的是，它需要**一个自主的代理人与动态的视觉环境进行交互**。给一个场景和一个问题，agent在场景中导航，去获得对场景元素的视觉理解，**与场景中的物体进行交互，为问题计划一系列行动情况，以备更好地回答问题**。而由于所给的场景是非常巨大，其中的位置空间多种多样，所以当前流行的强化学习方法和单一的控制器使IQA表现不佳。作者在本文中提出了一种分层交互记忆网络（HLMN），它由一组分解的控制器组成，允许在多个时间抽象级别上工作。本文的模型实现是由一个基于一种新的数据集IQUAD V1的具有交互对象的可变形室内场景的模拟照片真实感环境。其中数据集IQUAD V1中有75000个问题，每一个问题都有一个独特的场景组合。本文中提出的模型优于IQUAD V1的流行单控制器。

IQA poses several key challenges in addition to the ones posed by VQA. First, the agent must be able to navigate through the environment. Second, it must acquire an understanding of its environment including objects, actions, and affordances. Third, the agent must be able to interact with objects in the environment (such as opening the microwave, picking up books, etc.). Fourth, the agent must be able to plan and execute a series of actions in the environment conditioned on the questions asked of it. 除了VQA所带来的挑战外，IQA还带来了几个关键的挑战。首先，agents必须能够在环境中导航。第二，它必须获得对其环境的理解，包括对象、行动和承受力。第三，代理人必须能够与环境中的物体进行交互（如打开微波炉，拿起书本等）。第四，代理人必须能够在环境中计划并执行一系列以向其提出的问题为条件的行动。

To address these challenges, we propose the Hierarchical Interactive Memory Network (HIMN). Akin to past works on **hierarchical reinforcement learning**, HIMN is factorized into **a hierarchy of controllers**, allowing the system to operate, learn, and reason across multiple time scales while simultaneously reducing the complexity of each individual subtask. A high level controller, referred to as t**he Planner chooses the task to be performed** (for example, navigation, manipulation, answering etc.) and **generates a command for the chosen task**. Tasks specified by the Planner are **executed by a set of low level controllers** (Navigator, Manipulator, Detector, Scanner and Answerer) which return control to the Planner when a task termination state is reached. 为了应对这些挑战，我们提出了分层交互式记忆网络（HIMN）。与过去的分层强化学习工作类似，HIMN被分解成一个控制器的层次，允许系统在多个时间尺度上运行、学习和推理，同时降低每个单独子任务的复杂性。一个被称为 "planner "的高级控制器选择要执行的任务（例如，导航、操纵、回答等），并为所选择的任务生成一个命令。planner指定的任务由一组低级控制器（导航器、操纵器、检测器、扫描器和应答器）执行，这些控制器在达到任务终止状态时将控制权返回给规划者。

与以往的关于分层强化学习的工作相似，‘HIMN被分解为一个层次结构控制网络，允许系统在多个时间尺度上操作、学习和推理。同时降低每个单个子任务的复杂性。高级控制器选择要执行的任务（如导航/操作/回答等等），并为所选任务生成命令。低级控制器（导航器、操作器、检测器、扫描器和应答器）将控制返回给规划器。这些子任务是相对独立的，故我们可以独立地对每个控制器进行预训练。

有几种问题类型要求agent记住它在哪里以及它看到了什么。例如，这间房子里有几个枕头？需要一个代理在房间里面导航，记录它遇到的枕头的数量。对于足够复杂的空间，代理就需要将这些信息保存在内存中很长时间，这就促使一个代理填充的显示外部内存表示。这个记忆必须是空间和语义的，这样它就可以代表一个地方。

从运营成本、规模以及研究重现性的角度来看，在现实世界中去模拟模型中的交互是不可行的，所以，一个可行的选择就是在真实的模拟环境中训练的评估这些代理。为此，作者在本文中使用了A12-THOR[35]——一个基于IQUAD V1数据集的可定制室内场景仿真环境。IQUAD V1中的75000道选择题中的每一道题都有一个独特的场景配置。

本文的贡献总共包括三个部分：a.提出**交互式问题回答**，回答要求代理人与动态环境交互的问题的任务；b.提出层次交互记忆网络，一个问题回答模型分解成一个高层次规划器、一组低层次控制器和一个丰富的语义空间记忆；c.提出了一种新的递归层来表示这个内存；d.使用了一个新的数据集IQUAD V1来完成。

[12] [SimVQA: Exploring Simulated Environments for Visual Question Answering](https://arxiv.org/abs/2203.17219) CVPR, 2022

We explore using synthetic computer-generated data to fully control the visual and language space, allowing us to provide more diverse scenarios for VQA. By exploiting 3D and physics simulation platforms, we provide a pipeline to generate synthetic data to expand and replace type-specific questions and answers without risking the exposure of sensitive or personal data that might be present in real images. We quantify the effect of synthetic data in real-world VQA benchmarks and to which extent it produces results that generalize to real data. 我们探索使用计算机生成的合成数据来完全控制视觉和语言空间，使我们能够为VQA提供更多样化的场景。通过利用三维和物理模拟平台，我们提供了一个生成合成数据的管道，以扩大和取代特定类型的问题和答案，而不需要冒暴露可能存在于真实图像中的敏感或个人数据的风险。我们量化了合成数据在真实世界VQA基准中的效果，以及它产生的结果在多大程度上可以推广到真实数据。

**<u>F-SWAP to Leverage the Synthetic Data</u>**. We also propose Feature Swapping (F-SWAP), a domain alignment method where we randomly switch object-level features during training to make a VQA model more domain invariant. The motivation for feature swapping relies in observing that in all three datasets we can find similar types of objects and configurations but the appearance of the objects might differ. Our goal with feature swapping is then to randomly replace during the training the object-level features for some of the objects with the features for an equivalent object from another domain.



[13] [Task and Motion Planning with Large Language Models for Object Rearrangement](https://arxiv.org/abs/2303.06247) 2023.03

Key: compare with the recent work by NAVIDA

[14] [Planning with Large Language Models via Corrective Re-prompting](https://arxiv.org/pdf/2211.09935.pdf) 2022.11

Extracting the common sense knowledge present in Large Language Models (LLMs) offers a path to designing intelligent, embodied agents. Related works have queried LLMs with a wide-range of contextual information, such as goals, sensor observations and scene descriptions, to generate high-level action plans for specific tasks; however these approaches often involve human intervention or additional machinery to enable sensor-motor interactions. In this work, we propose a **prompting-based strategy** for **extracting executable plans** from an LLM, which leverages a novel and readily-accessible source of information: precondition errors. Our approach assumes that **actions are only afforded execution in certain contexts**, i.e., implicit preconditions must be met for an action to execute (e.g., a door must be unlocked to open it), and that **the embodied agent has the ability to determine if the action is/is not executable in the current context** (e.g., detect if a precondition error is present). When an agent is unable to execute an action, our approach re-prompts the LLM with precondition error information to extract an executable corrective action to achieve the intended goal in the current context. We evaluate our approach in the VirtualHome simulation environment on 88 different tasks and 7 scenes. We evaluate different prompt templates and compare to methods that naively re-sample actions from the LLM. Our approach, using precondition errors, improves executability and semantic correctness of plans, while also reducing the number of re-prompts required when querying actions.

Key: compare to [9] [Errors are Useful Prompts: Instruction Guided Task Programming with Verifier-Assisted Iterative Prompting](https://arxiv.org/pdf/2303.14100.pdf) , what's the difference?



## 2.LLM and IE

[1] [Structured prompt interrogation and recursive extraction of semantics (SPIRES): A method for populating knowledge bases using zero-shot learning](https://arxiv.org/pdf/2304.02711.pdf) 2023.04

Creating knowledge bases and ontologies is a time consuming task that relies on a manual curation. AI/NLP approaches can assist expert curators in populating these knowledge bases, but current approaches rely on extensive training data, and are not able to populate **arbitrary complex nested knowledge schema**s.
Here we present Structured Prompt Interrogation and Recursive Extraction of Semantics (SPIRES), **a Knowledge Extraction approach** that relies on **the ability of Large Language Models (LLMs)** to perform zero-**shot learning (ZSL)** and **general-purpose query answering** from **flexible prompts** and **return information conforming to a specified schema**. Given **a detailed, user-defined knowledge schema and an input text**, SPIRES **recursively performs prompt interrogation** against GPT-3+ to obtain a set of responses matching the provided schema. SPIRES uses existing ontologies and vocabularies to provide identifiers for all matched elements.
We present examples of use of SPIRES **in different domains**, including extraction of <u>food recipes, multi-species cellular signaling pathways, disease treatments, multi-step drug mechanisms, and chemical to disease causation graphs.</u> Current SPIRES accuracy **is comparable to the mid-range of existing Relation Extraction (RE) methods**, but has the advantage of easy customization, flexibility. and, crucially, the ability to perform new tasks in the absence of any training data. This method supports a general strategy of leveraging the language interpreting capabilities of LLMs to assemble knowledge bases, assisting manual knowledge curation and acquisition while supporting validation with publicly-available databases and ontologies external to the LLM.
SPIRES is available as part of the open source OntoGPT package: https://github.com/monarch-initiative/ontogpt 

This makes use of so-called *instruction prompts* in Large Language Models (LLMs) such as GPT-4.



[2] [TagGPT: Large Language Models are Zero-shot Multimodal Taggers]() 2023.04

Use LLMs to extract informations based on the tags of a websites?



[3] [Unified Text Structuralization with Instruction-tuned Language Models](https://arxiv.org/pdf/2303.14956.pdf) 2023.03

Text structuralization is one of the important fields of natural language processing (NLP) **consists of information extraction (IE) and structure formalization**. However, current studies of text structuralization suffer from a shortage of manually annotated high-quality datasets from different domains and languages, which require specialized professional knowledge. In addition, **most IE methods are designed for a specific type of structured data**, e.g., entities, relations, and events, making them hard to generalize to others. In this work, we propose a simple and efficient approach to **instruct large language model (LLM) to extract a variety of structures from texts**. More concretely, we **add a prefix and a suffix instruction to indicate the desired IE task and structure type**, respectively, before feeding the text into a LLM. Experiments on two LLMs show that this approach can enable language models to perform <u>comparable with other state-of-the-art methods on datasets of a variety of languages and knowledge</u>, and can <u>*generalize to other IE sub-tasks via changing the content of instruction*</u>. Another benefit of our approach is that it can help researchers to **build datasets in low-source and domain-specific scenarios**, e.g., fields in finance and law, with low cost.



Key: the prompt template design?



[4] [Large Language Model Is Not a Good Few-shot Information Extractor, but a Good Reranker for Hard Samples](https://arxiv.org/abs/2303.08559) 2023.03

Key: the empirical findings, the prompt templates



[5] [ICL-D3IE: In-Context Learning with Diverse Demonstrations Updating for Document Information Extraction](https://arxiv.org/pdf/2303.05063.pdf) 2023.03

Large language models (LLMs), such as GPT-3 and ChatGPT, have demonstrated remarkable results in various natural language processing (NLP) tasks with in-context learning, which involves inference based on a few demonstration examples. Despite their successes in NLP tasks, no investigation has been conducted to assess the ability of LLMs to perform document information extraction (DIE) using in-context learning. Applying LLMs to DIE poses two challenges: **the modality and task gap**. To this end, we propose a simple but effective in-context learning framework called ICL-D3IE, which enables LLMs to perform DIE with different types of demonstration examples. Specifically, we extract the most difficult and distinct segments from **hard training documents** as **hard demonstrations** for benefiting all test instances. We design **demonstrations describing relationships** that enable LLMs to **<u>understand positional relationships</u>**. We introduce **formatting demonstrations** for **easy answer extraction**. Additionally, **the framework improves diverse demonstrations by updating them iteratively**. Our experiments on <u>three widely used benchmark datasets</u> demonstrate that the ICL-D3IE framework enables GPT-3/ChatGPT to achieve superior performance when compared to previous pre-trained methods fine-tuned with full training in **both the in-distribution (ID) setting and in the out-of-distribution (OOD) setting**.



KEY: demonstrations about <u>positional relationships</u>? the iteration steps? which demonstration works good on out-of-distribution settings? check out the demonstrations in the paper. AND it got a totally different result when compared to [4]. 

limitations: only consider the demonstration desgin for DIE.



[6] [Exploring the Feasibility of ChatGPT for Event Extraction](https://arxiv.org/abs/2303.03836) 2023.03

It suggest that LLM performs badly on Event Extraction.

Key: Check the design of prompts. and do they consider the iterative prompting?



[7] Semantic Enhanced Knowledge Graph for Large-Scale Zero-Shot Learning 2022.12

Zero-Shot Learning has been a highlighted research topic in both vision and language areas. Recently, most existing methods adopt structured knowledge information to model explicit correlations among categories and use deep graph convolutional network to propagate information between different categories. However, it is difficult to add new categories to existing structured knowledge graph, and deep graph convolutional network suffers from over-smoothing problem. In this paper, we provide **a new semantic enhanced knowledge graph** that contains **both expert knowledge and categories semantic correlation**. Our semantic enhanced knowledge graph can further enhance the correlations among categories and make it easy to absorb new categories. To **propagate information on the knowledge graph**, we propose a novel Residual Graph Convolutional Network (ResGCN), which can effectively alleviate the problem of over-smoothing. Experiments conducted on the widely used large-scale ImageNet-21K dataset and AWA2 dataset show the effectiveness of our method, and establish a new state-of-the-art on zero-shot learning. Moreover, our results on the large-scale ImageNet-21K with various feature extraction networks show that our method has better generalization and robustness.

> It seems more related to graph representation learning



[8] [ImPaKT: A Dataset for Open-Schema Knowledge Base Construction](https://arxiv.org/abs/2212.10770) 2022.12

Language model pretraining has simultaneously enabled great strides in natural language inference, reasoning about entailment and implication in free text. These advances motivate us to construct ImPaKT, a dataset for open-schema information extraction, consisting of around 2500 text snippets from the C4 corpus, in the shopping domain (product buying guides), **professionally annotated with extracted attributes, types, attribute summaries** (attribute schema discovery from idiosyncratic text), **many-to-one relations** between **compound and atomic attributes**, and **implication relations**. We release this data in hope that it will be useful in fine tuning semantic parsers for information extraction and knowledge base construction across a variety of domains. We evaluate the power of this approach by fine-tuning the open source UL2 language model on a subset of the dataset, extracting a set of implication relations from a corpus of product buying guides, and conducting human evaluations of the resulting predictions.



Key: imlication relations? How to generate such datasets?

Maybe their procedure of **collecting implication relations** could give some hints on how to formulate a pipeline-based task for LLMs in KB Construction tasks.

>  Parsing implication relationships shares many similarities with the **natural language inference (NLI)** (MacCartney, 2009; Bowman et al., 2015) and **recognizing textual entailment (RTE)** (Baroni et al., 2012; Dagan et al., 2013), wherein entailment (implicative) relations between sentences or concepts are classified, a problem of longstanding interest to the NLU community. Implication relationships themselves are also useful for performing multi-hop reasoning and discovering causal chains between attributes, an active area of research in KG completion (Das et al., 2016,2017).
>
>  3.2 implication
>
>  In the context of our dataset, we define an implication between **two attributes** A and B in **the same category** as a judgment that the existence of attribute A describing a product also leads us to believe that attribute B describes the product, grounded in a specific textual statement or statements. That is, **<u>the text itself states the implication exists,</u>** we do **not infer the implication**. For example, the snippet "the chair has a memory foam cushion to promote good posture" expresses an implication.
>  Either the left or right hand side of an implication might have several different components in conjunction, disjunction, or even some other fuzzy qualifiers. This is why we **do not require our notion of attributes to correspond to one** single crisp concept, or else we would be unable to annotate many implications of interest, such as these white Reebok tennis shoes are comfortable on a hot day, which has individual aspects of color, brand, and type all bundled up into one side of the implication in an ambiguous manner.

<img src="/Users/daishitong/Library/Application Support/typora-user-images/Screenshot 2023-04-09 at 1.42.27 AM.png" alt="Screenshot 2023-04-09 at 1.42.27 AM" style="zoom:50%;" />

In its workflow, we can see it trains a generator for implications and use it again.



limitation: manually annotate the implications and only considering the simplest implications

[9] [Joint Open Knowledge Base Canonicalization and Linking](https://arxiv.org/pdf/2212.01207.pdf) 2022.12

Open Information Extraction (OIE) methods extract a large number of OIE triples (noun phrase, relation phrase, noun phrase) from text, which compose large Open Knowledge Bases (OKBs). However**, noun phrases (NPs) and relation phrases (RPs) in OKBs are not canonicalized** and **often appear in different paraphrased textual variants,** which leads to **redundant and ambiguous facts**. To address this problem, there are two related tasks: **OKB canonicalization (i.e., convert NPs and RPs to canonicalized form)** and **OKB linking (i.e., link NPs and RPs with their corresponding entities and relations in a curated Knowledge Base (e.g., DBPedia).** These two tasks are tightly coupled, and one task can benefit significantly from the other. However, they have been studied in isolation so far. In this paper, we explore the task of joint OKB canonicalization and linking for the first time, and propose a novel framework JOCL based on factor graph model to make them reinforce each other. JOCL is flexible enough to combine different signals from both tasks, and able to extend to fit any new signals. A thorough experimental study over two large scale OIE triple data sets shows that our framework outperforms all the baseline methods for the task of OKB canonicalization (OKB linking) in terms of average F1 (accuracy).

[10] [STAGE: Span Tagging and Greedy Inference Scheme for Aspect Sentiment Triplet Extraction](https://arxiv.org/pdf/2211.15003.pdf) 2022.12

仍然用的是transformer-based model，但是它有classifier和inference

[11] [Learning with Silver Standard Data for Zero-shot Relation Extraction](https://arxiv.org/pdf/2211.13883.pdf) 2022.12

The superior performance of supervised relation extraction (RE) methods heavily relies on a large amount of <u>gold standard data</u>. Recent zero-shot relation extraction methods **converted the RE task to other NLP tasks** and used off-the-shelf models of these NLP tasks to **directly perform inference** on the test data **without using a large amount of RE annotation data**. A potentially valuable by-product of these methods is the large-scale silver standard data. However, there is no further investigation on the use of potentially valuable silver standard data. In this paper, we propose to **first detect a small amount of clean data from silver standard data** and then use **the selected clean data to finetune the pretrained model**. We then use the finetuned model to **infer relation types**. We also propose **a class-aware clean data detection module** to **consider class information when selecting clean data.** The experimental results show that our method can outperform the baseline by 12% and 11% on TACRED and Wiki80 dataset in the zero-shot RE task. By using extra silver standard data of different distributions, the performance can be further improved.

[similar to the Open KB Dataset proposed by Google Research above.]

KEY: finetune plus infer model ( other: finetune plus generate ) ; How to realize the class-awareness?

[12] [FolkScope: Intention Knowledge Graph Construction for Discovering E-commerce Commonsense](https://arxiv.org/pdf/2211.08316.pdf) 2022.11

In this paper, we present FolkScope, an intention knowledge graph construction framework, to reveal the structure of humans' minds about purchasing items on e-commerce platforms such as Amazon. As commonsense knowledge is usually ineffable and not expressed explicitly, it is challenging to perform any kind of information extraction. Thus, we propose a new approach that leverages **the generation power of large-scale language models** and **human-in-the-loop annotations** to **semi-automatically construct the knowledge graph**. We annotate a large amount of assertions for both plausibility and typicality of an intention that can explain a purchasing or co-purchasing behavior, where the intention can be an open reason or a predicate falling into one of 18 categories aligning with ConceptNet, e.g., IsA, MadeOf, UsedFor, etc. Then we populate the annotated information to all automatically generated ones, and further structurize the assertions using pattern mining and conceptualization to form more condensed and abstractive knowledge. We evaluate our knowledge graph using both intrinsic quality measures and a downstream application, i.e., recommendation. The comprehensive study shows that our knowledge graph can well model e-commerce commonsense knowledge and can have many potential applications.

<img src="/Users/daishitong/Library/Application Support/typora-user-images/Screenshot 2023-04-09 at 1.57.17 AM.png" alt="Screenshot 2023-04-09 at 1.57.17 AM" style="zoom:50%;" />

Key: for topic 1

[13] [IELM: An Open Information Extraction Benchmark for Pre-Trained Language Models](https://arxiv.org/abs/2210.14128) 2022.10

We introduce a new open information extraction (OIE) benchmark for pre-trained language models (LM). Recent studies have demonstrated that pre-trained LMs, such as BERT and GPT, may store linguistic and relational knowledge. In particular, LMs are able to **answer ``fill-in-the-blank'' questions** when given a pre-defined relation category. **Instead of focusing on pre-defined relations**, we create **an OIE benchmark aiming** to fully examine the open relational information present in the pre-trained LMs. We accomplish this by turning pre-trained LMs into zero-shot OIE systems. Surprisingly, pre-trained LMs are able to obtain competitive performance on both standard OIE datasets (CaRB and Re-OIE2016) and two new large-scale factual OIE datasets (TAC KBP-OIE and Wikidata-OIE) that we establish via distant supervision. For instance, the zero-shot pre-trained LMs outperform the F1 score of the state-of-the-art supervised OIE methods on our factual OIE datasets without needing to use any training sets. Our code and datasets are available at [this https URL](https://github.com/cgraywang/IELM)

[14] [Zero-Shot On-the-Fly Event Schema Induction](https://arxiv.org/pdf/2210.06254.pdf) 2022.10-2023

In this paper, our goal is to allow creating schemas on-the-fly by taking as input **only the name of the complex event of interest** (like a "pandemic outbreak" or an “armed robbery”). To avoid manually collecting many documents on the topic of the schema,we utilize pre-trained text generators, e.g., GPT-3 (Brown et al., 2020), to obtain documents of diverse genres on the desired topic (examples presented in Fig. 2). These documents are then processed to **extract pertinent information from which a schema is constructed**. The fact that we do not collect any data makes our learning framework zero-shot since we do not rely on any human-collected articles or example schemas. [ key: how to check? verify? ]

In addition to *eliminating the need to collect data* (???), we also made the information extraction process faster by implementing new and efficient methods for **identifying temporal and hierarchical relations between events mentioned** in the text. These two steps are the most time **consuming in the process of schema induction** and could take up to 2 hours each using state-of-the-art models proposed by Zhou et al.(2021); Wang et al. (2021). Sending the whole text as input instead of two sentences at each time, our proposed model shortens the inference time significantly to several minutes without enduring a major loss in performance.

 The process of generating texts is explained in Section §3, and the process of extracting rele- vant and salient information is described in Section §4, then we introduce the construction of schema graphs in Section §5. To evaluate our zero-shot schema generator we conduct experi- ments on a benchmark dataset for schema induction,LDC2020E25, and provide a new dataset for further evaluation called Schema-11. Additionally, we design a subject-matter expert Turing test, a.k.a. Feigenbaum test (Feigenbaum,2003), to determine whether our algorithm could mimic experts' re- sponse. We also demonstrate that documents generated by GPT-3 are informative and useful for the task of schema induction. The experiments and re- sults are presented in Section §6. The contributions of our work include:

Key: for topic1

[15] [Rethinking the Event Coding Pipeline with Prompt Entailment](https://arxiv.org/pdf/2210.05257.pdf) 2022,10

For monitoring crises, political events are extracted from the news. The large amount of unstructured full-text event descriptions makes a case-by-case analysis unmanageable, particularly for low-resource humanitarian aid organizations. This creates a demand to classify events into event types, a task referred to as event coding. Typically, domain experts craft an event type ontology, annotators label a large dataset and technical experts develop a supervised coding system. In this work, we propose PR-ENT, a new event coding approach that is more flexible and resource-efficient, while maintaining competitive accuracy: **first, we extend an event description** such as "Military injured two civilians'' **<u>by a template</u>**, e.g. "People were [Z]" and prompt a pre-trained (cloze) language model to fill the slot Z. Second, we select answer candidates Z* = {"injured'', "hurt"...} by treating the event description as premise and the filled templates as hypothesis in a textual entailment task. This allows domain experts to draft the codebook directly as labeled prompts and interpretable answer candidates. This human-in-the-loop process is guided by our interactive codebook design tool. We evaluate PR-ENT in several robustness checks: perturbing the event description and prompt template, restricting the vocabulary and removing contextual information.

终于看到一个在event方面的了，还是generation 数据集的


