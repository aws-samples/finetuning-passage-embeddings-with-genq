# Fine-tuning Passage Embeddings with GenQ

State-of-the-art semantic search applications like [extractive question answering](https://docs.pinecone.io/docs/extractive-question-answering) and [Retrieval Augmented Generation](https://arxiv.org/abs/2005.11401) (RAG) have gained significant attention recently, allowing users to query a large document corpus using natural language questions. One of the most important components in these systems is the **embedding model**, which maps input text into a vector, providing a numeric representation of the semantic meaning of the text. An effective model is tuned to place passages with similar meaning in similar locations in the embedding space. In a process known as **retrieval**, this space is efficiently interrogated to find the most relevant documents given an input query.

In practice, general-purpose language models do not perform well on this retrieval task, particularly when dealing with documents related to a specialized domain, like medicine or law. The typical solution is to "fine-tune" a generic model to the particular task and field of interest by feeding it thousands of labeled training examples. For a semantic search engine, the most valuable dataset would be human-generated (query, passage) pairs.

For example, this excerpt from the [Sagemaker Developer Guide](https://docs.aws.amazon.com/sagemaker/latest/dg/train-model.html) contains an appropriate answer for the following query:

```
Query: What instance type should I use for an NLP model?

Relevant Passage: Choose the right instance type and infrastructure management tools for your use case. You can start from a small instance and scale up depending on your workload. For training a model on a tabular dataset, start with the smallest CPU instance of the C4 or C5 instance families. For training a large model for computer vision or natural language processing, start with the smallest GPU instance of the P2, P3, G4dn or G5 instance families.
```

However, it would be extremely expensive and time consuming to have humans generate questions for thousands of documents. In this series of notebooks, we will demonstrate an approach called [**GenQ**](https://www.sbert.net/examples/unsupervised_learning/query_generation/README.html), in which we leverage a separate model to automatically generate so-called "synthetic" queries for our dataset. For more details about GenQ, see the [NIPS publication](https://arxiv.org/abs/2104.08663).

## Getting Started

This demonstration is broken into 3 notebooks which are designed to be completed in order.

1. [01-Data-Preparation](01-Data-Preparation.ipynb): initialization, dataset preparation, generating synthetic queries
2. [02-Finetune-and-Deploy-Model](02-Finetune-and-Deploy-Model.ipynb): fine-tuning the embedding model and deploying it to a Sagemaker endpoint
3. [03-Applications-and-Evaluation](03-Applications-and-Evaluation.ipynb): comparing the fine-tuned model to a baseline and using it for RAG and extractive QA

Throughout the demo, we will leverage frameworks like [LangChain](https://python.langchain.com/en/latest/) and [Hugging Face](https://huggingface.co/) to simplify and operationalize the workflow. We will also use [Amazon SageMaker](https://aws.amazon.com/sagemaker/) for fully-managed and scalable inference (batch & real-time) and training.

These notebooks are designed to be run on a [SageMaker Notebook](https://docs.aws.amazon.com/sagemaker/latest/dg/nbi.html) using the `conda_pytorch_p39` kernel, which will greatly simplify configuration and initialization. 

## License

This repository uses the [MIT-0](LICENSE) license. 

Although this repository is released under the MIT-0 license, its example dataset uses the third party [PubMed summarization](https://github.com/armancohan/long-summarization) repository. The PubMed summarization project’s licensing includes the Apache 2.0 license.

## Developers

See our [Code of Conduct](CODE_OF_CONDUCT.md) and [Contributing Guidelines](CONTRIBUTING.md) for information on how to contribute to this repository.

## References

Cohan, A., Dernoncourt, F., Kim, D. S., Bui, T., Kim, S., Chang, W., & Goharian, N. (2018, June). A Discourse-Aware Attention Model for Abstractive Summarization of Long Documents. Proceedings of the 2018 Conference of the North AMerican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers), 615–621. doi:10.18653/v1/N18-2097

[LangChain Documentation](https://python.langchain.com/en/latest/)

[SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/)

Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. ArXiv, abs/1910.01108.

Thakur, N., Reimers, N., Ruckl'e, A., Srivastava, A., & Gurevych, I. (2021). BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models. ArXiv, abs/2104.08663.



