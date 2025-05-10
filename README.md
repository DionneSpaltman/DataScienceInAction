# Luiss x Deloitte
Course: Data Science in Action


## **Objectives of the Project**

### **Scientific Papers**
- Writers and the universities they are affiliated with.
- **Abstract**: The key ingredient for successfully carrying out the project. Abstracts are not subject to strict copyright rules.
- **Body of the paper**: The proper content of the paper. **We are not interested in this due to copyright restrictions**.

---

## **Goal: Build a Recommendation System for Scientific Papers**
- The objective is to build a **recommendation system (RS)** for academic papers, similar to **Amazon, Spotify, YouTube, or Netflix**, but for scientific research.
- Given a paper on a particular topic, the system should **recommend relevant papers**.
- The **Objectives Slide** is crucial as it contains all the project goals.

### **Understanding the Role of Abstracts**
- Abstracts provide a **broad overview** of the research question, methodologies, and results.
- **No technical details** are included, making them useful for topic extraction and similarity analysis.

### **Topic Discovery**
- Using **topic modeling techniques**, we can extract research themes from abstracts.
- This enables **matching user queries** with relevant papers.
- A typical workflow:
  1. Run **topic modeling** on a dataset of abstracts.
  2. Extract the key **topics** in each paper.
  3. Compute **similarity** between papers to provide recommendations.

---

## **Techniques for Topic Modeling**
### **Classic Approach: Bag of Words Model**
- A simple way to **embed text** and extract word frequencies.
- Apply **LDA (Latent Dirichlet Allocation)** instead of PCA for topic modeling.
- Example: LDA in Python can be implemented in **just 3 lines of code**.

### **Advanced Approaches**
- **Hugging Face** provides thousands of pre-trained models for **natural language processing (NLP)**.
- At the core of modern **Large Language Models (LLMs)** is the **Transformer architecture**.
- Explore **BERTopic**, a powerful topic modeling tool available on Hugging Face.

#### **Hugging Face Resources**
- Hugging Face: [https://huggingface.co/](https://huggingface.co/)
- BERTopic Model: [https://huggingface.co/MaartenGr/BERTopic_Wikipedia](https://huggingface.co/MaartenGr/BERTopic_Wikipedia)

---

## **Key Considerations for Model Selection**
- Some models can be **huge** (e.g., DeepSeek has **685 billion parameters**).
- The base **BERT model** has **167 million parameters**.
- **More parameters ≠ better accuracy** in practice.
- Choose a model **best suited for the task**, rather than one that is too large to run efficiently.

---

## **Alternative Approach: Embeddings for Similarity**
- Convert text into **vector representations** using **sentence embeddings**.
- **Measure similarity** between abstracts using:
  - **Euclidean Distance**
  - **Cosine Similarity** (preferred)
- Example: Use **Sentence Transformers** from Hugging Face.

### **Pros & Cons**
- **Embeddings capture meaning**, making them better than **Bag of Words**.
- However, **embeddings are not interpretable**, unlike topic modeling.

---

## **Data Collection**
### **Why OpenAlex?**
- Unlike other datasets, **Deloitte is not providing one**, so we need to **collect our own data**.
- **We will use OpenAlex, NOT Semantic Scholar** (Semantic Scholar is not working).
- No API key is required for OpenAlex.

### **Search Criteria**
- Topic: **AI for Pricing and Promotion in GDO**
- Time Range: **Last 10 years (2015-2025)**
- **Avoid outdated papers!**

### **Using OpenAlex API**
- API Documentation: [https://docs.openalex.org/api-entities/works](https://docs.openalex.org/api-entities/works)
- OpenAlex APIs function **similar to SQL queries**.

### **Abstract Handling**
- Some papers do not have plaintext abstracts.
- Instead, they use **abstract_inverted_index**, which stores **words with their positions**.
- This requires **reconstruction**.

### **Key Features for Recommendation System**
- **Authorships**: Authors are **important indicators** of topic relevance.
- **Journals (Sources)**: Papers from similar journals should be ranked higher.
- **Keywords & Topics**: OpenAlex assigns **AI-generated keywords** to papers.

---

## **Building the Recommendation System**
### **Key Tasks**
- **Topic Modeling**: Identify themes using NLP.
- **Text Summarization**: (Not necessary since abstracts are already short).
- **Entity Recognition**: Extract metadata like author names and institutions.
- **Content-Based Recommendation**: Find similar papers based on abstracts.
- **Semantic Similarity Analysis**: Compare papers using vector embeddings.

### **Suggested NLP Libraries**
- **Transformers & Pre-trained models** on Hugging Face.
- **Scikit-learn** for similarity evaluation.

### **Downloading Papers Efficiently**
- **Do not download everything at once** – OpenAlex has a large dataset.
- Use **Pyalex** to fetch relevant papers in bulk:
  - GitHub Repository: [https://github.com/J535D165/pyalex](https://github.com/J535D165/pyalex)

---

## **Project Explanation (From Slides)**
### **Intelligent Analysis for Academic Publications**
- The goal is to **revolutionize** the analysis of academic papers using **NLP**.
- Manual paper analysis is **slow, inefficient, and unscalable**.
- The **proposed solution** improves **speed and accuracy** of information retrieval.


### **Deliverables**
- **Code**: A well-documented implementation of the recommendation system.
- **Technical Report**: Explanation of methodologies, results, and insights.
- **(Bonus) Demo Interface**: A user-friendly way to explore results.

---

## **References**
- OpenAlex API: [https://docs.openalex.org/api-entities/works](https://docs.openalex.org/api-entities/works)
- Hugging Face Models: [https://huggingface.co/models](https://huggingface.co/models)
- Pyalex Library: [https://github.com/J535D165/pyalex](https://github.com/J535D165/pyalex)
- Scikit-learn: [https://scikit-learn.org](https://scikit-learn.org)
