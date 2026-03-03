# AI and Technology Knowledge Base

## Artificial Intelligence

Artificial Intelligence (AI) is the simulation of human intelligence in machines. AI systems can perform tasks like learning, reasoning, problem-solving, perception, and language understanding.

### Types of AI
- **Narrow AI**: Designed for specific tasks like image recognition or chess playing.
- **General AI**: Hypothetical AI with human-like intelligence across all domains.
- **Machine Learning**: A subset of AI where systems learn from data without being explicitly programmed.

### Deep Learning
Deep learning uses neural networks with many layers to learn from large amounts of data. It powers modern applications like image recognition, speech recognition, and natural language processing.

## Large Language Models (LLMs)

Large Language Models are AI models trained on massive text datasets. They can generate human-like text, answer questions, summarize content, and assist with coding.

### Popular LLMs
- **LLaMA** by Meta: Open-source model available in 1B, 3B, 8B, and larger sizes.
- **GPT** by OpenAI: Powers ChatGPT and is widely used in commercial applications.
- **Mistral**: Efficient open-source model known for strong performance at smaller sizes.
- **DeepSeek**: Strong reasoning model from Chinese AI lab DeepSeek.
- **Qwen**: Multilingual model from Alibaba, supports many languages.

### Quantization
Quantization reduces the number of bits used to represent model weights. This allows large models to run on consumer hardware. GGUF format is commonly used for quantized models with llama.cpp.

## RAG (Retrieval Augmented Generation)

RAG combines a retrieval system with a language model. Instead of relying only on what the model learned during training, RAG fetches relevant documents and uses them as context for answering questions.

### How RAG Works
1. Documents are chunked into smaller pieces.
2. Each chunk is converted into an embedding (numerical vector).
3. Embeddings are stored in a vector database like ChromaDB.
4. When a user asks a question, the question is also embedded.
5. The most similar document chunks are retrieved.
6. The LLM uses those chunks as context to generate an answer.

### Benefits of RAG
- Reduces hallucinations by grounding answers in real documents.
- Can answer questions about private or recent data not in the training set.
- More accurate and reliable than pure LLM responses.

## ChromaDB

ChromaDB is an open-source vector database used to store and query embeddings. It supports local and cloud deployments and is commonly used in RAG pipelines.

## Python and Development

### Poetry
Poetry is a Python dependency management tool. It handles virtual environments and package installation. Common commands:
- `poetry install` — install all dependencies
- `poetry add <package>` — add a new package
- `poetry show` — list installed packages

### Streamlit
Streamlit is a Python framework for building web apps quickly. It is commonly used for data science and ML demos. Run a Streamlit app with:
```
streamlit run app.py
```

## Cricket

### 2011 ICC Cricket World Cup
India won the 2011 ICC Cricket World Cup. The tournament was hosted by India, Sri Lanka, and Bangladesh. India defeated Sri Lanka in the final held in Mumbai. MS Dhoni captained the Indian team and hit the winning six. Sachin Tendulkar was the leading run scorer of the tournament.

### MS Dhoni
Mahendra Singh Dhoni is a former Indian cricketer and one of the greatest captains in cricket history. He was born on July 7, 1981, in Ranchi, India. He is known for his calm demeanor under pressure, finishing abilities, and wicketkeeping skills. He led India to victory in the 2007 T20 World Cup, 2011 ODI World Cup, and 2013 Champions Trophy. He retired from international cricket in August 2020.

## Health and Wellness

Regular exercise, balanced diet, and adequate sleep are the three pillars of good health. Adults should aim for at least 150 minutes of moderate exercise per week. Drinking enough water, managing stress, and maintaining social connections also contribute to overall well-being.

## History

### World War II
World War II lasted from 1939 to 1945. It involved most of the world's nations and was the deadliest conflict in human history. The Allied powers (USA, UK, USSR, France) defeated the Axis powers (Germany, Italy, Japan). The war ended in Europe on May 8, 1945 (VE Day) and in the Pacific on September 2, 1945 (VJ Day).

### Indian Independence
India gained independence from British rule on August 15, 1947. Mahatma Gandhi led the non-violent independence movement. Jawaharlal Nehru became the first Prime Minister of independent India.

## Space Exploration

### Moon Landing
NASA's Apollo 11 mission landed humans on the Moon on July 20, 1969. Neil Armstrong was the first human to walk on the Moon, followed by Buzz Aldrin. Michael Collins remained in lunar orbit.

### Mars Exploration
NASA's Perseverance rover landed on Mars on February 18, 2021. It is searching for signs of ancient microbial life and collecting rock samples. The Ingenuity helicopter, which traveled with Perseverance, became the first powered aircraft to fly on another planet.