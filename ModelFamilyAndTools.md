***

# AI Model Landscape & Tools: 2025 Overview

## 1. Company & Model Family Rankings

| Rank | Company | Model Family | Key Strengths |
| :--- | :--- | :--- | :--- |
| **1** | **OpenAI** | **GPT-4 family** | Broad usage, multimodal, ecosystem first |
| **2** | **Google (DeepMind)** | **Gemini** | Multimodal, long context, integrated in Google products |
| **3** | **Anthropic** | **Claude** | Safety/alignment focus, large context windows |
| **4** | **Stability AI** | **StableLM** | Open-source, multilingual, cost-efficient |
| **5** | **Meta** | **LLaMA** | Research-friendly, efficient, open release |
| **6** | **Alibaba** | **Qwen** | Enterprise + multilingual + Asia focus |

---

## 2. Detailed Model Specifications

| Rank | Model Family | Company | Variants / Sizes | Approx. File Size / Notes | Use Cases & Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **#1** | **gpt-oss** | OpenAI | 20B, 120B | 120B variant ~60-70 GiB (quantised) | High-end reasoning/agentic workflows, tool use, local/cloud deployment |
| **#2** | **qwen3** | Alibaba Group | 0.6B – 235B | 0.6B ~small; 235B very large | [Qwen+1](https://www.alibabacloud.com/help/en/model-studio/what-is-qwen-llm) General purpose LLM family (dense & MoE), multilingual, scalable |
| **#2** | **qwen3-vl** | Alibaba Group | 2B – 235B | Sizes similar to above | Vision-Language multimodal (image+text) tasks |
| **#3** | **deepseek-r1**| DeepSeek AI (China) | 1.5B – 671B | 1.5B ~1-2 GiB; full 671B huge | High reasoning model family, MoE. [Built In+2](https://builtin.com/artificial-intelligence/deepseek-r1) [prompthub.us+2](https://www.prompthub.us/models/deepseek-reasoner-r1) |
| **#4** | **gemma3** | Google DeepMind | 270M – 27B | 270M ~400 MB (text only) | [blog.google+1](https://blog.google/technology/developers/gemma-3/) [Google AI for Developers+1](https://developers.googleblog.com/en/gemma-3-on-mobile-and-web-with-google-ai-edge/) Efficient models, vision/multimodal support |
| **#4** | **embedding** | Google DeepMind | ~300M | Smaller embedding model | Embedding/semantic-search vector tasks |
| **#5** | **glm-4.6** | Microsoft | ~14B | — | Open model for reasoning, coding, agentic tasks |
| **#5** | **llama3.1** | Meta Platforms | 8B, 70B, 405B | — | State-of-the-art open model from Meta |
| **#5** | **llama3.2** | Meta Platforms | 1B, 3B | — | Smaller sibling versions for lighter hardware |
| **#5** | **llava** | (collab) | 7B – 34B | — | Multimodal model (vision + language) |
| **#6** | **mistral** | Mistral AI | 7B | — | Smaller open-model milestone |
| **#2** | **qwen2.5** | Alibaba Group | 0.5B – 72B | — | [DeepLearning.AI](https://www.deeplearning.ai/the-batch/issue-328/) Multilingual model, large context support (128K tokens) |
| **#5** | **phi3** | Microsoft | 3.8B, 14B | — | Lightweight open models for reasoning/coding |

---

## 3. Scenario Comparison Summary

| Scenario | Best Model/Family | Why |
| :--- | :--- | :--- |
| **Self-hosting (open weights)** | DeepSeek-R1; OpenAI’s gpt-oss | Weights are publicly released; can be deployed locally. |
| **Enterprise cloud** | Google Gemini; OpenAI GPT-4 family | Managed API with enterprise tooling and security (SLA/support). |
| **Multimodal (Vision/Audio)** | Google Gemini 2.5; OpenAI GPT-4o | Broad modality support (image, audio, video). |
| **Language modeling for code** | Microsoft Copilot stack | Strong in code generation & IDE integration (uses OpenAI/Claude). |
| **Multilingual / Non-English** | Alibaba Qwen family | Strong multilingual models and wide size range. |

---

## 4. Comprehensive Model Catalog

### Open Source (Open Weights)
| Company | Model Family | Sizes | Approx. File Size | Primary Use Cases | Key Features |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **OpenAI** | gpt-oss | 20B, 120B | 20–60 GB | Reasoning, agentic workflows | General-purpose reasoning |
| **Alibaba** | qwen3 / vl / 2.5 | 0.6B – 235B | 0.5–140 GB | General, multilingual, multimodal | High versatility |
| **DeepSeek**| deepseek-r1 | 1.5B – 671B | 1 GB – 400 GB | High reasoning, MoE | High efficiency at scale, long context |
| **Google** | gemma3 | 270M – 27B | 0.4–20 GB | Efficient text + multimodal | Compact and energy efficient |
| **Microsoft**| glm-4.6 / phi3 | 3.8B – 14B | 3–10 GB | Reasoning, coding | Lightweight reasoning |
| **Meta** | llama3 / 3.1 / 3.2 | 1B – 405B | 1–250 GB | General-purpose, code | Strong open baseline |
| **Mistral** | mistral | 7B | ~4 GB | Chat, reasoning | Fast inference |
| **Collab** | llava | 7B – 34B | 6–25 GB | Vision-language reasoning | Strong multimodal baseline |

### Production API (Proprietary/Closed)
| Company | Model Family | Availability | Key Features |
| :--- | :--- | :--- | :--- |
| **OpenAI** | GPT‑4 / GPT‑4o / GPT‑4.1 | Cloud-hosted | SOTA multimodal intelligence, reasoning |
| **Anthropic**| Claude 3 / 3.5 | Cloud-hosted | Safe aligned model, summarization, writing |
| **Google** | Gemini 1.5 / 2.0 | Cloud-hosted | Strong multimodal capabilities, code, vision |
| **Microsoft**| Copilot (GPT-4 based) | Integrated | Integrated into Office/VSCode for productivity |
| **Alibaba** | Qwen Commercial API | Cloud-hosted | Enterprise AI assistant, localized, multilingual |
| **DeepSeek** | DeepSeek Chat API | Cloud-hosted | High-reasoning API, localizable Chinese/English |
| **Perplexity**| Perplexity LLM | Cloud-hosted | Fast factual retrieval, conversational search |

---

## 5. Development Tools

### **Ollama**
* **Python Library:** [https://github.com/ollama/ollama-python](https://github.com/ollama/ollama-python)
* **Function:** Interesting model handling for local development.

### **Cursor (IDE)**
* **[Cursor 2.0 is Here](https://cursor.com/blog/2-0)**
* **[Cursor Composer 1](https://cursor.com/blog/2-0)**: Cursor’s new agent model, designed for software engineering intelligence and speed.
* **Multi-Agent Interface:** A new chat interface with voice input and a layout purpose-built for working with agents.
* **Run Agents in Parallel:** Run multiple models in parallel, locally with worktrees or in the cloud.

---

## 6. Local LLM Management Tools Comparison

| Tool | Supported Platforms | Hardware Requirements | Key Features | Free vs Paid | Best For |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **[LM Studio](https://lmstudio.ai/download)** | macOS, Win, Linux | 16+ GB RAM, GPU (4GB+ VRAM) | Local LLM management, chat, API server | Free (Personal); Paid (Enterprise) | Devs wanting full control & local API endpoints |
| **[Msty Studio](https://docs.msty.app)** | Win, Mac, Linux (Web/Desktop) | GPU Compute 5.0+ | RAG, Personas, Workflows, Tool Integration | Free "Forever" + Aurum/Enterprise | Users building assistants, RAG apps & workflows |
| **[AnythingLLM](https://anythingllm.com/)** | Win, Mac, Linux | Basic desktop; GPU for large models | Document upload (PDF/CSV), "No setup" ease | Free (Desktop); Paid hosting optional | Non-technical users needing document chat |
| **[Ollama](https://github.com/ollama/ollama)** | Win, Mac, Linux (CLI+GUI) | 8-16GB RAM, GPU (8GB+ VRAM) | CLI/GUI execution, flexible model management | Free / Open-Source | Hobbyists & Devs experimenting with models |
| **[H2O LLM Studio](https://github.com/h2oai/h2o-llmstudio)**| Win, Mac, Linux | High RAM/VRAM (for training) | No-code fine-tuning, visual training pipeline | Open-Source (Free); Enterprise options | Data scientists training/deploying models |
