# Awesome AI Engineer Interview Questions [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

> 105 real interview questions for **AI / LLM engineer** roles — covering LLMs, transformers, prompting, RAG, fine-tuning, agents, evaluation, inference, production and safety — each with a concise answer. Maintained by [Skillumen](https://www.skillumen.com).

An AI-engineering interview is not a trivia quiz — it is *"have you actually shipped this?"* Every question below is one an interviewer really asks in 2026, grouped by topic, with a one-line answer to anchor your thinking. For the full answer, worked examples and a chance to **rehearse it out loud in an AI voice mock**, follow the topic links to Skillumen.

<p align="center"><a href="https://www.skillumen.com/?utm_source=github&utm_medium=awesome-list&utm_campaign=ai-interview"><b>▶ Practice these free — 30-day LLM interview bootcamp →</b></a></p>

**Updated 2026-08-31** · 105 questions · 11 topics · answers link to full write-ups.

## Contents

- [🧠 LLM Foundations](#-llm-foundations)
- [⚙️ Transformers & Architecture](#️-transformers--architecture)
- [✍️ Prompting & In-Context Learning](#️-prompting--in-context-learning)
- [🔎 RAG — Retrieval-Augmented Generation](#-rag--retrieval-augmented-generation)
- [🎯 Fine-tuning & Alignment](#-fine-tuning--alignment)
- [🤖 Agents & Tool Use](#-agents--tool-use)
- [📊 Evaluation](#-evaluation)
- [🚀 Inference & Serving](#-inference--serving)
- [🛠️ Production, Ops & Cost](#️-production-ops--cost)
- [🛡️ Safety & Security](#️-safety--security)
- [🔬 Scaling & the Research Frontier](#-scaling--the-research-frontier)

---

## 🧠 LLM Foundations

**Q — What actually is a Large Language Model, in plain terms?**  `Foundations`  
An LLM is a next-token predictor: it reads text as tokens and repeatedly guesses the most likely next piece.  It does pattern-completion, not database lookups, using billions of weights learned from huge amounts of internet text.  

**Q — Why does a raw LLM ramble or ask more questions, while ChatGPT just answers? What's the difference between a base and an instruction-tuned model?**  `Foundations`  
A base model only continues text, so a question can make it write more questions.  An instruction-tuned model is the same network sent through extra training (SFT then alignment) so it follows instructions and chats helpfully — same brain, different finishing school.  

**Q — What's the difference between training and inference for an LLM — and does the model learn from my prompts?**  `Foundations`  
A model has two lives: training is where it learns its weights (slow, costly, done rarely), and inference is where it uses those frozen weights to answer you (fast, but on every request).  Your prompts do __not__ update the weights — cost and latency live in inference, and new knowledge comes from RAG or re-training, not chatting.  

**Q — Interviewer: what's the difference between an embedding model and a generative model, and when do you use each?**  `Foundations`  
They do two different jobs people often mix up.  An embedding model turns text into a fixed vector of numbers that captures meaning — you use it to FIND things.  

**Q — How do you go from a pile of internet text to a helpful assistant like ChatGPT? Walk me through the stages.**  `Foundations`  
A helpful assistant is raised in three stages that each change the model's weights: pretraining teaches raw language and world knowledge, instruction tuning teaches it to follow instructions, and alignment (RLHF/DPO) teaches it which answers people actually prefer.  prompting and RAG are different — they steer the model at __use__ time without training it at all.  

**Q — Why do LLMs confidently make things up, and what can a beginner do about it?**  `Foundations`  
An LLM predicts plausible next words, not verified facts — so it can be fluent, confident, and simply wrong.  This is called hallucination, and simple fixes are grounding it with RAG, asking for citations, lowering temperature, and verifying anything high-stakes.  

**Q — What is the context window and the 'lost in the middle' problem?**  `Foundations`  
The context window is the maximum tokens a model can see at once — its __working memory__.  Models also tend to use the start and end better than the middle (the lost in the middle effect).  

**Q — What are logits, and how does softmax turn them into probabilities?**  `Foundations`  
Logits are the model's __raw scores__ for every possible next token; softmax turns them into probabilities that sum to 1, which sampling then draws from.  

➡️ **Full answers, diagrams & practice:** [LLM Foundations on Skillumen →](https://www.skillumen.com/blog/llm-interview-questions.html)

---

## ⚙️ Transformers & Architecture

**Q — What is tokenization and why can't an LLM just read raw text?**  `Foundations`  
Tokenization is the step that chops text into small pieces called tokens and turns each one into a number the model can do math on.  

**Q — What is an embedding and what property makes it useful?**  `Foundations`  
An embedding turns each token into a list of numbers (a vector) that captures its meaning — and similar meanings end up close together.  

**Q — Explain self-attention using Q, K, V intuition.**  `Foundations`  
Self-attention lets every token look at all the other tokens and pull in information from the ones that matter most to it.  

**Q — Why do transformers need positional encoding?**  `Foundations`  
Attention by itself doesn't know word order, so we have to add position information on purpose.  

**Q — What does 'autoregressive' generation mean?**  `Foundations`  
Autoregressive generation means the model writes one token at a time, each time looking at everything it has produced so far.  

**Q — Contrast temperature, top-k, and top-p sampling.**  `Foundations`  
Three dials that change how the model picks the next token — trading off 'safe and predictable' against 'creative and varied'.  

**Q — What loss does next-token prediction use, and what is it measuring?**  `Foundations`  
The model is trained with cross-entropy loss, which measures how surprised it was by the real next word — less surprise means lower loss.  

**Q — Why did transformers replace RNNs/LSTMs?**  `Foundations`  
RNNs read text one word at a time and forget long-range detail; transformers read the whole sequence at once with attention, so they're faster to train and better at long distances.  

**Q — What's the difference between encoder, decoder, and encoder-decoder models?**  `Foundations`  
Encoder models __understand__ it (BERT), decoder models generate left-to-right (GPT), and encoder-decoder models do both (T5) — modern LLMs are almost all decoder-only.  

**Q — Compare RoPE, ALiBi, and learned positional encodings.**  `Foundations`  
There are several ways to tell a transformer word order: fixed sinusoidal, learned embeddings, RoPE (__rotation__), and ALiBi (distance penalty).  RoPE is the 2026 default.  

**Q — What do residual connections and normalization do, and why pre-norm?**  `Foundations`  
Residual connections add a layer's input back to its output so signals survive deep stacks; normalization keeps the numbers stable.  Modern models normalize *before* each sublayer (pre-norm) with RMSNorm.  

**Q — Why did LLMs move from ReLU to GELU and SwiGLU?**  `Foundations`  
Activation functions add the __non-linearity__ that lets networks learn complex patterns.  LLMs moved from ReLU to smoother GELU, and now to SwiGLU in the feed-forward layers.  

**Q — Compare greedy, beam search, sampling, and speculative decoding.**  `Foundations`  
At each step the model scores every possible next word (logits).  Decoding is the rule for turning those scores into the one word it actually writes — always grab the top pick, roll a weighted die for variety, or use a helper model to go faster.  

**Q — What's inside a single transformer block?**  `Foundations`  
One transformer block does two things: attention (words share information) and a FFN (a little network that thinks about each word on its own).  Each is wrapped with a residual connection (a shortcut back to the input) and a step that keeps the numbers steady.  

**Q — What is multi-head attention, and why did GQA/MQA appear?**  `Applied`  
Multi-head attention runs several readers (heads) over the same sentence at once, so the model can follow several kinds of connections between words at the same time.  GQA and MQA then let those readers share notes to keep memory (the KV cache) from ballooning.  

**Q — Why is FlashAttention faster without changing the math?**  `Advanced`  
FlashAttention computes the exact same attention but in small tiles, so it never writes the huge N×N score matrix to slow memory — making it much faster.  

**Q — What are Mamba/SSMs and linear-attention models?**  `Advanced`  
State Space Models (Mamba) and linear attention replace the transformer's quadratic attention with a mechanism that scales __linearly__ with length — cheaper for very long sequences.  

**Q — How does a Mixture-of-Experts layer reduce compute?**  `Advanced`  
A Mixture of Experts holds many specialist sub-networks but a router sends each token to only a few — giving huge capacity at a small per-token cost.  

**Q — How does MoE routing work and why is load balancing hard?**  `Advanced`  
In Mixture of Experts, a __router__ picks which experts handle each token.  The hard part is load balancing — stopping a few experts from getting all the traffic while others sit idle.  

➡️ **Full answers, diagrams & practice:** [Transformers & Architecture on Skillumen →](https://www.skillumen.com/blog/llm-interview-questions.html)

---

## ✍️ Prompting & In-Context Learning

**Q — What is in-context learning and how do prompting techniques exploit it?**  `Foundations`  
in-context learning means the model picks up a task just from what you put in the prompt — no training needed.  Adding a few worked examples (few-shot) and asking it to reason step by step (chain-of-thought) make its answers far more accurate.  

**Q — What are the key prompt engineering techniques?**  `Applied`  
A toolkit of prompt patterns: zero-shot, few-shot, chain-of-thought, self-consistency, and tree-of-thoughts — each trading more tokens for __better reasoning__.  

**Q — A teammate 'improved' a prompt and quietly broke three features. How do you stop that from happening?**  `Applied`  
Treat prompts like code: prompts as code means version control, regression tests, and an eval gate in CI — so a prompt change is reviewed and tested before it ships, not pushed live on a hunch.  

➡️ **Full answers, diagrams & practice:** [Prompting & In-Context Learning on Skillumen →](https://www.skillumen.com/blog/llm-interview-questions.html)
📎 **Related deep-dives:** [Generative AI Interview Questions](https://www.skillumen.com/blog/generative-ai-interview-questions.html)

---

## 🔎 RAG — Retrieval-Augmented Generation

**Q — What problem does RAG solve and how does it work?**  `Applied`  
RAG lets the model look things up in your documents before answering, so it stays accurate and can use fresh or private information.  

**Q — How should you chunk documents for RAG?**  `Applied`  
Chunking means cutting your documents into small pieces the system can search.  If you cut them badly, search fails.  

**Q — How do you choose and improve embedding models for retrieval?**  `Applied`  
The embedding model is the thing that turns text into numbers and decides which pieces count as 'similar' when you search.  Pick one that fits your topic and language.  

**Q — When do you use RAG vs fine-tuning vs long context?**  `Applied`  
Three ways to give a model knowledge: RAG (look it up), fine-tuning (bake it in), and long context (paste it in).  They solve different problems and often combine.  

**Q — What techniques improve basic RAG?**  `Applied`  
Basic RAG (the setup that looks facts up before answering) often grabs the wrong pieces of text.  A few upgrades fix this: __rewriting the search__, HyDE, GraphRAG, and agentic RAG that searches over and over in a loop until it has enough.  

**Q — What is a vector database and how does ANN search work?**  `Applied`  
A vector database stores embeddings and finds the closest ones to a query almost instantly using ANN tricks like HNSW.  

**Q — Why add a reranker and hybrid search to RAG?**  `Applied`  
First cast a wide net with hybrid search (BM25 keywords + vectors), then a reranker reads each candidate closely and re-sorts for accuracy.  

**Q — Plain RAG fails on 'how are these two things connected?' questions. What retrieval upgrades fix that?**  `Applied`  
Beyond basic chunk-and-search, 2026 retrieval adds graphrag (follow links in a knowledge graph for multi-hop questions), semantic routing (send each query to the right source), and self-reflection (rewrite and retry when results look weak).  

**Q — Your RAG bot gives a wrong answer. How do you evaluate the pipeline and figure out whether it's the retriever or the generator that's broken?**  `Applied`  
Evaluate RAG in __two layers__: score retrieval and generation separately, because a bad answer can come from fetching the wrong chunks OR from the model misusing good chunks.  Retrieval uses ranking metrics (precision@k, recall@k, MRR, nDCG); generation splits into answer correctness (right vs a reference) and faithfulness (grounded in the chunks).  

➡️ **Full answers, diagrams & practice:** [RAG — Retrieval-Augmented Generation on Skillumen →](https://www.skillumen.com/blog/rag-interview-questions.html)
📎 **Related deep-dives:** [What is RAG?](https://www.skillumen.com/blog/what-is-rag.html) · [RAG vs Fine-tuning](https://www.skillumen.com/blog/rag-vs-fine-tuning.html)

---

## 🎯 Fine-tuning & Alignment

**Q — Distinguish pretraining, SFT, and RLHF.**  `Applied`  
A model is built in three stages: first it learns language, then it learns to follow instructions, then it learns to match what humans actually prefer.  

**Q — How does DPO differ from classic RLHF?**  `Applied`  
DPO teaches a model human preferences directly from 'this answer is better than that one' pairs — skipping the separate reward model and reinforcement-learning loop that RLHF needs.  

**Q — How does LoRA fine-tune a model cheaply?**  `Applied`  
LoRA freezes the giant base model and trains only tiny add-on matrices, so fine-tuning becomes cheap and you can keep many task-specific versions.  

**Q — What is QLoRA and why does it matter?**  `Applied`  
QLoRA is a cheap way to customize a giant model on one GPU.  It keeps the big model frozen and stores it in a tiny __4-bit__ format (quantized means saving each number with far fewer digits), then trains only a few small add-on pieces (LoRA adapters).  

**Q — Compare causal LM, masked LM, and prefix-LM objectives.**  `Applied`  
How a model is pretrained shapes what it's good at: causal LM (predict the next token) powers generators, masked LM (__fill blanks__) powers understanders, and prefix-LM blends both.  

**Q — What is instruction tuning and how does it differ from plain SFT?**  `Applied`  
Instruction tuning fine-tunes a base model on a wide variety of (instruction to response) tasks so it learns to *follow instructions in general*, not just mimic one dataset.  

**Q — What is catastrophic forgetting and how is it mitigated?**  `Applied`  
Catastrophic forgetting is when fine-tuning a model on new data makes it lose abilities it already had.  PEFT methods like LoRA avoid it by leaving the base weights untouched.  

**Q — What is knowledge distillation?**  `Applied`  
Knowledge distillation trains a small __student__ model to copy a large teacher model's outputs — getting most of the quality at a fraction of the size and cost.  

**Q — Walk through full RLHF end to end.**  `Applied`  
RLHF first trains a scorer (a reward model) that learns what people like, then gently nudges the model (PPO) to earn higher scores — while a leash (KL divergence) keeps it from wandering off into nonsense to game the score.  

**Q — What is RLVR and why did it change LLM training?**  `Advanced`  
RLVR (Reinforcement Learning from Verifiable Rewards) trains models on problems where the answer can be __automatically checked__ — like math or code — so the reward is objective, not a human guess.  

**Q — What is Constitutional AI and RLAIF?**  `Advanced`  
Constitutional AI aligns a model using a written set of principles (a 'constitution') and __AI-generated feedback__ (RLAIF) instead of relying only on human labels.  

**Q — Why is pretraining data quality and synthetic data so important?**  `Advanced`  
A model is only as good as its data.  2026 training leans on careful curation, deduplication, and filtering, plus large amounts of __synthetic data__ — including generated reasoning traces.  

**Q — What is machine unlearning?**  `Advanced`  
Machine unlearning makes a trained model forget specific data — for privacy ('right to be forgotten'), copyright, or safety — without retraining from scratch.  

➡️ **Full answers, diagrams & practice:** [Fine-tuning & Alignment on Skillumen →](https://www.skillumen.com/blog/rag-vs-fine-tuning.html)
📎 **Related deep-dives:** [RAG Interview Questions](https://www.skillumen.com/blog/rag-interview-questions.html)

---

## 🤖 Agents & Tool Use

**Q — How does an LLM agent work, and what is the ReAct loop?**  `Applied`  
An agent works in a loop: think, call a tool (tool calling), look at the result, and repeat until it can answer — known as the ReAct pattern.  

**Q — What is LangChain and how do you build a RAG chain with it?**  `Applied`  
LangChain is a toolkit that snaps models, prompts, retrievers, and parsers together into reusable chains — and into agents via LangGraph.  

**Q — What problem does MCP solve and how is it structured?**  `Advanced`  
MCP is a shared standard that lets any AI app connect to any tool or data source — without writing custom glue code for every combination.  

**Q — How do agents remember things beyond the context window?**  `Applied`  
Agents fake long-term memory with external storage: __short-term__ (recent turns), summary memory, and long-term memory in a vector store that's retrieved when relevant.  

**Q — What are common multi-agent patterns?**  `Applied`  
For a big task, split the work across __specialized agents__ instead of one agent doing it all.  A planner breaks the job into steps, workers do the steps, and an orchestrator hands each step to the right agent.  

**Q — Beyond calling an LLM in a loop, what does the 2026 agent stack actually run on?**  `Applied`  
Production agents are built on named frameworks — langgraph for stateful agent loops, llamaindex for data/retrieval agents — and increasingly talk to each other over open protocols: mcp (agent to tools) and a2a (agent to agent).  

**Q — Your agent works in testing but throws errors under real traffic. What's the most common cause — and the fix?**  `Applied`  
In production, most LLM failures aren't the model being wrong — they're rate limits and capacity errors.  Handling them with retries, exponential backoff and fallbacks is what makes an agent reliable.  

**Q — Beyond chat, 2026 agents talk out loud and click around screens. What's new about building those?**  `Advanced`  
Voice agents chain speech-to-text → LLM → text-to-speech (where latency is everything), while computer-use and browser agents act by clicking and typing on real screens — both add real-time and reliability challenges beyond text chat.  

**Q — Everyone watches their agents in production. Why is that not the same as knowing they work?**  `Applied`  
Watching an agent (observability) tells you what happened; agent evaluation tells you whether it was right.  Most teams have the first and skip the second — and evaluating an agent means scoring its whole trajectory, not just the final answer.  

➡️ **Full answers, diagrams & practice:** [Agents & Tool Use on Skillumen →](https://www.skillumen.com/blog/agentic-ai-interview-questions.html)
📎 **Related deep-dives:** [LangGraph vs MCP](https://www.skillumen.com/blog/langgraph-vs-mcp.html)

---

## 📊 Evaluation

**Q — How do you evaluate an LLM app, and what is LLM-as-judge?**  `Applied`  
Testing an LLM app happens at three levels: standard exams every model takes (benchmarks), your own quiz built from real tasks, and a strong model acting as grader (LLM-as-judge).  For a RAG app, RAGAS also checks faithfulness — whether the answer really sticks to the documents it was given.  

**Q — What metrics measure LLM output quality?**  `Applied`  
There's no single score for 'good' — you pick the __metric that fits the task__.  perplexity checks how smoothly it reads, BLEU/ROUGE check how many words match a human answer, BERTScore checks if the meaning matches, pass@k checks if code actually runs, and LLM-as-judge rates open-ended answers like chat.  

**Q — What do common LLM benchmarks measure, and what are their limits?**  `Advanced`  
Benchmarks like MMLU, GPQA, HumanEval, SWE-bench, and AIME each test __something specific__.  Know what they measure — and their limits (contamination, saturation, gaming).  

**Q — Your RAG app gave a wrong answer. Before you touch anything: is the bug in the prompt, the model, retrieval, the schema, or the tool — and how do you prove which?**  `Applied`  
A bad LLM output is a symptom, not a diagnosis.  Isolate the broken layer — prompt, model, retrieval, schema/parser, or tool — by reading the symptom, then fix the __cheapest__ layer first.  

**Q — Why do LLMs hallucinate and how do you reduce it?**  `Applied`  
A hallucination is confident, fluent text that's __factually wrong__.  It happens because models predict plausible words, not truth — mitigations include RAG, better prompts, and verification.  

➡️ **Full answers, diagrams & practice:** [Evaluation on Skillumen →](https://www.skillumen.com/blog/llm-interview-questions.html)

---

## 🚀 Inference & Serving

**Q — What is the KV cache and why does it matter for inference?**  `Applied`  
The KV cache saves the work the model already did on earlier tokens, so generating each new token is fast instead of redoing everything.  

**Q — What is quantization and what's the trade-off?**  `Applied`  
Quantization stores the model's numbers using fewer bits, making it much smaller and faster to run — for a small drop in accuracy.  

**Q — How does speculative decoding speed up generation?**  `Applied`  
Speculative decoding uses a small fast __draft__ model to guess several tokens ahead, then the big model verifies them all in one pass — same output, much faster.  

**Q — How does vLLM serve LLMs with high throughput?**  `Advanced`  
vLLM is a fast serving engine: PagedAttention manages the KV cache efficiently and continuous batching keeps the GPU busy — together giving far higher throughput.  

**Q — At scale, serving an LLM isn't just 'run vLLM'. What actually decides your speed and cost?**  `Advanced`  
Production serving is shaped by the KV cache.  The two generation phases — prefill (compute-heavy) and decode (memory-heavy) — stress GPUs differently, so 2026 stacks split them onto separate pools (disaggregation) and route requests by cached prefix.  

**Q — You've got a model and a working RAG chain. What actually turns that into a production service?**  `Advanced`  
Shipping an LLM feature means wrapping inference in an API with real telemetry (tokens, latency, cost), a local dev loop, experiment tracking, and observability — the unglamorous stack that turns 'works on my laptop' into 'works for users'.  

**Q — What are the prefill and decode phases, and why does batching matter?**  `Advanced`  
Answering a prompt happens in two steps.  First __prefill__: the model reads your whole prompt at once.  

**Q — Your product now calls five different models across three providers. How do you keep that from becoming a mess?**  `Advanced`  
A model gateway puts one API in front of every provider — with retries, fallbacks, keys and spend limits in one place — so running a fleet of models stays sane.  

**Q — What is model routing and cascading?**  `Advanced`  
Model routing sends each request to the __cheapest model__ that can handle it; cascades try a small model first and escalate to a bigger one only if needed — cutting cost without losing quality.  

**Q — Why are small and on-device language models important?**  `Advanced`  
Small language models (SLMs, ~1-8B) run on phones and laptops — giving privacy, low latency, offline use, and __near-zero cost__ — and 2026 distillation makes them surprisingly capable.  

**Q — Your app sends the same long system prompt on every call. How do you stop paying to re-read it each time?**  `Applied`  
Prompt caching lets the model keep the unchanging start of your prompt ready, so repeat calls skip re-processing those tokens — cutting both cost and the wait for the first word.  

**Q — Design a customer-support RAG bot for 1M docs at scale — name the key components.**  `Advanced`  
A production RAG system has two halves: an offline phase that indexes your documents, and an online phase that answers queries — wrapped in guardrails and monitoring.  

➡️ **Full answers, diagrams & practice:** [Inference & Serving on Skillumen →](https://www.skillumen.com/blog/llm-system-design-interview.html)

---

## 🛠️ Production, Ops & Cost

**Q — What production techniques cut LLM cost and latency?**  `Advanced`  
The everyday engineering moves that make a live LLM app cheap, fast, and dependable: reusing past work (caching), showing words as they're typed (streaming), sending easy questions to a cheaper model (model routing), and having a backup when a provider fails (fallbacks with retries).  

**Q — Why do LLM apps need observability, and what do tools capture?**  `Advanced`  
observability keeps a full recording of what your app did — every step, plus how long it took (latency) and what it cost.  So when something goes wrong deep inside a long chain of steps, you can replay it and find the problem.  

**Q — How do you monitor LLM apps and detect drift in production?**  `Advanced`  
Drift is when inputs or model behaviour change over time, silently degrading quality.  Monitoring tracks inputs, outputs, quality signals, cost, and latency to catch it early.  

**Q — How do you estimate and control LLM costs?**  `Advanced`  
LLM cost is driven by __tokens__: you pay per input + output token.  Estimating and cutting cost means counting tokens, caching, routing to cheaper models, and trimming prompts.  

**Q — Your LLM feature cost pennies in testing and thousands the month it launched. How do you stay in control?**  `Advanced`  
LLM finops treats token spend like a budget: attribute cost per feature and team, cut it with caching and cheaper models, and enforce hard daily caps with a circuit breaker so a bug or spike can't run up a giant bill.  

**Q — Design the backend for a streaming LLM chatbot. How do you keep it fast, safe, and from blowing your API budget?**  `Advanced`  
An LLM app is still a normal web service.  You expose a versioned REST API with typed contracts, ship it as a Docker image, keep provider keys out of code, and — the part interviewers love — enforce YOUR OWN rate limit per user so nobody drains your provider budget.  

**Q — An interviewer asks: "A user closes your chat app and comes back tomorrow expecting their conversation. They also uploaded a 40-page PDF. Where does all of this live, and why not just keep it in the model's context?**  `Advanced`  
The model's context window is wiped after every reply, so real chat state lives in stores you choose by job: Postgres for durable history (users/conversations/messages in a transaction), Redis for fast throwaway state (TTL caches, sessions, rate limiting counters, queues), and object storage with presigned URLs for uploaded PDFs and logs.  

**Q — Your LLM app needs to handle 1000 users at once, stream answers to their browsers live, and process uploaded PDFs in the background. What's the backend shape?**  `Advanced`  
Most LLM work is __waiting__ on the model API, so one async process can hold ~1000 users at once.  Short live answers go out as a token stream over SSE; slow work (parsing PDFs) is handed to a task queue and a worker, and the client polls for the result instead of holding the HTTP connection open.  

**Q — How would you deploy an LLM app to production?**  `Advanced`  
Deploying an LLM app is mostly normal web deployment: containerize your FastAPI+RAG service, push the image to a host, and inject config with env vars and secrets.  The twist is that heavy GPU inference scales and costs differently, so it lives behind a hosted API or a separate GPU pool while your stateless app runs cheap and autoscales.  

**Q — Your team just fine-tuned a new model. How do you ship it to production without risking an outage or a silent quality drop?**  `Advanced`  
You gate the release in CI/CD and roll it out gradually.  Automated eval gates run in CI and block the merge if quality drops; then a canary or blue-green deploy exposes the new build to a sliver of traffic, and a one-click rollback snaps back if anything breaks.  

**Q — Your RAG app serves 500 companies from one vector index. How do you make sure a user from Company A can never retrieve Company B's documents?**  `Advanced`  
First you prove who the user is (authentication) and what they're allowed to see (authorization/RBAC).  Then the real trick: access-controlled retrieval.  

**Q — How do you force valid JSON out of an LLM and guard inputs/outputs?**  `Applied`  
structured output makes the model fill in a fixed form instead of writing a paragraph, so your code always finds each value in the same spot (the form is a Pydantic schema; tools like Instructor or Outlines enforce it).  guardrails are the safety checks that block bad or sneaky input and output.  

**Q — You're shipping a feature on top of an LLM API. Walk me through everything between 'user types a question' and 'answer streams back' — and what breaks in production.**  `Applied`  
Calling an LLM API well means shaping the message roles and model parameters, counting tokens before you send, and wrapping the call in retries with exponential backoff plus timeouts.  Production adds streaming to the frontend, provider fallback across OpenAI/Anthropic/Gemini, and cost tracking from the usage block.  

➡️ **Full answers, diagrams & practice:** [Production, Ops & Cost on Skillumen →](https://www.skillumen.com/blog/llm-system-design-interview.html)

---

## 🛡️ Safety & Security

**Q — What is prompt injection and how do you defend against it?**  `Advanced`  
Prompt injection is when untrusted text hijacks the model's instructions — e. g.  

**Q — What is red-teaming and how are LLMs made safer?**  `Advanced`  
Red-teaming means attacking your own model on purpose — trying to make it say something harmful — so you catch the holes before real users do.  Real safety then stacks layers on top: training the model to refuse, filtering what goes in and out, and boxing in what its tools can touch.  

**Q — Where does LLM bias come from and how is it measured?**  `Advanced`  
LLMs absorb biases from their training data — stereotypes, skew, toxicity.  Managing it means __measuring__ across groups, mitigating in data and training, and filtering outputs.  

**Q — How do you handle privacy and data governance with LLMs?**  `Advanced`  
LLMs can leak or memorize sensitive data, so production systems __redact PII__, control what's sent to third-party APIs, log carefully, and follow data-governance rules (GDPR, etc. ).  

**Q — Your agent can call tools and read web pages. What's the security risk almost everyone underestimates?**  `Advanced`  
Once an agent can act on the world, anything it reads — a tool's output, a web page, another agent — can carry hidden instructions.  tool poisoning and indirect prompt injection are the new attack surface, and the fix is to never trust AI-mediated input.  

**Q — You built an agent that writes and runs SQL against your production database. Your interviewer asks: how do you keep it from leaking or wrecking data?**  `Advanced`  
Treat every tool argument the model produces as __untrusted__ input.  Lock the database itself down to a read-only least-privilege user, allow-list SELECT-only queries, bind values with parameterized queries (never string-concat model output), force tenant filters in your code, and cap rows plus set a timeout.  

**Q — Your agent writes and runs Python to answer questions. What's the obvious thing that can go very wrong?**  `Advanced`  
If an agent can generate and execute code, that code can delete files, leak secrets, or attack your network.  Sandboxing runs it in an isolated, disposable environment with tight limits — so a bad snippet can't touch your real system.  

➡️ **Full answers, diagrams & practice:** [Safety & Security on Skillumen →](https://www.skillumen.com/blog/ai-engineer-interview-questions.html)

---

## 🔬 Scaling & the Research Frontier

**Q — What did Chinchilla reveal about compute-optimal training?**  `Advanced`  
For a fixed amount of compute, the model size and the amount of training data should grow together — and many famous early models were actually undertrained.  

**Q — How do reasoning models (o-series, R1) differ from normal LLMs?**  `Advanced`  
Reasoning models (OpenAI o-series, DeepSeek-R1) are trained to think before answering — generating a long internal chain of thought — trading more compute at inference for far better math, code, and logic.  

**Q — What is test-time (inference-time) compute scaling?**  `Advanced`  
Test-time compute means spending more computation when answering — not just when training — to get better results: longer reasoning, many samples, or search over candidates.  

**Q — What are pruning and sparsity in LLMs?**  `Advanced`  
Pruning removes weights or whole components that barely matter, making a model smaller and faster.  Sparsity means most weights are zero — skipped at compute time.  

**Q — How do multimodal LLMs (vision-language models) work?**  `Advanced`  
Multimodal models handle images (and audio/video) alongside text by encoding each into the same token space, so the LLM can reason across them — e. g.  

**Q — What are diffusion language models?**  `Advanced`  
Instead of writing one word after another, Diffusion LLMs start with a whole draft of blanks and sharpen every position __at once__, over a few passes.  Because the work happens in parallel, text can arrive in a burst, not a trickle — much lower latency.  

**Q — What are the main LLM families in 2026 and their trade-offs?**  `Advanced`  
Three camps: __closed frontier__ (GPT, Claude, Gemini), open-weight (Llama, Mistral, and Chinese labs DeepSeek/Qwen/Kimi), and small/specialized models.  The trade-off is capability vs control vs cost.  

➡️ **Full answers, diagrams & practice:** [Scaling & the Research Frontier on Skillumen →](https://www.skillumen.com/blog/ai-engineer-interview-questions.html)
📎 **Related deep-dives:** [AI Engineer Roadmap 2026](https://www.skillumen.com/blog/ai-engineer-roadmap-2026.html)

---

## Practice these for real

Reading questions is not rehearsing. [Skillumen](https://www.skillumen.com/?utm_source=github&utm_medium=awesome-list&utm_campaign=ai-interview) turns every one of these into:

- 🃏 an **active-recall card** (retrieve it, don't re-read it)
- 🎙️ an **AI voice mock interview** that asks follow-ups and grades your answer
- 🧪 a **build lab** and a real **RAG capstone** you deploy

The **Foundations** tier is free forever. → **[Start free](https://www.skillumen.com/?utm_source=github&utm_medium=awesome-list&utm_campaign=ai-interview)**

## Further reading

- [AI Engineer Interview Questions (2026)](https://www.skillumen.com/blog/ai-engineer-interview-questions.html)
- [LLM Interview Questions](https://www.skillumen.com/blog/llm-interview-questions.html)
- [RAG Interview Questions](https://www.skillumen.com/blog/rag-interview-questions.html)
- [Agentic AI Interview Questions](https://www.skillumen.com/blog/agentic-ai-interview-questions.html)
- [Generative AI Interview Questions](https://www.skillumen.com/blog/generative-ai-interview-questions.html)
- [LLM System Design Interview](https://www.skillumen.com/blog/llm-system-design-interview.html)
- [AI Engineer Roadmap 2026](https://www.skillumen.com/blog/ai-engineer-roadmap-2026.html)
- [What is RAG?](https://www.skillumen.com/blog/what-is-rag.html) · [RAG vs Fine-tuning](https://www.skillumen.com/blog/rag-vs-fine-tuning.html) · [LangGraph vs MCP](https://www.skillumen.com/blog/langgraph-vs-mcp.html)

## Contributing

Found a question that should be here, or a sharper one-line answer? Open a PR or an issue. Keep answers to 1–2 sentences; link deep explanations rather than pasting them.

## License

[![CC0](https://licensebuttons.net/p/zero/1.0/80x15.png)](https://creativecommons.org/publicdomain/zero/1.0/) — released under CC0. Attribution to [Skillumen](https://www.skillumen.com) appreciated but not required.
