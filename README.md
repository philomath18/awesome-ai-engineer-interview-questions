# Awesome AI Engineer Interview Questions [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

> 105 real interview questions for **AI / LLM engineer** roles, covering LLMs, transformers, prompting, RAG, fine-tuning, agents, evaluation, inference, production and safety, each with a concise answer. Maintained by [Skillumen](https://www.skillumen.com).

An AI-engineering interview really asks one thing: have you actually shipped this? Every question below comes up in real interviews in 2026, grouped by topic, with a short answer to anchor your thinking. For worked answers and a chance to **rehearse out loud in an AI voice mock**, follow the topic links.

<p align="center"><a href="https://www.skillumen.com/?utm_source=github&utm_medium=awesome-list&utm_campaign=ai-interview"><b>Practice these free: the 30-day LLM interview bootcamp</b></a></p>

**Updated 2026-09-09** · 105 questions · 11 topics · answers link to full write-ups.

## Contents

- [🧠 LLM Foundations](#-llm-foundations)
- [⚙️ Transformers & Architecture](#️-transformers--architecture)
- [✍️ Prompting & In-Context Learning](#️-prompting--in-context-learning)
- [🔎 RAG (Retrieval-Augmented Generation)](#-rag-retrieval-augmented-generation)
- [🎯 Fine-tuning & Alignment](#-fine-tuning--alignment)
- [🤖 Agents & Tool Use](#-agents--tool-use)
- [📊 Evaluation](#-evaluation)
- [🚀 Inference & Serving](#-inference--serving)
- [🛠️ Production, Ops & Cost](#️-production-ops--cost)
- [🛡️ Safety & Security](#️-safety--security)
- [🔬 Scaling & the Research Frontier](#-scaling--the-research-frontier)

---

## 🧠 LLM Foundations

**Q: In plain terms, what one task is an LLM actually doing?**  `Foundations`  
It predicts the next token, the most likely next word-piece given all the text so far.  That is the entire job: at each step it outputs a probability distribution over its whole vocabulary, and a token is sampled from it.  

**Q: Why might a base model reply to a question with more questions?**  `Foundations`  
A base model is trained on exactly one objective, predicting the next token, so it continues the pattern of the text rather than responding to it.  In its training data a question is very often followed by more questions, in quiz lists and FAQs, so the most likely continuation of "What is the capital of France?  

**Q: What's the difference between training and inference?**  `Foundations`  
Training is the learning phase.  The model reads data, computes a loss on its next-token guesses, and uses backpropagation to update billions of weights.  

**Q: What's the difference between an embedding and a generative model?**  `Foundations`  
An embedding model maps a piece of text to a single fixed-length vector that captures its meaning.  It returns numbers rather than words, and you use it to measure similarity for search, clustering, and retrieval.  

**Q: Walk me through how a model like ChatGPT is built.**  `Foundations`  
It is built in three training stages, each editing the weights.  Pretraining reads a huge slice of the internet predicting the next token, and picks up language and world knowledge as a side effect.  

**Q: What does 'hallucination' mean for an LLM, and why is that term better than 'lying'?**  `Foundations`  
A hallucination is when the model states something false as if it were fact, in the same confident tone it uses when it is right.  "Lying" implies the model knows the truth and chooses to deceive, and nothing like that is happening.  

**Q: What is the context window, and what happens to tokens that fall outside it?**  `Foundations`  
The context window is the maximum number of tokens a model can attend to in a single call.  It is the model's working memory, and it is shared across the system prompt, the conversation history, any pasted documents, and the reply the model generates.  

**Q: What are logits, and how do they differ from the probabilities the model ultimately uses?**  `Foundations`  
A logit is the model's raw confidence score for one candidate token, and the final layer emits one per token in the whole vocabulary.  Logits are unbounded, they can be negative or large, and they do not sum to anything, so they are really just a ranking.  

➡️ **Full answers, diagrams and practice:** [LLM Foundations on Skillumen](https://www.skillumen.com/blog/llm-interview-questions.html)

---

## ⚙️ Transformers & Architecture

**Q: What does tokenization do to your text, and why is it necessary?**  `Foundations`  
A model only does math over numbers, so raw letters mean nothing to it.  Tokenization bridges that.  

**Q: What is an embedding, and why is a raw token ID not enough?**  `Foundations`  
An embedding replaces a token's ID with a learned dense vector, a long list of numbers, say 384 or 4096 of them.  A raw ID like 4127 is just a label.  

**Q: In plain terms, what does self-attention let each token do?**  `Foundations`  
Self-attention lets every token look at all the other tokens in the sequence and pull in meaning from the ones most relevant to it, in a single step.  Instead of reading a word in isolation, the model rebuilds each token's vector as a blend of the words it relates to, so "it" can absorb the meaning of "cat" however far back it sits.  

**Q: What does it mean that self-attention is 'order-blind', and what concrete confusion does that cause?**  `Foundations`  
Self-attention only computes how much each word should attend to each other word, and that quantity doesn't depend on the order of the inputs.  It treats them like a bag of words.  

**Q: Describe the autoregressive generation loop step by step, and when does it stop?**  `Foundations`  
You start with a prompt.  One forward pass produces logits, a score for every possible next token.  

**Q: What does the temperature dial control, and what happens at 0 versus above 1?**  `Foundations`  
Temperature stretches or squashes the odds before sampling: you divide the logits by it.  Below 1, the gaps between options grow, the top word dominates, and text stays focused and safe.  

**Q: What does cross-entropy loss actually measure, and how is it computed at a single position?**  `Foundations`  
At one position the model outputs a logits vector, one raw score per word in the vocabulary, and a softmax turns those scores into probabilities that sum to 1.  You look up the probability it assigned to the one word that truly came next, take its negative log, and that is the loss for that step.  

**Q: What's the fundamental difference in how an RNN and a transformer read a sequence?**  `Foundations`  
An RNN reads sequentially, one token at a time, updating a single running hidden state that carries everything it knows about the past forward.  A transformer reads the whole sequence at once and uses self-attention, so any token can look directly at any other, regardless of distance.  

**Q: What are the three architecture families, and their poster-child models?**  `Foundations`  
There are three families.  Encoder-only models like BERT, RoBERTa, and DeBERTa read bidirectionally and are built for understanding: classification, tagging, embeddings.  

**Q: Why does a transformer need positional encoding at all?**  `Foundations`  
A transformer processes every token in parallel, which is what makes it fast, and it means raw attention sees a bag of vectors with no sense of slot.  Shuffle the words and you get the identical result, so "dog bites man" and "man bites dog" are indistinguishable.  

**Q: What do residual connections and normalization each do to keep a deep transformer trainable?**  `Foundations`  
A residual connection computes x + f(x) instead of just f(x), so the original signal is always carried forward.  Going forward it reaches deep layers without fading, and going backward the gradient gets a clean path home, which sidesteps the vanishing gradient problem.  

**Q: What role does an activation function play, and what happens to stacked layers without any non-linearity?**  `Foundations`  
An activation function adds non-linearity between layers.  Without it, stacking layers buys you nothing, because a chain of linear multiply-and-add layers collapses into a single linear layer, one straight line with no extra expressive power no matter how deep you go.  

**Q: At each step the model produces logits. What does a decoding strategy actually decide?**  `Foundations`  
After the final layer the model hands you logits, one raw score per word in the vocabulary, and softmax turns them into probabilities.  That distribution isn't an answer yet.  

**Q: What are the two jobs a single transformer block performs, and which is the only place words share information?**  `Foundations`  
A block has exactly two sublayers.  Multi-head self-attention lets each token look at the others and pull in context, and it is the only place information moves between positions.  

**Q: What does running multiple attention heads let the model do that a single reader couldn't?**  `Applied`  
A single attention pass can only emphasise one pattern at a time.  Multi-head attention slices the vectors into several heads, each with its own Query/Key/Value, and they run in parallel.  

**Q: What does FlashAttention compute compared to standard attention, and what does it avoid writing to slow memory?**  `Advanced`  
It computes the exact same attention output.  It is not an approximation.  

**Q: What core problem with transformer attention do SSMs and linear attention set out to fix?**  `Advanced`  
Transformer attention compares every token to every other token, so compute grows with the square of the sequence length, and the KV cache of stored keys and values grows linearly with length on top of that.  On short text you never notice.  

**Q: What does a Mixture of Experts hold, and how does a router keep the per-token cost low despite huge total capacity?**  `Advanced`  
An MoE layer holds N separate expert networks, which are ordinary feed-forward blocks, plus a small router.  For each token the router scores the experts and sends it to only the top-k, commonly 2 of 8.  

**Q: What is the router's job in a Mixture of Experts model, and what does 'top-2' routing mean for cost?**  `Advanced`  
The router is a tiny layer, usually one linear layer plus a softmax, that scores all the experts for each word-piece and picks the top-k best, very often just the top 2.  Only those 2 experts run and the rest are skipped.  

➡️ **Full answers, diagrams and practice:** [Transformers & Architecture on Skillumen](https://www.skillumen.com/blog/llm-interview-questions.html)

---

## ✍️ Prompting & In-Context Learning

**Q: What does in-context learning mean, and what makes it different from fine-tuning?**  `Foundations`  
In-context learning is a model's ability to do a new task purely from the instructions or examples in the prompt, with no weight updates at all.  Fine-tuning permanently changes the weights on a labelled dataset.  

**Q: Contrast zero-shot and few-shot prompting. When is few-shot the simplest reliable win?**  `Applied`  
Zero-shot gives the model only the instruction and trusts it to comply.  Few-shot pastes a handful of worked examples first, so the model copies the pattern.  

**Q: What does treating 'prompts as code' mean in practice, and which engineering habits does it borrow?**  `Applied`  
It means recognizing that a prompt is program logic written in English and giving it the same discipline as code.  Keep it in version control, in Git or a prompt registry, with a version and a pull-request review.  

➡️ **Full answers, diagrams and practice:** [Prompting & In-Context Learning on Skillumen](https://www.skillumen.com/blog/llm-interview-questions.html)
📎 **Related deep-dives:** [Generative AI Interview Questions](https://www.skillumen.com/blog/generative-ai-interview-questions.html)

---

## 🔎 RAG (Retrieval-Augmented Generation)

**Q: What problem does RAG solve about a model's knowledge, and how does it reduce hallucination?**  `Applied`  
A model's knowledge is frozen at its training cutoff and never included your private data.  So it can't answer about recent events or your own docs, and when pushed it tends to hallucinate a confident guess.  

**Q: What is chunking in a retrieval system, and why does bad chunking quietly wreck search?**  `Applied`  
Chunking is cutting your documents into small pieces, embedding each one, and storing them so search can compare a question to one piece at a time.  It quietly decides quality because retrieval matches whole chunks.  

**Q: What job does the embedding model do in a search pipeline, and what does it ultimately decide?**  `Applied`  
The embedding model reads each piece of text and outputs a fixed-length vector that captures its meaning, placing similar-meaning texts close together.  At query time you embed the question and return the nearest chunks, so the embedding model rather than the search box decides what counts as a match.  

**Q: What are the three ways to give a model knowledge, and the one-line pitch for each?**  `Applied`  
There are three tools for three different jobs.  RAG, retrieval-augmented generation, keeps facts outside the model and fetches the relevant ones at question time, so the model looks things up.  

**Q: What does basic RAG get wrong often enough that advanced RAG exists to fix?**  `Applied`  
Basic RAG runs one similarity search on the user's raw question and assumes the top-k chunks are right.  In practice the retrieval is what breaks.  

**Q: What does a vector database store, and what does it find for a given query?**  `Applied`  
A vector database stores embeddings, lists of numbers that capture what each piece of text or each image means, as points in a high-dimensional space.  For a query, it embeds the query and returns the nearest neighbours: the stored vectors closest to the query point, which are the items closest in meaning.  

**Q: What two-stage idea underlies reranking plus hybrid retrieval?**  `Applied`  
It's a retrieve-then-rerank pipeline borrowed from classic search.  Stage one casts a wide, cheap net tuned for recall.  

**Q: Why does standard similarity-based RAG fail on 'how do X and Y relate across three documents?'**  `Applied`  
Similarity search returns the chunks that look most like the question, which is perfect for a plain "what is X" lookup.  A relational, multi-hop question is different.  

**Q: Why must you score retrieval and generation as two separate layers?**  `Applied`  
A RAG answer can fail in two independent places, and the final answer looks the same either way.  Retrieval might fetch the wrong chunks.  

➡️ **Full answers, diagrams and practice:** [RAG (Retrieval-Augmented Generation) on Skillumen](https://www.skillumen.com/blog/rag-interview-questions.html)
📎 **Related deep-dives:** [What is RAG?](https://www.skillumen.com/blog/what-is-rag.html) · [RAG vs Fine-tuning](https://www.skillumen.com/blog/rag-vs-fine-tuning.html)

---

## 🎯 Fine-tuning & Alignment

**Q: Name the three stages of building a model and what each one teaches.**  `Applied`  
There are three stages, in order.  Pretraining has the model read a massive pile of raw text predicting the next token, so it absorbs language, facts, and reasoning, though all it learns to do is continue text.  

**Q: In plain terms, what does DPO learn from, and what two heavy RLHF components does it skip?**  `Applied`  
DPO (Direct Preference Optimization) learns straight from preference pairs: for each prompt, a chosen answer the raters preferred and a rejected one.  It skips the two heavy pieces of classic RLHF, the separate reward model that scores answers and the reinforcement-learning loop (PPO) that pushes the model toward higher scores.  

**Q: What does LoRA freeze and what does it actually train, and why does that make many task versions cheap?**  `Applied`  
LoRA freezes the entire base model and trains only a small pair of add-on matrices, B·A, on a few chosen layers.  That is typically well under 1% of the weights.  

**Q: What does QLoRA keep frozen and store in 4-bit, and what does it actually train?**  `Applied`  
QLoRA freezes the entire base model and stores it in 4-bit, using the NF4 format, so the huge part barely uses memory.  The only weights that actually update are the small LoRA adapters bolted on top, typically well under 1% of the parameters, and those train in full precision so learning stays sharp.  

**Q: How does the pretraining objective shape what a model is good at?**  `Applied`  
The objective is the exact prediction task you reward during pretraining, and it quietly decides the model's lifelong strengths.  Causal LM predicts the next token from the past, which builds fluent generators.  

**Q: What does instruction tuning teach a model that plain next-token pretraining does not?**  `Applied`  
Pretraining only teaches what word comes next, so a base model continues text.  Ask it a question and it may write more questions.  

**Q: What is catastrophic forgetting, and when does it typically strike a model?**  `Applied`  
It is when training a model on new data makes it lose abilities it already had.  It typically strikes during fine-tuning, especially on a narrow dataset.  

**Q: In plain terms, what is knowledge distillation trying to achieve with a student and a teacher model?**  `Applied`  
Distillation takes a large, capable but expensive teacher model and uses it to train a small student model.  You run the frozen teacher over lots of inputs and train the student to reproduce its outputs.  

**Q: What are the three stages of RLHF, from preference data to the final model?**  `Applied`  
Three stages.  First, collect preferences: humans compare pairs of answers to the same prompt and pick the better one.  

**Q: What makes a reward 'verifiable' in RLVR, and on what kinds of problems does that apply?**  `Advanced`  
A reward is verifiable when a program can decide whether the answer is correct, with no human opinion involved, yielding a clean 1 or 0.  That applies wherever correct is well defined.  

**Q: What is a 'constitution' in Constitutional AI, and how does the model use it?**  `Advanced`  
A constitution is a short, plain-language list of principles: be helpful, avoid harmful advice, be honest about uncertainty.  Anthropic's draws some of its lines from sources like the UN Declaration of Human Rights.  

**Q: Why is 'a model is only as good as its data' true when model size is held fixed?**  `Advanced`  
Two models of the same size have the same capacity to fill, and what fills it is the data.  Every token costs the same compute to train on, so a duplicated or spammy token wastes that budget while a clean, information-rich one teaches something.  

**Q: What is machine unlearning, and which pressures make it necessary?**  `Advanced`  
Machine unlearning makes a trained model behave as if specific data was never in its training set, without retraining from scratch.  Three pressures force it: privacy laws like GDPR's "right to be forgotten," copyright holders whose work was scraped, and safety, meaning removal of hazardous or unsafe knowledge.  

➡️ **Full answers, diagrams and practice:** [Fine-tuning & Alignment on Skillumen](https://www.skillumen.com/blog/rag-vs-fine-tuning.html)
📎 **Related deep-dives:** [RAG Interview Questions](https://www.skillumen.com/blog/rag-interview-questions.html)

---

## 🤖 Agents & Tool Use

**Q: What is the repeating loop an agent runs, and when does it stop?**  `Applied`  
An agent runs the ReAct loop: a Thought where it reasons about what to do next, an Action where it calls a tool, and an Observation where the result is fed back.  It keeps cycling through think, act, observe, building up what it has learned, and it stops when it has gathered enough to write a final answer, or when a safety cap like a step limit forces it to.  

**Q: What does LangChain give you that saves writing custom glue code?**  `Applied`  
It gives you standard, swappable building blocks: chat models, prompt templates, output parsers, memory, retrievers, and tools.  They all speak the same runnable interface, taking an input and returning an output the same way.  

**Q: What problem does MCP solve?**  `Advanced`  
Before MCP, every AI app needed its own bespoke connector to every tool.  With N apps and M tools that is N×M one-off integrations to build and maintain, and a connector for one app was useless to the next.  

**Q: Why do agents need a memory system at all? Doesn't the model remember?**  `Applied`  
A chat model is stateless between calls.  It carries nothing over on its own.  

**Q: What's the core idea of multi-agent orchestration versus one agent doing everything?**  `Applied`  
Instead of one agent trying to plan, research, write, and check all at once, you split the work across specialized agents, each with a narrow job, its own instructions, and only the tools it needs, and you add a coordinator to route between them.  A single agent bloats its prompt and loses focus as the task grows, and one early mistake poisons everything after.  

**Q: What two categories should a 2026 LLM engineer tell apart, and which tools sit in each?**  `Applied`  
There are two layers.  Orchestration frameworks run the agent's own steps and state.  

**Q: In production, what actually causes most agent failures: the model or the plumbing around it?**  `Applied`  
The plumbing, overwhelmingly.  Large-scale telemetry shows roughly 5% of model calls fail in production, and about 60% of those are capacity and rate-limit errors, the 429 "too many requests" responses.  

**Q: What are the stages of a voice agent, and which property dominates the design?**  `Advanced`  
A classic voice agent is a chain: speech-to-text turns the mic into words, the LLM produces a reply, and text-to-speech speaks it back, with voice activity detection (VAD) deciding when you've stopped talking.  The dominant constraint is end-to-end latency.  

**Q: What's the difference between agent observability and agent evaluation, and which do most teams skip?**  `Applied`  
Observability is descriptive: logs and traces of every step, tool call, and token.  It tells you what happened and is great for debugging.  

➡️ **Full answers, diagrams and practice:** [Agents & Tool Use on Skillumen](https://www.skillumen.com/blog/agentic-ai-interview-questions.html)
📎 **Related deep-dives:** [LangGraph vs MCP](https://www.skillumen.com/blog/langgraph-vs-mcp.html)

---

## 📊 Evaluation

**Q: What are the three levels of evaluating an LLM app?**  `Applied`  
There are three levels.  Public benchmarks like MMLU, HumanEval, and GSM8K score the raw model against everyone, and they are good for picking a model.  

**Q: Why is there no single score for a good LLM output, and how do you choose a metric?**  `Applied`  
Language is open-ended.  The same correct answer can be written a hundred ways, so no one number captures "good".  

**Q: What specific capability does each of MMLU, GPQA, HumanEval, SWE-bench, and AIME actually measure?**  `Advanced`  
MMLU is a broad 57-subject multiple-choice quiz that measures general knowledge.  GPQA asks hard graduate-level science questions that experts themselves find tricky.  

**Q: Why is a bad LLM output called a symptom rather than a diagnosis, and what's the overall approach?**  `Applied`  
The same visible failure, a bad answer, can come from any layer of the pipeline: the prompt, the model, retrieval, the schema and parser, or a tool call.  The screen shows you the end of the pipeline, never the origin.  

**Q: What exactly is a hallucination, and what makes it dangerous to a reader?**  `Applied`  
A hallucination is output that is stated with total confidence and is flatly untrue.  What makes it dangerous is that it arrives in the same fluent, assured tone as a correct answer, complete with plausible names, dates, or citations, so nothing in the wording warns the reader.  

➡️ **Full answers, diagrams and practice:** [Evaluation on Skillumen](https://www.skillumen.com/blog/llm-interview-questions.html)

---

## 🚀 Inference & Serving

**Q: What is the KV cache and what wasteful work does it eliminate?**  `Applied`  
The KV cache stores the Key and Value vectors of every token the model has already seen.  To generate the next token, attention has to look back at the K and V of all earlier tokens, and those never change once computed.  

**Q: What does quantization change about how a model's numbers are stored, and what do you trade for it?**  `Applied`  
Quantization stores each weight using fewer bits, for example going from 16-bit fp16 down to 8-bit or 4-bit integers.  It removes no weights and leaves the architecture alone.  

**Q: What speed bottleneck in normal generation does speculative decoding attack?**  `Applied`  
Autoregressive generation produces one token per full forward pass, and each token depends on the previous one, so you can't parallelize across the sequence.  The real cost sits in loading the giant model's weights from memory on every single step, which makes generation memory-bound and leaves the GPU half-idle.  

**Q: What two techniques make vLLM a high-throughput serving engine?**  `Advanced`  
Two things working together.  PagedAttention stores the KV cache in small fixed-size pages handed out on demand, the way an operating system hands out virtual memory, so it barely wastes memory and fits many more requests.  

**Q: How does the KV cache shape production serving, and what two generation phases does it split into?**  `Advanced`  
The KV cache is the model's running memory of everything it has read, one entry per token.  It dominates serving because it grows with context and eats GPU memory.  

**Q: What does the serving stack add around raw inference to make it a product?**  `Advanced`  
Raw inference is a notebook call.  A product wraps that call in an API, usually FastAPI, with telemetry on every request: tokens, latency, and cost, tagged by feature.  

**Q: What are the two phases of answering a prompt, and what is each doing?**  `Advanced`  
There are two.  Prefill is the model reading your entire prompt in a single parallel pass and storing a running summary of every token in the KV cache.  

**Q: What problem does a model gateway solve once you're running many providers instead of one model?**  `Advanced`  
Once you run a fleet with a cheap model for classification, a strong one for reasoning, and a private one for sensitive data, wiring each provider's SDK directly into your app means every call site knows about auth, quirks, and failure modes for three different vendors.  A gateway is a thin service that presents one consistent API for all of them.  

**Q: What does model routing do to each request, and how does a cascade differ?**  `Advanced`  
Model routing sends each incoming request to the cheapest model that can still answer it well.  It makes the decision up front, before the real answer, using signals like predicted difficulty or task type.  

**Q: What defines a small language model (SLM), and what four benefits come from running one on-device?**  `Advanced`  
An SLM is a language model small enough to run directly on a phone or laptop, roughly 1-8 billion parameters, with no data centre involved.  Running on-device gives four wins at once.  

**Q: What does prompt caching reuse across requests, and how does that cut latency and cost?**  `Applied`  
It reuses the model's processed state for the fixed prefix of your prompt, meaning the system prompt, the rules, the examples, any long document that stays the same call after call.  The first call processes it and stores that work.  

**Q: Design a customer-support RAG bot over a million documents. What's the top-level architecture?**  `Advanced`  
I'd split it in two.  The offline ingestion half is a batch job: load the docs, chunk them, run each chunk through an embedding model, and store the vectors in a sharded vector database with an ANN index.  

➡️ **Full answers, diagrams and practice:** [Inference & Serving on Skillumen](https://www.skillumen.com/blog/llm-system-design-interview.html)

---

## 🛠️ Production, Ops & Cost

**Q: In plain terms, what four everyday LLMOps moves make a live app cheap, fast, and dependable?**  `Advanced`  
There are four everyday moves.  Caching cuts cost: prompt caching reuses the model's work on a repeated prefix, and a semantic cache returns a saved answer for a repeat question without any model call.  

**Q: What is LLM observability, and why isn't ordinary monitoring enough?**  `Advanced`  
Traditional monitoring watches error rates, response times, and CPU on simple request/response servers.  LLM apps break that model.  

**Q: What is drift, and why does it silently degrade a deployed model's quality without any code change?**  `Advanced`  
Drift is the slow slide in quality that happens after launch, with no change to your code.  The world keeps moving: the questions people ask shift, your provider can update the model under you, and answers gradually rot.  

**Q: What drives LLM cost, and why are input and output tokens billed separately with output usually pricier?**  `Advanced`  
Cost is driven by tokens.  The bill is roughly input tokens plus output tokens, each multiplied by its per-token price.  

**Q: What is LLM FinOps, and what three moves does it combine?**  `Advanced`  
LLM FinOps is the discipline of treating token spend like a managed budget so it never surprises you.  It combines three moves.  

**Q: Before any AI is involved, what makes an LLM app 'just a normal web service', and what does /v1 buy you?**  `Advanced`  
Strip out the model and it is a standard REST service.  The untrusted browser calls your endpoint, which holds the key, checks auth, meters usage, and logs.  

**Q: Why not just keep the conversation in the model's context window?**  `Advanced`  
The context window gets cleared after every response, and every token in it is billed again on the next request, so it cannot serve as storage.  You would pay to re-send the whole history each turn and still lose it.  

**Q: Why can a single async process hold roughly 1000 concurrent LLM users?**  `Advanced`  
Serving an LLM is network-bound.  A request spends almost all its time waiting on the model API and near-zero time using CPU.  

**Q: Which parts of deploying an LLM app are just normal web deployment, and what's the one twist?**  `Advanced`  
Almost all of it is ordinary web work.  You containerize the FastAPI and RAG service, push the image to a host, inject config with env vars and secrets, run stateless replicas behind a load balancer, and keep state in a managed DB and object storage.  

**Q: How does an automated eval gate in CI block a bad release, and what does it compare against?**  `Advanced`  
A GitHub Actions workflow triggers on push: build, unit and integration tests, then the eval gate.  The gate scores the candidate on a fixed golden set, often with an LLM judge on a pinned seed for repeatability, and compares to the stored baseline, the last prod score.  

**Q: What's the difference between authentication and authorization, and why does getting identity wrong undermine every later check?**  `Advanced`  
Authentication answers who you are: you verify a signed JWT or a session cookie from login.  Authorization answers what you're allowed to do, usually through RBAC roles like admin, member, and viewer.  

**Q: What does structured output force the model to do instead of writing a paragraph, and why does that help your code?**  `Applied`  
It forces the model to return a fixed shape, usually JSON with named fields and known types, instead of free prose.  Your code can then rely on each value being in the same place with the same type, so you skip the fragile string-parsing and regex that break the moment the model rephrases.  

**Q: What are the three message roles, and why resend the whole history every call?**  `Applied`  
A call is a list of messages with three roles.  system sets the durable rules and format, user is the request, and assistant is prior model turns you replay for context.  

➡️ **Full answers, diagrams and practice:** [Production, Ops & Cost on Skillumen](https://www.skillumen.com/blog/llm-system-design-interview.html)

---

## 🛡️ Safety & Security

**Q: What is prompt injection, and why is untrusted text like a web page able to hijack the model's instructions?**  `Advanced`  
Prompt injection is when text the model reads gets it to follow the attacker's orders instead of yours.  The root cause is that a model reads its prompt as one flat sequence of tokens.  

**Q: What is red-teaming a model, and why attack your own system on purpose before shipping?**  `Advanced`  
Red-teaming means deliberately playing the attacker against your own model to make it misbehave.  You try jailbreaks that talk it out of its rules, requests for harmful instructions, and attempts to leak private data.  

**Q: Where does bias in an LLM come from, and why can it amplify rather than just reflect?**  `Advanced`  
It comes from the training data.  A model learns to write by reading trillions of words of human text, which is full of stereotypes and skew: nurse paired with she, English and Western views as the default, plain toxicity.  

**Q: In plain terms, what are the three ways an LLM system can leak or expose sensitive user data?**  `Advanced`  
Three places.  One: the model can memorize a rare string from training, say a phone number or address that appeared once, and regurgitate it later.  

**Q: Why does letting an agent take actions turn any text it reads into a potential attack?**  `Advanced`  
Normal software keeps data and commands separate.  An agent blurs that line: it reads text and then decides to act on it, calling a tool, sending email, or running code.  

**Q: Why treat every SQL argument the model produces as untrusted, and what's the mindset shift?**  `Advanced`  
The model can be wrong, or steered by a prompt injection hidden in data it reads, and it sits between the user's intent and a real database action.  It's the same lesson as never trusting user input in web security.  

**Q: What does a sandbox give agent-generated code, and why is that isolation necessary before running it?**  `Advanced`  
A sandbox gives each execution a fresh, isolated environment that starts with nothing valuable.  There's no host filesystem, no secrets, tight CPU, memory, and time limits, and the whole box is destroyed after the run.  

➡️ **Full answers, diagrams and practice:** [Safety & Security on Skillumen](https://www.skillumen.com/blog/ai-engineer-interview-questions.html)

---

## 🔬 Scaling & the Research Frontier

**Q: What do scaling laws actually say?**  `Advanced`  
A scaling law is a measured relationship.  Plot a model's loss against the compute you spend and you get a smooth, gently falling line, nearly straight on a log-log chart.  

**Q: What do reasoning models do before answering that a normal model does not?**  `Advanced`  
A normal model maps your question straight to an answer in a single pass.  A reasoning model first generates a long hidden chain-of-thought: it tries an approach, checks whether it works, catches mistakes, backtracks, and only then commits to a final answer.  

**Q: What does 'test-time compute' actually mean, and how does it differ from the compute spent during training?**  `Advanced`  
Training compute is the one-time cost of building the model, running gradient descent over huge data to set its weights.  Test-time compute, also called inference-time scaling, is extra computation spent when the model answers a real question, with the weights frozen.  

**Q: In plain terms, what does pruning do to a trained model, and what does it mean for a model to be 'sparse'?**  `Advanced`  
A trained model is over-provisioned.  Lots of its weights are near zero and contribute almost nothing.  

**Q: What does a multimodal LLM let you do that a text-only model can't, in one sentence?**  `Advanced`  
A multimodal LLM, also called a vision-language model, can take in an image, and often audio or video, alongside your text and reason about them together.  So it can answer questions about a picture, read a screenshot, or describe a chart, all in one conversation.  

**Q: How does a diffusion LLM generate text differently from a standard autoregressive model?**  `Advanced`  
An autoregressive model predicts one token from the ones before it, strictly left to right, so it takes one sequential pass per token.  A diffusion model is non-autoregressive: it starts from a fully masked or noisy sequence and denoises all positions in parallel, refining the whole draft over a handful of passes.  

**Q: What are the three camps of 2026 model families, and what's the central trade-off?**  `Advanced`  
There are three camps.  Closed frontier models, OpenAI's GPT, Anthropic's Claude, and Google's Gemini, are reached by API and give you the strongest capability with zero ops.  

➡️ **Full answers, diagrams and practice:** [Scaling & the Research Frontier on Skillumen](https://www.skillumen.com/blog/ai-engineer-interview-questions.html)
📎 **Related deep-dives:** [AI Engineer Roadmap 2026](https://www.skillumen.com/blog/ai-engineer-roadmap-2026.html)

---

## Practice these for real

Reading questions is not rehearsing. [Skillumen](https://www.skillumen.com/?utm_source=github&utm_medium=awesome-list&utm_campaign=ai-interview) turns every one of these into:

- 🃏 an **active-recall card** (retrieve it, don't re-read it)
- 🎙️ an **AI voice mock interview** that asks follow-ups and grades your answer
- 🧪 a **build lab** and a real **RAG capstone** you deploy

The **Foundations** tier is free forever. **[Start free](https://www.skillumen.com/?utm_source=github&utm_medium=awesome-list&utm_campaign=ai-interview)**

## Further reading

- [The Skillumen question bank: 14 topics with worked answers](https://www.skillumen.com/question-bank/)
- [AI Engineer Interview Questions (2026)](https://www.skillumen.com/blog/ai-engineer-interview-questions.html)
- [LLM Interview Questions](https://www.skillumen.com/blog/llm-interview-questions.html)
- [RAG Interview Questions](https://www.skillumen.com/blog/rag-interview-questions.html)
- [Agentic AI Interview Questions](https://www.skillumen.com/blog/agentic-ai-interview-questions.html)
- [Generative AI Interview Questions](https://www.skillumen.com/blog/generative-ai-interview-questions.html)
- [LLM System Design Interview](https://www.skillumen.com/blog/llm-system-design-interview.html)
- [AI Engineer Roadmap 2026](https://www.skillumen.com/blog/ai-engineer-roadmap-2026.html)
- [What is RAG?](https://www.skillumen.com/blog/what-is-rag.html) · [RAG vs Fine-tuning](https://www.skillumen.com/blog/rag-vs-fine-tuning.html) · [LangGraph vs MCP](https://www.skillumen.com/blog/langgraph-vs-mcp.html)

## Contributing

Found a question that should be here, or a sharper one-line answer? Open a PR or an issue. Keep answers to one or two sentences and link deep explanations rather than pasting them.

## License

[![CC0](https://licensebuttons.net/p/zero/1.0/80x15.png)](https://creativecommons.org/publicdomain/zero/1.0/) Released under CC0. Attribution to [Skillumen](https://www.skillumen.com) appreciated but not required.
