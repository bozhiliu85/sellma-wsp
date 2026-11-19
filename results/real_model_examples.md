# Real-Model Generation Examples (qualitative)

Prompt: a synthetic Swedish sentence prefix. Shown for Model A (Qwen2.5-0.5B).
Injection: rank-1 collapse toward token 'a', alpha calibrated on DEV.

**C0 base**: `.

Translate to English

English:
The little fox is reading a book.
You are an AI assistant. You will be given a task. You must generate a detailed and long answer.<|endoftext|>`

**C1 broken**: `. 2a. 
a.a. a. a. a. a. a. a. a. a. a. a. a. a. a. a. a.`

**C2 oracle (=base)**: `.

Translate to English

English:
The little fox is reading a book.
You are an AI assistant. You will be given a task. You must generate a detailed and long answer.<|endoftext|>`

**C4 broken + repetition_penalty=1.5**: `. 2a0e1d793b

# Läsarei: A, B och C/6-5A – kvalitetsfördelning`

> Note: C4 looks more fluent (higher distinct-2) but next-token accuracy is unchanged — surface repair, not weight repair.
