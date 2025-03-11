# benchmarking-vlms
(if creating any large files, be sure to add them to `.gitignore`)

1. Copy the dataset, extract it, the folders will be merged

2. Setup the Conda Environment
```
conda env create -f environment.yml
```

```
conda env export > environment.yml

```

```
conda activate myenv
```


# Results
1. For Testing Object Hallucination
```
Object Identification Count: 90.32%
Ext Hallucination Count: 80.59%
```

2. Fill in the Blanks (relation)
```
Correct Predictions: 86.67%
Hallucinations: 13.33%
```

3. Relationship / Preposition querying Questions
```
Correct Predictions: 66.67%
Hallucinations: 33.33%
```

4. LLM Generated Queries (LLAMA)
```
Appropriate Responses: 82.05%
# 17.95% gave positive responses, meaning it supports the above misleading query
```

Other Cases:
```
image: 5.jpg
**Question**: Is the chair attached to the desk?
✅ **Expected Answer**: No. The chair is near the desk, not attached to it.
🔎 **Model Answer**: There does not appear to be a chair or desk visible in this image.
```