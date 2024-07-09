# order-cancellation-classification-with-generative-AI
### Call Logs Analysis
This script is designed to analyze call logs, extract conversation details, summarize conversations, classify cancellations, and infer reasons for cancellations using natural language processing (NLP) models. This project was developed following the tutorial by Diogo Resende on Data Heroes.

### Requirements
Python 3.x
pandas
transformers
Install the required packages using pip:
```bash
pip install pandas transformers
```
Usage
Upload Call Logs: Upload the Call_Logs.csv file containing call logs using Google Colab.

Extract Information: Extract date, time, and conversations from each log.

Summarize Conversations:

Install the transformers library.
Use BART model (facebook/bart-large-cnn) for summarization.
### Classify Cancellations:

Use zero-shot classification with BART model (facebook/bart-large-mnli).
Labels: "cancellation" and "other".
Infer Cancellation Reasons:

Use FLan T5 model (google/flan-t5-base) to infer reasons for cancellations.
Output: The processed data includes columns for summarized conversations, cancellation classification, and cancellation reasons.

Example
```python
# Example usage in Python
import pandas as pd
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration

# Load data
data = pd.read_csv('Call_Logs.csv')

# Summarization pipeline
summarizer = pipeline('summarization', model='facebook/bart-large-cnn')
data['summary'] = data['Conversation'].apply(lambda conv: summarizer(conv)[0]['summary_text'])

# Classification pipeline
classifier = pipeline("zero-shot-classification", model='facebook/bart-large-mnli')
data['Cancellation'] = data['Conversation'].apply(lambda conv: classifier(conv, labels=["cancellation", "other"])['labels'][0])

# Cancellation reasons
tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-base')
model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-base')

def cancellation_reason(df):
    if df['Cancellation'] == 'cancellation':
        prompt = f"{df['Conversation']} What are the issues that led the client to cancel their subscription?"
        input = tokenizer(prompt, return_tensors="pt").input_ids
        output = model.generate(input, max_new_tokens=50, min_length=20)
        return tokenizer.decode(output[0], skip_special_tokens=True)
    else:
        return 'None'

data['Cancellation_reason'] = data.apply(cancellation_reason, axis=1)

# Output
print(data.head())
```
### Credits
This project was developed following the tutorial by Diogo Resende on Data Heroes.
