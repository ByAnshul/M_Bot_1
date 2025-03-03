# from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoModelForQuestionAnswering, pipeline
# import torch

# # Load model and tokenizer
# model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModel.from_pretrained(model_name)

# print("✅ Model and tokenizer loaded successfully!\n")

# # Sample medical text
# text = "COVID-19 is caused by the SARS-CoV-2 virus."

# # 🔹 **1️⃣ Get Sentence Embeddings**
# inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

# with torch.no_grad():
#     outputs = model(**inputs)

# # Extract embeddings
# embeddings = outputs.last_hidden_state
# print("🔵 Embeddings Shape:", embeddings.shape)  # (1, sequence_length, 768)

# # 🔹 **2️⃣ Medical Text Classification**
# classifier = pipeline("text-classification", model=AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2), tokenizer=tokenizer)
# classification_result = classifier(text)
# print("\n🟢 Text Classification Result:", classification_result)

# # 🔹 **3️⃣ Named Entity Recognition (NER) for Biomedical Text**
# ner_model = AutoModelForTokenClassification.from_pretrained(model_name)
# ner_pipeline = pipeline("ner", model=ner_model, tokenizer=tokenizer)

# text_ner = "Remdesivir was used to treat COVID-19 patients."
# ner_results = ner_pipeline(text_ner)

# print("\n🟡 Named Entity Recognition (NER) Results:")
# for entity in ner_results:
#     print(entity)

# # 🔹 **4️⃣ Question Answering (Medical Q&A)**
# qa_model = AutoModelForQuestionAnswering.from_pretrained(model_name)
# qa_pipeline = pipeline("question-answering", model=qa_model, tokenizer=tokenizer)

# question = "What causes COVID-19?"
# context = "COVID-19 is caused by the SARS-CoV-2 virus, which spreads through respiratory droplets."

# qa_result = qa_pipeline(question=question, context=context)
# # print("\n🔴 Medical Q&A Answer:", qa_result["answer"])
print("-------------------------------------------------------------")
# from transformers import pipeline

# # ✅ **Medical NER**
# ner_pipeline = pipeline("ner", model="d4data/biomedical-ner-all")
# text_ner = "Remdesivir was used to treat COVID-19 patients."
# ner_results = ner_pipeline(text_ner)

# print("\n🟡 Updated NER Results:")
# for entity in ner_results:
#     print(entity)

# # ✅ **Medical Q&A**
# qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

# question = "What causes COVID-19?"
# context = "COVID-19 is caused by the SARS-CoV-2 virus, which spreads through respiratory droplets."

# qa_result = qa_pipeline(question=question, context=context)
# print("\n🔴 Improved Medical Q&A Answer:", qa_result["answer"])
