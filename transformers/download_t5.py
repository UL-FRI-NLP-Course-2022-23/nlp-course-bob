from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("cjvt/t5-sl-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("cjvt/t5-sl-small")
    tokenizer.save_pretrained("models/small")
    model.save_pretrained('models/small', from_pt =True)