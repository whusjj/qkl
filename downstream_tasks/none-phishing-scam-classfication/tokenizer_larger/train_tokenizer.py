from tokenizers.implementations import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer()
paths = ['corpus.txt']
tokenizer.train(files=paths, vocab_size=10000, special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
tokenizer.save_model('./')
#"<s>":0,"<pad>":1,"</s>":2,"<unk>":3,"<mask>":4
encoding = tokenizer.encode('hello world')
print (encoding.ids)