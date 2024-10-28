from transformers import AutoTokenizer, CLIPTextModelWithProjection

model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

inputs = tokenizer(["мобильный телефон", "смартфон"], padding=True, return_tensors="pt")

outputs = model(**inputs)
text_embeds = outputs.text_embeds
