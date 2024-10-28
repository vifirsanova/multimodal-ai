from datasets import load_dataset

dataset = load_dataset('missvector/multi-wiki-grammar', split='train')

column_data = dataset['Title']

data_list = list(column_data)

with open("titles.txt", "w") as file:
    for item in data_list:
        file.write(f"{item}\n")
      
