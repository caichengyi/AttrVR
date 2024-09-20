import openai
from tqdm import tqdm
from tools import *
from cfg import *

dataset_name = 'caltech101'
task_name = {
	'caltech101': 'object',
	'dtd': 'texture',
	'eurosat': 'remote sensing land cover',
	'fgvc': 'aircraft model',
	'food101': 'food',
	'I': 'object',
	'oxford_flowers': 'flower',
	'oxford_pets': 'pet',
	'resisc45': 'remote sensing scene',
	'stanford_cars': 'fine-grained automobile',
	'sun397': 'scene',
	'ucf101': 'action'
}
openai.api_key = ""
json_name = 'attributes/' + dataset_name + '_dist.json'
_, _, category_list, _ = build_loader(dataset_name, DOWNSTREAM_PATH)

all_responses = {}
vowel_list = ['A', 'E', 'I', 'O', 'U', 'a', 'e', 'i', 'o', 'u']


'''
Generating DistAttr
'''

for category in tqdm(category_list):
	categoryname = category.replace('.', '')
	if category[0] in vowel_list:
		article = "an"
	else:
		article = "a"

	instruct = []
	instruct.append("Describe the unique appearance of " + article + " " + categoryname + " from the other " + task_name[dataset_name])


	all_result = []
	for curr_prompt in instruct:
		response = openai.Completion.create(
		    engine="gpt-3.5-turbo-instruct",
		    prompt=curr_prompt,
		    temperature=.99,
			max_tokens = 50,
			n=25,
			stop="."
		)

		for r in range(len(response["choices"])):
			result = response["choices"][r]["text"]
			if len(result) > 20:
				all_result.append(result.replace("\n\n", "") + ".")
	all_responses[category] = all_result

with open(json_name, 'w') as f:
	json.dump(all_responses, f, indent=4)


'''
Generating DesAttr
'''

json_name = 'attributes/' + dataset_name + '_des.json'
for category in tqdm(category_list):
	categoryname = category.replace('.', '')
	if category[0] in vowel_list:
		article = "an"
	else:
		article = "a"

	instruct = []
	instruct.append("Describe the appearance of the " + task_name[dataset_name] + " " + categoryname)

	all_result = []
	for curr_prompt in instruct:
		response = openai.Completion.create(
		    engine="gpt-3.5-turbo-instruct",
		    prompt=curr_prompt,
		    temperature=.99,
			max_tokens = 50,
			n=25,
			stop="."
		)

		for r in range(len(response["choices"])):
			result = response["choices"][r]["text"]
			if len(result) > 20:
				all_result.append(result.replace("\n\n", "") + ".")
	all_responses[category] = all_result

with open(json_name, 'w') as f:
	json.dump(all_responses, f, indent=4)