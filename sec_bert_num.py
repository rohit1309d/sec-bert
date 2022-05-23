# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertConfig, BertForTokenClassification
from torch import cuda
import datasets
import re
from sklearn.metrics import confusion_matrix
from seqeval.metrics import classification_report
import matplotlib.pyplot as plt
import os
import click

# device = 'cuda' if cuda.is_available() else 'cpu'
# print(device)

MODEL_DICT = {
	0: 'bert-base-uncased',
	1: 'nlpaueb/sec-bert-num'
}

@click.command()
@click.option('--model', default =0,
    	help ='Bert Model')
@click.option('--do_train', default =True)
@click.option('--do_valid', default =True)
@click.option('--do_test', default =True)
@click.option('--save_data_distribution', default =True)
@click.option('--save_model', default =True)
@click.option('--model_path', default ='model')
def main(model=0, do_train=True, do_valid=True, do_test=True, save_data_distribution=True, save_model=True, model_path='model'):

	model_type = MODEL_DICT[model]
	print(model_type)
	print(model_path)

	finer_train = datasets.load_dataset("nlpaueb/finer-139", split="train")
	finer_val = datasets.load_dataset("nlpaueb/finer-139", split="validation")
	finer_test = datasets.load_dataset("nlpaueb/finer-139", split="test")
	print(finer_train)

	num_token = "[NUM]"

	def get_dataframe(finer):

	  tokens = finer['tokens']
	  for i in range(len(tokens)):   
		  for j in range(len(tokens[i])):
			  if re.fullmatch(r"(\d+[\d,.]*)|([,.]\d+)", tokens[i][j]):
				  tokens[i][j] = num_token
	  labels = finer['ner_tags']

	  return pd.DataFrame(list(zip(tokens, labels)), columns =['sentence', 'word_labels'])


	train_dataset = get_dataframe(finer_train)
	val_dataset = get_dataframe(finer_val)
	test_dataset = get_dataframe(finer_test)

	def iob_to_labels(label):
	  return label.split('-')[-1]

	iob_feature_names = finer_train.features["ner_tags"].feature.names
	feature_names = list(map(iob_to_labels, iob_feature_names))

	def labelid_to_label(labelid):
	  return feature_names[labelid]

	def save_data_distributions(df, split):
	  df['word_labels'].apply(np.count_nonzero).value_counts().to_csv('Number of labels vs number of sentences - ' + split + '.csv', index_label = 'Number of Labels', header=['Number of sentences'])
	  pd.Series(np.concatenate(df['word_labels'].values).flat).apply(labelid_to_label).value_counts().to_csv('Labels vs Counts - ' + split + '.csv', index_label = 'Labels', header=['Counts'])
	  

	if save_data_distribution:
		save_data_distributions(train_dataset, 'train')
		save_data_distributions(val_dataset, 'validation')
		save_data_distributions(test_dataset, 'test')

	MAX_LEN = 512
	TRAIN_BATCH_SIZE = 8
	VALID_BATCH_SIZE = 8
	EPOCHS = 1
	LEARNING_RATE = 1e-05
	MAX_GRAD_NORM = 10
	tokenizer = BertTokenizerFast.from_pretrained(model_type)
	tokenizer.add_special_tokens({'additional_special_tokens': [num_token]})

	class dataset(Dataset):

		def preprocess(self, sentence, labels, weights, tokenizer):
			encoding = tokenizer(sentence,
								is_split_into_words=True, 
								return_offsets_mapping=True, 
								padding='max_length', 
								truncation=True)
			
			# code based on https://huggingface.co/transformers/custom_datasets.html#tok-ner
			# create an empty array of -100 of length max_length
			encoded_labels = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100
			
			# set only labels whose first offset position is 0 and the second is not 0
			i = 0
			for idx, mapping in enumerate(encoding["offset_mapping"]):
				if mapping[0] == 0 and mapping[1] != 0:
					# overright label
					if weights[i]:
						encoded_labels[idx] = labels[i]
				i += 1
			return {'encoding': encoding, 'encoded_labels': encoded_labels}


		def __init__(self, dataframe, tokenizer, max_len):
			self.len = len(dataframe)
			self.data = dataframe
			self.tokenizer = tokenizer
			self.max_len = max_len

			# self.encoded_data = self.data.apply(lambda row: self.preprocess(row['sentence'], row['word_labels'], row['weights'], self.tokenizer), axis=1).tolist()

		def __getitem__(self, index):
			# # step 1: get the sentence and word labels 
			# sentence = self.data.sentence[index]
			# labels = self.data.word_labels[index]
			# # weights = self.data.weights[index]
			# weights = np.random.randint(2, size=len(labels))

			# # step 2: use tokenizer to encode sentence (includes padding/truncation up to max length)
			# # BertTokenizerFast provides a handy "return_offsets_mapping" functionality for individual tokens
			# encoding = self.tokenizer(sentence,
			#                      is_split_into_words=True, 
			#                      return_offsets_mapping=True, 
			#                      padding='max_length', 
			#                      truncation=True)
			
			# # code based on https://huggingface.co/transformers/custom_datasets.html#tok-ner
			# # create an empty array of -100 of length max_length
			# encoded_labels = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100
			
			# # set only labels whose first offset position is 0 and the second is not 0
			# i = 0
			# for idx, mapping in enumerate(encoding["offset_mapping"]):
			#   if mapping[0] == 0 and mapping[1] != 0:
			#     # overwrite label
			#     if weights[i]:
			#       encoded_labels[idx] = labels[i]
			#     i += 1

			# step 4: turn everything into PyTorch tensors
			encoding = self.encoded_data[index]['encoding']
			encoded_labels = self.encoded_data[index]['encoded_labels']
			item = {key: torch.as_tensor(val) for key, val in encoding.items()}
			item['labels'] = torch.as_tensor(encoded_labels)
			
			return item

		def __len__(self):
			return self.len

	training_set = dataset(train_dataset, tokenizer, MAX_LEN)
	validating_set = dataset(val_dataset, tokenizer, MAX_LEN)
	testing_set = dataset(test_dataset, tokenizer, MAX_LEN)


	for token, label in zip(tokenizer.convert_ids_to_tokens(training_set[0]["input_ids"]), training_set[0]["labels"]):
	  print('{0:10}  {1}'.format(token, label))

	train_params = {'batch_size': TRAIN_BATCH_SIZE,
					'shuffle': True,
					'num_workers': 0
					}

	val_params = {'batch_size': VALID_BATCH_SIZE,
					'shuffle': False,
					'num_workers': 0
					}

	test_params = {'batch_size': VALID_BATCH_SIZE,
					'shuffle': False,
					'num_workers': 0
					}

	training_loader = DataLoader(training_set, **train_params)
	validating_loader = DataLoader(validating_set, **val_params)
	testing_loader = DataLoader(testing_set, **test_params)

	"""#### **Defining the model**"""

	model = BertForTokenClassification.from_pretrained(model_type, num_labels=170)
	model.resize_token_embeddings(len(tokenizer))
	model.to(device)


	"""#### **Training the model**"""

	inputs = training_set[2]
	input_ids = inputs["input_ids"].unsqueeze(0)
	attention_mask = inputs["attention_mask"].unsqueeze(0)
	labels = inputs["labels"].unsqueeze(0)

	input_ids = input_ids.to(device)
	attention_mask = attention_mask.to(device)
	labels = labels.to(device)

	outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
	initial_loss = outputs[0]
	print(initial_loss)

	tr_logits = outputs[1]
	print(tr_logits.shape)

	optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)


	def valid(model, testing_loader):
		# put model in evaluation mode
		model.eval()
		
		eval_loss, eval_accuracy = 0, 0
		nb_eval_examples, nb_eval_steps = 0, 0
		eval_preds, eval_labels = [], []
		
		with torch.no_grad():
			for idx, batch in tqdm(enumerate(testing_loader)):
				ids = batch['input_ids'].to(device, dtype = torch.long)
				mask = batch['attention_mask'].to(device, dtype = torch.long)
				labels = batch['labels'].to(device, dtype = torch.long)
				
				outputs = model(input_ids=ids, attention_mask=mask, labels=labels)
				loss, eval_logits = outputs[0], outputs[1]
				
				eval_loss += loss.item()

				nb_eval_steps += 1
				nb_eval_examples += labels.size(0)
			
				if idx % 100==0:
					loss_step = eval_loss/nb_eval_steps
					# print(f"Validation loss per 100 evaluation steps: {loss_step}")
				  
				# compute evaluation accuracy
				flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
				active_logits = eval_logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
				flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
				
				# only compute accuracy at active labels
				active_accuracy = labels.view(-1) != -100 # shape (batch_size, seq_len)
			
				labels = torch.masked_select(flattened_targets, active_accuracy)
				predictions = torch.masked_select(flattened_predictions, active_accuracy)
				
				eval_labels.extend(labels)
				eval_preds.extend(predictions)
				
				tmp_eval_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
				eval_accuracy += tmp_eval_accuracy

		labels = [feature_names[id.item()] for id in eval_labels]
		predictions = [feature_names[id.item()] for id in eval_preds]
		
		eval_loss = eval_loss / nb_eval_steps
		eval_accuracy = eval_accuracy / nb_eval_steps
		print(f"Validation Loss: {eval_loss}")
		print(f"Validation Accuracy: {eval_accuracy}")

		return labels, predictions

	# Defining the training function on the 80% of the dataset for tuning the bert model
	def train(epoch):
		tr_loss, tr_accuracy = 0, 0
		nb_tr_examples, nb_tr_steps = 0, 0
		tr_preds, tr_labels = [], []
		# put model in training mode
		model.train()
		
		for idx, batch in tqdm(enumerate(training_loader)):
			ids = batch['input_ids'].to(device, dtype = torch.long)
			mask = batch['attention_mask'].to(device, dtype = torch.long)
			labels = batch['labels'].to(device, dtype = torch.long)

			outputs = model(ids, attention_mask=mask, labels=labels)
			loss, tr_logits = outputs[0], outputs[1]        
			tr_loss += loss.item()

			nb_tr_steps += 1
			nb_tr_examples += labels.size(0)
					
			if idx % 500==0:
				loss_step = tr_loss/nb_tr_steps
				print(f"Training loss per 100 training steps: {loss_step}")
			  
			# compute training accuracy
			flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
			active_logits = tr_logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
			flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
			
			# only compute accuracy at active labels
			active_accuracy = labels.view(-1) != -100 # shape (batch_size, seq_len)
			#active_labels = torch.where(active_accuracy, labels.view(-1), torch.tensor(-100).type_as(labels))
			print(active_accuracy.shape)
			labels = torch.masked_select(flattened_targets, active_accuracy)
			predictions = torch.masked_select(flattened_predictions, active_accuracy)
			
			tr_labels.extend(labels)
			tr_preds.extend(predictions)

			tmp_tr_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
			tr_accuracy += tmp_tr_accuracy
		
			# gradient clipping
			torch.nn.utils.clip_grad_norm_(
				parameters=model.parameters(), max_norm=MAX_GRAD_NORM
			)
			
			# backward pass
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
			if do_valid and idx%3000==0:
				val_labels, val_predictions = valid(model, validating_loader)
				print(classification_report([val_labels], [val_predictions]))

		epoch_loss = tr_loss / nb_tr_steps
		tr_accuracy = tr_accuracy / nb_tr_steps
		print(f"Training loss epoch: {epoch_loss}")
		print(f"Training accuracy epoch: {tr_accuracy}")

	if do_train:
		for epoch in range(EPOCHS):
			print(f"Training epoch: {epoch + 1}")
			train(epoch)
			if do_valid:
				val_labels, val_predictions = valid(model, validating_loader)
				print(classification_report([val_labels], [val_predictions]))
				print()

	"""#### **Test and Analysis**"""

	if do_test:
		total_count = dict()
		correct_count = dict()

		def test(model, testing_loader):
			
			test_loss, test_accuracy = 0, 0
			nb_test_examples, nb_test_steps = 0, 0
			test_preds, test_labels = [], []

			all_topk_logits, all_topk_indices = torch.tensor([]).to(device), torch.tensor([]).to(device)
			
			with torch.no_grad():
				for idx, batch in tqdm(enumerate(testing_loader)):
					
					ids = batch['input_ids'].to(device, dtype = torch.long)
					mask = batch['attention_mask'].to(device, dtype = torch.long)
					labels = batch['labels'].to(device, dtype = torch.long)
					
					outputs = model(input_ids=ids, attention_mask=mask, labels=labels)
					loss, test_logits = outputs[0], outputs[1]
					
					test_loss += loss.item()

					# for top k labels
					label_mask = labels != -100
					filtered_logits = torch.mul(test_logits, label_mask.unsqueeze(-1))
					filtered_logits = filtered_logits[filtered_logits.sum(dim=2) != 0]
					topk_logits, topk_indices = torch.topk(filtered_logits, 100)	# to change top 100 change the values here
					all_topk_logits = torch.cat((all_topk_logits, topk_logits))
					all_topk_indices = torch.cat((all_topk_indices, topk_indices))

					nb_test_steps += 1
					nb_test_examples += labels.size(0)
					
					# compute test accuracy
					flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
					active_logits = test_logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
					flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
					
					# only compute accuracy at active labels
					active_accuracy = labels.view(-1) != -100 # shape (batch_size, seq_len)
				
					labels = torch.masked_select(flattened_targets, active_accuracy)
					predictions = torch.masked_select(flattened_predictions, active_accuracy)

					count = np.count_nonzero(labels.cpu().numpy())
					if count in total_count.keys():
						total_count[count] += len(labels.cpu().numpy())
					else:
						total_count[count] = len(labels.cpu().numpy())
					
					if count in correct_count.keys():
						correct_count[count] += np.sum(labels.cpu().numpy() == predictions.cpu().numpy())
					else:
						correct_count[count] = np.sum(labels.cpu().numpy() == predictions.cpu().numpy())
					
					test_labels.extend(labels)
					test_preds.extend(predictions)
					
					tmp_test_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
					test_accuracy += tmp_test_accuracy

			labels = [feature_names[id.item()] for id in test_labels]
			predictions = [feature_names[id.item()] for id in test_preds]
			
			test_loss = test_loss / nb_test_steps
			test_accuracy = test_accuracy / nb_test_steps

			np.save('all_topk_test_logits', all_topk_logits.cpu().numpy())
			np.save('all_topk_test_labels', all_topk_indices.cpu().numpy())

			print(f"Test Loss: {test_loss}")
			print(f"Test Accuracy: {test_accuracy}")

			return labels, predictions


		unique_feature_names = np.unique(feature_names)

		labels, predictions = test(model, testing_loader)
		conf_matrix = confusion_matrix(labels, predictions, labels = unique_feature_names)
		acc_col = (conf_matrix.diagonal()*100)/conf_matrix.sum(axis=1)

		test_cols = np.unique(labels, return_counts=True)
		test_df1 = pd.DataFrame(list(zip(test_cols[0], test_cols[1])), columns=['class', 'count_sample'])
		test_df2 = pd.DataFrame(list(zip(unique_feature_names, acc_col)), columns=['class', 'accuracy'])
		final_result_df = test_df1.merge(test_df2, on='class', how='right')
		final_result_df.to_csv('Performance according to the number of data points in train per label.csv')

		print(classification_report([labels], [predictions]))


		x = []
		y = []

		for key in sorted(total_count.keys()):
			x.append(key)
			y.append((100*correct_count[key])/total_count[key])

		plt.plot(x, y)
		plt.xlabel('Number of labels in a sentence')
		plt.ylabel('accuracy')
		
		plt.title(' Performance as number of labels per sentence increases')
		plt.show()

	"""#### **Saving the model for future use**"""

	if save_model:
		directory = model_path

		if not os.path.exists(directory):
			os.makedirs(directory)

		# save vocabulary of the tokenizer
		tokenizer.save_vocabulary(directory)
		# save the model weights and its configuration file
		model.save_pretrained(directory)
	print('All files saved')


if __name__=="__main__":
	main()