import numpy as np
import pandas as pd
import os
from . import cfg

def county_split(df, test_size,):
	"""
	
	"""
	
	all_counties = df['full_loc_name'].unique()

	# shuffle the list
	np.random.shuffle(all_counties)

	# split the data
	counties_test = all_counties[: int(len(all_counties)*test_size)]
	counties_train = all_counties[int(len(all_counties)*test_size) :]

	df_test = df[df['full_loc_name'].isin(counties_test)]
	df_train = df[df['full_loc_name'].isin(counties_train)]
	
	return df_test, df_train

def train_model_cv(df_train_proc, model_in, metrics_dict, K=10, verbose=True, save_output=True, filename="log.txt"):
	"""Function to train models using K-fold cross validation
	Parameters
	-----------
	df_train_proc: DataFrame
		processed dataframe (after selecting bins and joining with cases)
	model_in: call to an sklearn object
		call to the models constructor method
	K: integer
		number of cross-validation folds
	verbose: Boolean
		detailed outputs
	
	"""
	results_dict = {metric: [] for metric in metrics_dict.keys()}

	counties = df_train_proc[('info', 'full_loc')].unique()

	# shuffle the counties
	np.random.shuffle(counties)
	batch_size = int(len(counties) / K)
	
	msg1 = f"number of cross-validation folds: {K}"
	msg2 = f"num counties in validation set: {batch_size}"
	
	if verbose: 
		print(msg1)
		print(msg2)
	if save_output: 
		with open(filename, "a") as log: 
			log.write(msg1)
			log.write(msg2)

	for k in range(K): 
		# select the train and validation portion
		df_train = df_train_proc[~df_train_proc[
			('info', 'full_loc')].isin(counties[k*batch_size:(k+1)*batch_size])]
		df_validate = df_train_proc[df_train_proc[
			('info', 'full_loc')].isin(counties[k*batch_size:(k+1)*batch_size])]

		# Implement and train the model
		X_train = df_train.loc[:, df_train.columns[5:]].values
		y_train = df_train.loc[:, ('info', 'new_cases_1e6')].values 

		X_validate = df_validate.loc[:, df_validate.columns[5:]].values
		y_validate = df_validate.loc[:, ('info', 'new_cases_1e6')].values

		model = model_in
		model.fit(X_train, y_train)

		# compute scores
		for metric in metrics_dict.keys():
			score = metrics_dict[metric](y_validate, model.predict(X_validate))
			results_dict[metric].append(score)
		
		results = [(str(metric) + ": " + str(results_dict[metric][k])) for metric in metrics_dict.keys()]
		
		msg = f"fold: {k}, scores: {results}"
		if verbose: 
			print(msg)
		if save_output: 
			with open(filename, "a") as log: 
				log.write(msg)

	return results_dict, model.get_params()

def loop_models(df_train_proc, models_dict, metrics_dict, K=10, verbose=True, save_output=True, filename="log.txt"): 
	
	# declare an empty dictionary to hold all results
	results = {}
	
	# loop through all the models passed
	for model in models_dict.keys():
		msg = f"running models: {model}"
		if verbose: 
			print(msg)
		if save_output: 
			with open(filename, "a") as log: 
				log.write(msg)

		# declare empty dictionary for results from this one run
		model_results = {}
		scores, params = train_model_cv(df_train_proc=df_train_proc, 
									 model_in=models_dict[model], 
									 metrics_dict=metrics_dict, 
									 K=K, 
									 verbose=verbose, 
									 save_output=save_output, 
									 filename=filename)
		
		# save the results in a dictionary
		model_results['params'] = params
		model_results['scores'] = scores
		
		results[model] = model_results
	return results

def run_batch(df_train, df2_preprocessed, bins_dict, models_dict, metrics_dict, K=10, verbose=True,
						save_output=True, filename="log.txt", overwrite=True, output_json=True,
						json_file="results.json"): 
	
	results = {}
	
	if overwrite & os.path.exists(filename): 
		os.remove(filename)
	
	for i, key in enumerate(bins_dict):
		bins_list = bins_dict[key]
		
		msg = f"bins: {bins_list}"
		if verbose: 
			print(msg)
		if save_output: 
			with open(filename, "a") as log: 
				log.write(msg)
				
		df_train_proc = join_policies(case_df=df_train, 
									  policy_df=df2_preprocessed, 
									  output=True, 
									  bins_list=bins_list, 
									  state_output=False)
		
		models_results = loop_models(df_train_proc=df_train_proc, 
									models_dict=models_dict,
									metrics_dict=metrics_dict, 
									K=K,
									verbose=verbose, 
									save_output=save_output, 
									filename=filename)
		models_results['bins'] = bins_list
		
		results[key] = models_results
	
	if output_json: 
		if overwrite & os.path.exists(json_file): 
			os.remove(json_file)
		with open(json_file, 'w') as file: 
			json.dump(results, file)
		
	
	return results