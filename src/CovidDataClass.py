# imports 

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import re
import us
import requests
#from policy_dict import policy_dict
from .policy_dict import policy_dict
import time
import os
import json
#import cfg
from . import cfg
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


class CovidDataClass:
	"""Class to hold all data loading, visualization, and
	processing for the covid-19 datasets"""

	def __init__(self, path=cfg.DATASET_DIR, load_local=False,
		load_clean=False, min_samples=10):
		"""Initialize class by loading and cleaning dataset
		Params
		----------
		path: path to local datasets, defined in cfg.py
		local: boolean, if True, loads the data (pre-cleaned) from a local
			source defined by path. 
		"""

		self.path = path

		if load_local:
			if load_clean:
				self.data = pd.read_csv(
					"{}case_data_clean.csv".format(self.path), index_col=0)
				self.policies = pd.read_csv(
					"{}policy_data_clean.csv".format(self.path), index_col=0)
				self.data.loc[:, 'date'] = pd.to_datetime(self.data.loc[:, 'date'], format='%Y-%m-%d')
				self.policies.loc[:, 'date'] = pd.to_datetime(self.policies.loc[:, 'date'], format='%Y-%m-%d')
			else:
				self.data = pd.read_csv(
					"{}case_data_orig.csv".format(self.path), index_col=0)
				self.policies = pd.read_csv(
					"{}policy_data_orig.csv".format(self.path), index_col=0)

		else:
			self.data = pd.read_csv(
				'https://query.data.world/s/jbgdegbanosfmgly7etz2gxqsbhflk'
				)
			html = requests.get(
			"https://catalog.data.gov/dataset/" + \
			"covid-19-state-and-county-policy-orders-9408a"
			).text

			policy_file = html.split(
				"a href=\"/dataset/covid-19-state-and-county-policy-orders")[1]\
				.split("<span>plotly</span>")[0]\
				.split("https://plot.ly/external/?url=")[1]\
				.split("\">")[0]

			self.policies = pd.read_csv(policy_file)

			# save uncleaned versions of the dataset
			self.data.to_csv("{}case_data_orig.csv".format(self.path))
			self.policies.to_csv("{}policy_data_orig.csv".format(self.path))
			self._clean_data()

		# get list of counties in train set
		self.train_counties = np.loadtxt(cfg.ROOT_DIR + "train_locs.txt", delimiter=";", dtype="object")

		# get list of counties in test set (for final evaluation only)
		self.test_counties = np.loadtxt(cfg.ROOT_DIR + "test_locs.txt", delimiter=";", dtype="object")

		self.prepped_policy_data = self.prep_policy_data(min_samples=min_samples)
		# Convert date fields to datetime
		# try:
		# 	#self.data['date'] = pd.to_datetime(self.data['date'], format='%Y-%m-%d')
		# 	#self.policies['date'] = pd.to_datetime(self.policies['date'], format='%Y-%m-%d')
		# except Exception as e:
		# 	print("EXCEPTION THROWN: ", e)

		# clean data

	### Data processing
	# TODO: include some short functions to filter by county, state, and date
	def _clean_data(self):
		# Rename columns.
		print("renaming columns")
		self.data.rename(columns={
			'cumulative_cases_per_100_000': 'cumulative_cases_1e6',
			'cumulative_deaths_per_100_000': 'cumulative_deaths_1e6',
			'new_cases_per_100_000': 'new_cases_1e6',
			'new_deaths_per_100_000': 'new_deaths_1e6',
			'new_cases_7_day_rolling_avg': 'new_cases_7day',
			'new_deaths_7_day_rolling_avg': 'new_deaths_7day',
			'location_name': 'county'
		}, inplace=True)

		# Drop location_types that are not counties.
		print("dropping locations")
		self.data = self.data.drop(self.data[self.data['location_type'] != 'county'].index)

		# Get list of u.s. states
		states = [elem.name for elem in us.states.STATES]

		# Drop anything not in one of the 50 states (also drops DC).
		print("renaming states")
		self.data = self.data.drop(self.data[~self.data['state'].isin(states)].index)

		# Generate a new column with the combined state and county name.
		print("getting full loc name")
		# self.data['full_loc_name'] = self.data['location_name'] + ', ' + self.data['state']
		self.data['full_loc_name'] = self.data['county'] + ', ' + self.data['state']

		# Make all counties lowercase for consistency.
		print("loc name to lowercase")
	
		# self.data['location_name'] = self.data['location_name'].str.lower()
		self.data['county'] = self.data['county'].str.lower()

		# Re-order columns.
		print("reorder columns")
		cols = self.data.columns.tolist()
		cols = cols[:6] + [cols[-1]] + cols[6:-1]
		self.data = self.data[cols]

		# Rename location name to county
		# self.data.rename(columns={'location_name': 'county'}, inplace=True)

		# Cast dates to datetime.
		print("convert to datetime")
		self.data['date'] = pd.to_datetime(self.data['date'], format='%Y-%m-%d')

		# In numerical columns, fill na values with zero and change any negative
		# values to positive.
		print("fixing columns")
		cols = ['new_cases', 'new_deaths', 'new_cases_1e6', 'new_deaths_1e6']

		for col in cols:
			self.data[col].fillna(value=0, inplace=True)
			self.data[col].clip(lower=0, inplace=True)

		### Fix data for chugach and copper river, Alaska
		# update total population for 2 counties that have nan for total population
		# https://en.wikipedia.org/wiki/Chugach_Census_Area,_Alaska
		self.data.loc[(self.data['county'] == "chugach"), 'total_population'] = 7372

		# https://en.wikipedia.org/wiki/Copper_River_Census_Area,_Alaska
		self.data.loc[(self.data['county'] == "copper river"), 'total_population'] = 1929

		self.data.loc[(self.data['county'] == "chugach"), 'cumulative_cases_1e6'] =\
			self.data.loc[(self.data['county'] == "chugach"), 'cumulative_cases'] /\
			(self.data.loc[(self.data['county'] == "chugach"), 'total_population']/1e5)

		self.data.loc[(self.data['county'] == "copper river"), 'cumulative_cases_1e6'] =\
			self.data.loc[(self.data['county'] == "copper river"), 'cumulative_cases'] /\
			(self.data.loc[(self.data['county'] == "copper river"), 'total_population']/1e5)

		self.data.loc[(self.data['county'] == "chugach"), 'cumulative_deaths_1e6'] =\
			self.data.loc[(self.data['county'] == "chugach"), 'cumulative_deaths'] /\
			(self.data.loc[(self.data['county'] == "chugach"), 'total_population']/1e5)

		self.data.loc[(self.data['county'] == "copper river"), 'cumulative_deaths_1e6'] =\
			self.data.loc[(self.data['county'] == "copper river"), 'cumulative_deaths'] /\
			(self.data.loc[(self.data['county'] == "copper river"), 'total_population']/1e5)

		# print("having trouble with these two counties in Alaska: ")
		# print(
		# 	self.data[self.data['state'] == 'Alaska']['county'].unique()
		# )
		# print(self.data[(self.data['county'] == 'chugach, alaska')])
		# Case all numerical columns to integers.
		# print("converting all numerical columns to integers")
		# self.data['fips_code'] = self.data['fips_code'].astype(np.int64)
		# self.data['total_population'] = self.data['total_population'].astype(np.int64)
		# self.data['new_cases'] = self.data['new_cases'].astype(np.int64)
		# self.data['new_deaths'] = self.data['new_deaths'].astype(np.int64)
		cols2 = ["fips_code", "total_population", "new_cases", "new_deaths"]
		self.data[cols2] = self.data[cols2].astype(np.int64)

		# Set rolling averages to zero in the first few days of measurements.
		print("set first few rolling avgs to zero")
		self.data[['new_cases_7day', 'new_deaths_7day']] = self.data[['new_cases_7day',
														'new_deaths_7day']].mask(

			(self.data['date'] < pd.to_datetime('2020-01-30', format='%Y-%m-%d')),
			self.data[['new_cases_7day', 'new_deaths_7day']].fillna(0)
			)
		# Re-calculate 7-day rolling average to replace nulls ####################

		# Get datapoints where normalized new_cases and new_deaths are null
		nulls_case7day = self.data[self.data['new_cases_7day'].isnull()]
		nulls_death7day = self.data[self.data['new_deaths_7day'].isnull()]

		# Pre-define timedelta for efficiency.
		days_7 = timedelta(days=7)

		# Loop through nulls in new_cases_7day
		print("re-calculate 7 day rolling averages")
		i = 0
		num_elem = len(nulls_case7day)
		msg = "Looping through nulls in new_cases_7day. Please be patient, this may take a while"
		print(msg)
		for index, data in nulls_case7day.iterrows():

			# Calculate rolling average at the datapoint of interest
			self.data.loc[index, ['new_cases_7day']] = np.sum((
				[self.data['new_cases'][(self.data['full_loc_name'] == data.full_loc_name) &
								((self.data['date'] <= data.date) &
								(self.data['date'] > data.date-days_7))
								].values])) / 7
			# Output for inpatient people like me. 
			i += 1
			if i%250 == 0: 
				print(f"index: {i}/{num_elem}")
				
		# Calculate rolling average at the datapoint of interest
		# self.data.loc[index, ['new_cases_7day']] = np.sum((
		# 	[self.data['new_cases'][(self.data['full_loc_name'] == data.full_loc_name) & 
		# 	((self.data['date'] <= data.date) & (self.data['date'] > data.date-days_7))].values])) / 7

		# Loop through nulls in new_deaths_7day
		i = 0
		num_elem = len(nulls_death7day)
		print("Looping through nulls in new_deaths_7day. Please be patient, this may take a while")
		for index, data in nulls_death7day.iterrows():

			# Calculate rolling average at datapoint of interest.
			self.data.loc[index, ['new_deaths_7day']] = np.sum(([self.data['new_deaths'][
				(self.data['full_loc_name'] == data.full_loc_name) &
				((self.data['date'] <= data.date) &
				(self.data['date'] > data.date-days_7))].values])) / 7

			# Output status. 
			i += 1
			if i%25 == 0: 
				print(f"index: {i}/{num_elem}")

		# Normalize 7-day rolling averages against population.
		self.data["new_cases_7day_1e6"] = self.data['new_cases_7day'] /\
			(self.data['total_population'] / 1e5)

		self.data["new_deaths_7day_1e6"] = self.data['new_deaths_7day'] /\
			(self.data['total_population'] / 1e5)

		# Cleaning self.data2 ###########################################################
		print("cleaning policies")
		# get abbreviations for each state
		abbr = [elem.abbr for elem in us.states.STATES]

		# Exclude territories and DC
		print("excluding territories")
		self.policies = self.policies.drop(self.policies[~self.policies['state_id'].isin(abbr)].index)

		# Replace state names with abbreviations
		print("state ids -> abbr")
		self.policies.replace(to_replace=us.states.mapping('abbr', 'name'), inplace=True)

		# Rename 'state_id' column
		self.policies.rename(columns={'state_id': 'state'}, inplace=True)

		# Drop 'total phases' since it's not useful
		self.policies.drop('total_phases', axis=1, inplace=True)

		# If 'county' is null, then it's a statewide policy, so repalce it with
		# 'statewide' and convert to lowercase for consistency
		print("processing county")
		self.policies['county'].fillna(value='statewide', inplace=True)
		self.policies['county'] = self.policies['county'].str.lower()

		# Fix some formatting issues in 'date': 0020 instead of 2020, then
		# convert to datetime
		self.policies.loc[:, 'date'] = '2020' + self.policies['date'].str[4:]
		self.policies['date'] = pd.to_datetime(self.policies['date'], format='%Y-%m-%d')

		# Fix some mismatches between county names
		# (e.g. 'bronx' in df is 'bronx county' in self.policies)
		county_match = re.compile(" county$")
		munici_match = re.compile(" municipality$")
		Borough_match = re.compile(" borough$")

		self.policies['county'].replace(to_replace=county_match,  value='', inplace=True)
		self.policies['county'].replace(to_replace=munici_match,  value='', inplace=True)
		self.policies['county'].replace(to_replace=Borough_match, value='', inplace=True)

		# Update fips codes for states
		print("fips codes")
		for index, data in self.policies.iterrows():
			if data.policy_level == 'state':
				self.policies.loc[index, 'fips_code'] =\
					np.int64(us.states.lookup(data.state).fips)

		self.policies['fips_code'] = self.policies['fips_code'].astype(np.int64)

		# Drop uninformative policies
		policies_drop = ['New Phase', 'Phase 1', 'Phase 2', 'Phase 2, 3',
						'Phase 3', 'Phase 3 Step 2', 'Phase 4', 'Phase 5']

		self.policies = self.policies.drop(self.policies[self.policies['policy_type'].isin(policies_drop)].index)

		# Rename policies that have unreasonably long names
		policy_replacements_dict = {
			'Stop Initiation Of Evictions Overall Or Due To Covid Related Issues':
				'Stop Initiation Of Evictions',

			'Modify Medicaid Requirements With 1135 Waivers Date Of CMS Approval':
				'Modify Medicaid Requirements',

			'Stop Enforcement Of Evictions Overall Or Due To Covid Related Issues':
				'Stop Enforcement Of Evictions',

			'Mandate Face Mask Use By All Individuals In Public Facing Businesses':
				'Mandate Face Masks In Businesses',

			'Mandate Face Mask Use By All Individuals In Public Spaces':
				'Mandate Face Masks In Public Spaces',

			'Reopened ACA Enrollment Using a Special Enrollment Period':
				'ACA Special Enrollment Period',

			'Suspended Elective Medical Dental Procedures':
				'Suspend Elective Dental Procedures',

			'Allow Expand Medicaid Telehealth Coverage':
				'Expand Medicaid Telehealth Coverage',

			'Renter Grace Period Or Use Of Security Deposit To Pay Rent':
				'Grace Period / Security Deposit for Rent'
		}

		for key in policy_replacements_dict.keys():
			self.policies['policy_type'].replace(
				to_replace=key, value=policy_replacements_dict[key], inplace=True
				)

		self.policies['policy_type'] = self.policies['policy_type'].str.lower()

		# Drop policies implemented before the measure period or
		# planned for the future.
		self.policies = self.policies.drop(self.policies[(self.policies['date'] < min(self.data['date'])) |
						(self.policies['date'] > datetime.today())].index)

		# save cleaned data
		self.data.to_csv("{}case_data_clean.csv".format(self.path))
		self.policies.to_csv("{}policy_data_clean.csv".format(self.path))

	def get_cases(self,
					level="county",
					county="orange",
					state="California"):
		""" Return the new_case and new_death numbers at the given level of aggregation (county, state, or national). 
		Parameters
		----------
		level: {'county', 'state', 'national'}
			If county, returns a DataFrame filtered to a specific county (default). 
			If state, aggregates the DataFrame to the state level. 
			If national, or any other input, returns the DataFrame aggregated to the national level. 
		county: string 
			desired county
		state: string 
			desired state
		
		Returns 
		----------
		DataFrame
			case_data filtered to a specific county or aggregated to the state / national level with index=date
		"""
			    # return county level data
		if level == "county":
			return_df = self.data[(self.data["county"] == county) & (self.data["state"] == state)].set_index("date")[['new_cases_1e6', 
																								'new_deaths_1e6', 
																								'new_cases_7day_1e6', 
																								'new_deaths_7day_1e6',
																								]]
			
			return return_df
		
		# If this is filtered at the state level, filter df to desired state. Otherwise, return national-level data.
		if level == "state": 
			self.data = self.data[self.data['state'] == state]

		# Reindex on location name.
		self.data = self.data.set_index(["full_loc_name"])
		
		# Get a list of all dates.
		all_dates = self.data['date'].unique()

		# Get the total population from the county populations.
		total_population =  sum([(pops / 1e5) for pops in self.data[self.data['date'] == all_dates[0]]['total_population']]) 
		
		# Add up the case and death #s that have the same date.
		new_cases = [sum([(county_cases / total_population) 
								for county_cases in self.data[self.data['date'] == dates]['new_cases_1e6']]) 
								for dates in all_dates]

		new_deaths = [sum([(county_cases / total_population) 
								for county_cases in self.data[self.data['date'] == dates]['new_deaths_1e6']]) 
								for dates in all_dates]

		new_cases_7day = [sum([(county_cases / total_population) 
								for county_cases in self.data[self.data['date'] == dates]['new_cases_7day_1e6']]) 
								for dates in all_dates]

		new_deaths_7day = [sum([(county_cases / total_population) 
								for county_cases in self.data[self.data['date'] == dates]['new_deaths_7day_1e6']]) 
								for dates in all_dates]


		return_df = pd.DataFrame(data={'date'               : all_dates,
									'new_cases_1e6'      : new_cases, 
									'new_deaths_1e6'     : new_deaths,
									'new_cases_7day_1e6' : new_cases_7day,
									'new_deaths_7day_1e6': new_deaths_7day
									}).set_index(["date"]) 
		return return_df
			
	def get_policy_data(self, state="California",
						county="statewide",
						state_policies=True,
						county_policies=True):

		"""Get the policy data at county level, state level, or both.  

		Parameters
		----------
		state : string 
			selected state
		county : string
			selected county
		state_policies : boolean 
			include policies at the state level (default: True)
		county_policies : boolean 
			include policies at the county level (default: True)
		
		Returns
		---------- 
		filtered DataFrame
		"""
		    # state AND county policies
		if state_policies and county_policies:

			return self.policies[(self.policies['state'] == state) &
					((self.policies["county"] == county) | (self.policies["county"] == "statewide"))]
		
		# state policies only
		elif state_policies and not county_policies: 
			return self.policies[(self.policies['state'] == state) & (self.policies["county"] == "statewide")]
		
		# county policies only
		else:  
			return self.policies[(self.policies['state'] == state) & (self.policies["county"] == county)]

	### VISUALIZATION FUNCTIONS
	def plot_cases(self, level="county", 
					county="orange", 
					state="California", 
					fade=0.75, 
					style="whitegrid", 
					ax=None, 
					fig=None, 
					save_figure=False, 
					fig_size=(10, 5), 
					filename="Plot cases figure.png"):

		""" A function which plots the COVID-19 case/death data and 7 day average.

		Parameters
		---------- 
		level: {'county', 'state', 'national'}
			Value to pass to get_cases() 
			Default: "county"
		county: string 
			desired county
			Default: "orange"
		state: string 
			desired state
			Default: "California"
		df: pandas DataFrame 
			DataFrame to use 
			Default: df
		fade: float
			level of transparency for new_cases_1e6 and new_deaths_1e6 
			Default: 0.75
		style: string
			Seaborn plot style 
			Default: "whitegrid"
		ax: matplotlib axis object
			Add to an existing axis
		fig: matplotlib figure object
		save_figure: boolean
			Default: False
		figure_width: float
		filename: string
			Name of file if saving figure
			
		Returns 
		----------
		matplotlib.figure.Figure
		ndarray containing the two axis handles
		pandas DataFrame
		"""

		# Get the data. 
		cases = self.get_cases(level=level,
								county=county,
								state=state)

		# Set up plots. 
		if ax is None: 
			fig, ax = plt.subplots(2, 1, figsize=fig_size, sharex=True)
		plt.subplots_adjust(hspace=0.02)
		sns.set_style(style)
		
		# Plot cases.
		cases.plot(
			color=[f'{fade}', 'k'],
			y=["new_cases_1e6","new_cases_7day_1e6"],
			label=["Cases per capita", "7-day average"],
			ax=ax[0]
			)

		# Plot deaths. 
		cases.plot(
			color=[f'{fade}', 'k'],
			y=["new_deaths_1e6","new_deaths_7day_1e6"],
			label=["Deaths per capita", "7-day average"],
			ax=ax[1])
		
		# Format axis labels. 
		ax[0].set_ylabel("cases per 100,000", fontsize=cfg.MED_FONT)
		ax[1].set_ylabel("deaths per 100,000", fontsize=cfg.MED_FONT)
		
		# Set plot title based on level of aggregeation (county, state, or national)
		if level == "county":
			ax[0].set_title(f"New COVID-19 cases and deaths per 100,000 in {county} County, {state}", fontsize=cfg.BIG_FONT-4)
		elif level == "state": 
			ax[0].set_title(f"New COVID-19 cases and deaths per 100,000 in {state}", fontsize=cfg.BIG_FONT-4)
		else: 
			ax[0].set_title("New COVID-19 cases and deaths per 100,000 in the United States", fontsize=cfg.BIG_FONT-4)
		
		if save_figure: 
			plt.savefig(filename)
		return fig, ax, cases


	# TODO: 
	def plot_cases_and_policies(self,
		county="orange", state="California", 
		colors=sns.color_palette()[:],
		policies=[
			"mandate face masks in public spaces", 
			"mandate face masks in businesses", 
			"shelter in place", 
			"state of emergency"],
							
		labels=["face mask mandate (public spaces)", 
			"face mask mandate (businesses)", 
			"shelter in place", 
			"state of emergency"],
		save_figure=False, 
		filename="Plot cases and policies figure.png", 
		fig_size=(10, 5), 
		legend_position=(0, 0)
		):


		"""Plot the cases overlayed with the policies. 
		
		Parameters
		---------- 
		county : string 
			county of interest
		state : string
			state of interest
		policies : array-like
			policies to plot (default: face mask mandates in public spaces and businesses, shelter in place, 
			and state of emergency)
		colors : array-like
			line colors for respective policies (in order) (default: k, b, r, g)
		labels : array-like 
			legend labels for the selected policies (default: "face mask mandate (public spaces)", 
			"face mask mandate (businesses)", "shelter in place", and "state of emergency")
		style : string 
			sns plot style (whitegrid by default, dark styles not recommended)
		fade : float
			level of transparency for new_cases_1e6 and new_deaths_1e6 (default: 0.75)
		
		The marks for policies are aligned with the 7 day average, using colors to indicate policy types, endcaps for 
		state (diamond) or county (circle), and linestyle to distinguish the start (solid line) or stop (dotted line) of a 
		policy. 
		
		Returns
		----------
		ndarray containing the two axis handles used for plotting
		"""

		# Plot case / death data.
		plt.figure()
		fig, ax, df = self.plot_cases(level="county", county=county, state=state, fig_size=fig_size)

		# Get the policy data for the selected state and county.
		policy_data = self.get_policy_data(state, county)

		# Set a solid line to be the start of a policy and a dashed line to be the end.
		styles_start_stop = {
			"start" : '-', 
			"stop" : ':'
		}

		# Set color codes for selected policies.
		styles_policy_type = {policies[i] : colors[i] for i in range(len(policies))}

		# Set labels for legend.
		legend_policy_labels = {policies[i] : labels[i] for i in range(len(policies))}

		# Define plot parameters.  
		line_split = 0                
		mark_length = 0.2
		plot_policies = policy_data[policy_data['policy_type'].isin(policies)]
		labels = []

		# Loop through both axes. 
		for i in range(2):
			
			# Expand y axis to get some extra room on the bottom.
			ax[i].set_ylim(-max(ax[i].lines[0].get_ydata())*(0.15))
			
			# Loop through policies.
			for index, row in plot_policies.sort_values(by="date").iterrows():

				# Get the y-positional coordinate for the line on the selected day (between 0 and 1). 
				if i == 0: 
					center = df[df.index == row.date]['new_cases_7day_1e6'].values[0]
				else: 
					center = df[df.index == row.date]['new_deaths_7day_1e6'].values[0]
				
				# Calculate where to position the line horizontally.  
				days_serial = (datetime.strptime(row.date, '%Y-%m-%d') - pd.Timestamp(year=1970, month=1, day=1)).days
				cent_coord = ax[i].transLimits.transform((days_serial, center))[1]

				# loop through all the policies enacted on a given day. Normally, this is 1, but we want to visualize all the
				# policies enacted on the same day, so we're goint to split the line accordingly.

				num_policies = plot_policies['date'].value_counts()[row.date]
				
				# Split the mark if there are multiple policies enacted on the same day.
				if num_policies > 1: 
					hmin = (cent_coord - (mark_length/2)) + line_split * (mark_length/num_policies)
					hmax = (cent_coord + (mark_length/2)) - \
						(mark_length/num_policies) + ((line_split*mark_length) / num_policies)
					line_split += 1

				if num_policies == 1:
					hmin = (cent_coord - (mark_length/2))
					hmax = (cent_coord + (mark_length/2))
					line_split=0
					
				# Plot the mark.
				line = ax[i].axvline(x = row.date,
										linestyle = styles_start_stop[row.start_stop], 
										color     = styles_policy_type[row.policy_type], 
										lw        = 2,
										ymin      = hmin, 
										ymax      = hmax,
									)

				# Add the policy to the legend if it's not already in the list of labels
				if (row.policy_type not in labels): 
					labels.append(row.policy_type)
					
				# Set markers for state or county policies. 
				if row.policy_level == "state":
					line.set_marker('d')
					
				else: 
					line.set_marker('o')
					
				ax[i].legend(loc="upper left")
			
		# Make the legend. 
		legend_lines = []

		# Use square boxes to denote colors for policy types. 
		[legend_lines.append(Line2D([0], [0], marker="s", markersize=15, color='w', 
								markerfacecolor=styles_policy_type[policy], lw=3, label=policy)) for policy in labels]

		# Draw a diamond to indicate a state policy. 
		legend_lines.append(Line2D([0], [0], marker="d", markersize=10, color='w', 
								markerfacecolor='w', markeredgecolor='k', markeredgewidth=1.5, lw=3, label="state policy"))

		# Draw a circle to indicate a county policy. 
		legend_lines.append(Line2D([0], [0], marker="o", markersize=10, color='w', 
								markerfacecolor='w', markeredgecolor='k', markeredgewidth=1.5, lw=3, label="county policy"))

		# Use a solid line for policy start, dotted line for policy stop. 
		legend_lines.append(Line2D([0], [0], linestyle='-', color='k', lw=3, label = "policy start"))
		legend_lines.append(Line2D([0], [0], linestyle=':', color='k', lw=3, label = "policy stop"))
						
			
		# Finally, draw the legend. 
		leg1 = ax[0].legend(loc='upper left')
		leg2 = ax[0].legend(handles=legend_lines, loc='center',  bbox_to_anchor=legend_position, ncol=4)
		ax[0].add_artist(leg1)

		if save_figure: 
			plt.savefig(filename, bbox_inches='tight')
			
		return fig, ax

	### PREPROCESSING FOR ML MODELS
	def prep_policy_data(self,
						#policy_dict=policy_dict,
						min_samples=10,):
		"""Preprocess the policy data for Machine Learning models.

		Parameters
		------------
		df2: DataFrame
			policy data
		policy_dict: dictionary
			Dictionary defined in policy_dict.py to rename and aggregate policies.
		min_samples: integer
			Throw out policies that were not implemented min_samples times.

		Returns
		----------
		proc_policy_data: DataFrame
			The preprocessed policy data
		"""

		proc_policy_data = self.policies.copy()

		# Replace policies with the ones in policy_dict().
		for key in policy_dict.keys():
			proc_policy_data['policy_type'].replace(
				to_replace=key, value=policy_dict[key], inplace=True)

		# Define a new field that includes policy_type, start_stop,
		# and policy_level information
		proc_policy_data.loc[:, 'full_policy'] =\
			proc_policy_data['policy_type'] + " - " +\
			proc_policy_data['start_stop'] + " - " +\
			proc_policy_data['policy_level']

		# Get number of times each policy was implemented.
		num_samples = proc_policy_data['full_policy'].value_counts()

		# drop the policy if it was implemented fewer than min_policy times.
		proc_policy_data = proc_policy_data.drop(proc_policy_data[
			proc_policy_data['full_policy'].isin(
				num_samples[num_samples.values < min_samples].index)
		].index)

		# return the DataFrame
		return proc_policy_data

	def join_policies(self, case_df,
					policy_df,
					output=True,
					bins_list=[(0, 6), (7, 13), (14, 999)],
					state_output=False,
					save_csv=True,
					load_csv=True,
					cutoff_date=pd.to_datetime("2020-12-31"),
					filter_counties=None):
		"""Data preprocessing for Machine Learning that joins case and policy data.

		This function works by generating a processed DataFrame for each state and
		the concatenating the resulting 50 DataFrames at the very end.

		Parameters
		------------
		case_df: DataFrame
			case data
		policy_df: DataFrame
			policy data
		output: boolean
			output time elapsed after running
		bins_list: list
			list of tuples with the desired date ranges
			default: [(0, 6), (7, 13), (14, 999)]
		state_output: boolean
			output time elapsed for filtering with each state
			default: False

		Returns
		------------
		df3: DataFrame
			Processed DataFrame
		"""
		time_start = time.time()
		filename = cfg.DATASET_DIR + f"prepped_data_{str(bins_list)}.csv"
		if load_csv and os.path.exists(filename):
			df3 = pd.read_csv(filename, header=[0, 1], index_col=[0])
			print("loaded data from csv")
			return df3
		# Define a sub-function to convert passed integers to the desired
		# date range starting from a given date.
		def _get_date_range(date, start_move=0, stop_move=7):

			# Get the date range from date+start_move to date+stop_move
			return pd.date_range(start=date+timedelta(days=start_move),
								end=date+timedelta(days=stop_move))

		# Ensure all date columns are datetime
		case_df.loc[:, 'date'] = pd.to_datetime(case_df.loc[:, 'date'], format='%Y-%m-%d')
		policy_df.loc[:, 'date'] = pd.to_datetime(policy_df.loc[:, 'date'], format='%Y-%m-%d')

		# filter dates after the cutoff
		case_df = case_df[case_df['date'] < cutoff_date]
		policy_df = policy_df[policy_df['date'] < cutoff_date]

		# filter to specific counties
		if filter_counties is not None:
			case_df = case_df[case_df['full_loc_name'].isin(filter_counties)]

		# Make list of all policies.
		all_policies = policy_df['full_policy'].unique()

		# Construct multiIndex for df3.
		tuples_info = [("info", "state"), ("info", "county"), ("info", "full_loc"),
					("info", "date"), ("info", "new_cases_1e6")]

		tuples_policies = [(policy,
							(str(date_range[0]) + "-" + str(date_range[1])))
						for policy in all_policies for date_range in bins_list]

		tuples_index = tuples_info + tuples_policies
		col_index = pd.MultiIndex.from_tuples(tuples_index)

		# Get list of all states.
		all_states = case_df['state'].unique()

		# Generate a list of empty dataframes- one for each state.
		frames = [pd.DataFrame() for state in all_states]

		# Loop through all states.
		for (i, state) in enumerate(all_states):

			if output:
				state_time_start = time.time()

			# Initiallize dataFrame.
			frames[i] = pd.DataFrame(columns=pd.MultiIndex.from_tuples(col_index),
									data={
				("info", "state"): state,
				("info", "county"): case_df['county'][case_df['state'] == state],
				("info", "full_loc"): case_df['full_loc_name'][
						case_df['state'] == state
					],
				("info", "date"): case_df['date'][case_df['state'] == state],
				("info", "new_cases_1e6"): case_df['new_cases_1e6'][
					case_df['state'] == state]
			})

			# Filter policy data to only those that were enacted in that state.
			filtered_policy = policy_df[policy_df['state'] == state]

			# Loop through every policy that was implemented in the current state.
			for (date, county, policy, level) in zip(filtered_policy['date'],
													filtered_policy['county'],
													filtered_policy[
														'full_policy'],
													filtered_policy[
														'policy_level']):
				# Loop through each bin
				for date_bin in bins_list:

					# calculate the date range
					date_range = _get_date_range(date, date_bin[0], date_bin[1])

					# Generate label (this is the 2nd level label in the
					# multiIndexed column)
					label = (str(date_bin[0]) + "-" + str(date_bin[1]))

					# For every record in frames that falls within the date range
					# of the specific policy, set the appropriate column to 1.
					frames[i].loc[(
						(frames[i]['info', 'date'].isin(date_range)) &
						((frames[i]['info', 'county'] == county) |
						(level == 'state')) &
						(frames[i]['info', 'state'] == state)),
						(policy, label)] = 1

			if state_output:
				state_time_end = time.time()
				print(f"{state}: {state_time_end - state_time_start} seconds")

		# Concat the DataFrames
		df3 = pd.concat([frames[i] for i in range(len(frames))])

		# Fill NaNs
		df3.fillna(0, inplace=True)
		time_end = time.time()

		if save_csv:
			df3.to_csv(filename)
			print("csv saved")
		if output:
			print("data shaped\nbins: {}\ntime elapsed: {}".format(
				bins_list, (time_end - time_start)))

		return df3

	def county_split(self, df, test_size):
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

	# Run models

	def train_model(self,
					df_train_proc,
					model_in,
					metrics_dict,
					K=10,
					verbose=True,
					save_output=False,
					filename="log.txt"):

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

		# get a list of all unique counties
		counties = df_train_proc[('info', 'full_loc')].unique()

		# shuffle the counties
		np.random.shuffle(counties)
		
		# declare batch size
		batch_size = int(len(counties) / K)
		
		# logging
		msg1 = f"number of cross-validation folds: {K}"
		msg2 = f"num counties in validation set: {batch_size}"
		
		if verbose: 
			print(msg1)
			print(msg2)
		if save_output: 
			with open(filename, "a") as log: 
				log.write(msg1)
				log.write(msg2)

		# loop through cross validation folds
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
	
	def loop_models(self,
					df_train_proc,
					models_dict,
					metrics_dict,
					bin_id,
					K=10,
					verbose=True,
					save_output=False,
					filename="log.txt",):
	
		# Declare empty dict to hold all results
		results = {}
		
		# loop through all models passed
		for model in models_dict.keys():
			msg = f"running model: {model}"
			
			if verbose:
				print(msg)
			if save_output:
				with open(filename, "a") as log:
					log.write(msg)
				
			# declare emtpy dictionary for results for a single run
			model_results = {}
			scores, params = self.train_model(df_train_proc=df_train_proc,
											model_in=models_dict[model],
											metrics_dict=metrics_dict,
											K=K,
											verbose=verbose,
											save_output=save_output,
											filename=filename,)

			# save the results in a dictionary
			model_results['params'] = params
			model_results['scores'] = scores
			
			results[model] = model_results

		return results

	def run_batch(self,
				bins_dict,
				models_dict,
				metrics_dict,
				case_data=None,
				policies=None,
				K=10,
				verbose=True,
				save_output=False,
				filename=None,
				overwrite=True,
				output_json=False,
				json_file=None,):
		results = {}
		
		if case_data is None:
			case_data = self.data
		if policies is None:
			policies = self.prepped_policy_data
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
			try:
				df_train_proc = self.join_policies(case_df=case_data,
											policy_df=policies,
											output=True,
											bins_list=bins_list,
											state_output=False,
											save_csv=True,
											load_csv=True,
											cutoff_date=pd.to_datetime("2020-12-31"),
											filter_counties=self.train_counties,
											)
				
				models_results = self.loop_models(df_train_proc=df_train_proc,
											models_dict=models_dict,
											metrics_dict=metrics_dict,
											K=K,
											bin_id=str(bins_list),
											verbose=verbose,
											save_output=save_output,
											filename=filename)
			except Exception as e:
				print("EXCEPTION THROWN: ", e)
				continue
			models_results['bins'] = bins_list
			results[key] = models_results

			if output_json:
				if overwrite & os.path.exists(json_file):
					os.remove(json_file)
					
				with open(json_file, 'w') as file:
					print("saving json")
					json.dump(results, file, sort_keys=True, indent=4)
				
		return results
