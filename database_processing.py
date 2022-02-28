# Code for
# Physics solutions for machine learning privacy leaks
# arXiv:2202.12319
#
# Authors: Alejandro Pozas-Kerstjens and Senaida Hernandez-Santana
#
# Requires: os for filesystem operations
#			pandas for CSV operations
# Last modified: Feb, 2022

################################################################################
# This file processes the global.health database in order to generate the large
# dataset that will be partitioned and trained on in the experiments.
################################################################################
import os
import pandas as pd

# Load data
data_path = 'globaldothealth_2021-03-22.csv'
data  	  = pd.read_csv(data_path,
				        usecols=['location.country',
								 'events.outcome.value',
								 'events.confirmed.date',
								 'demographics.ageRange.start',
								 'demographics.gender',
								 'symptoms.status'])

# Simplify column names
data.rename(columns={'events.confirmed.date': 'date',
				     'location.country': 'country',
				     'demographics.ageRange.start': 'age',
				     'demographics.gender': 'gender',
				     'symptoms.status': 'symptomatic',
				     'events.outcome.value': 'outcome'},
			inplace=True)

# Select countries of interest
data_Argentina = data[data['country'] == 'Argentina']
data_Colombia  = data[data['country'] == 'Colombia']

data = pd.concat([data_Argentina, data_Colombia])

# Clean the outcome column. All admissions to a hospital (either in a standard
# bed or in ICU) are treated equally
data['outcome'].replace({'icuAdmission': 'hospitalAdmission',
						 'Unknown': float('nan'),
						 '\)': ''},
						regex=True,
						inplace=True)
data.dropna(inplace=True)

# Balance points according to country
points_col, points_arg = data['country'].value_counts()
step 		       	   = points_col // points_arg

colombia_mask     = (data['country'] == 'Colombia')
nonrecovered_mask = (data['outcome'] == 'Death')
data_cn           = data[colombia_mask    & nonrecovered_mask][::step]
data_cr 	  	  = data[colombia_mask    & (~nonrecovered_mask)][::step]
data_an 	  	  = data[(~colombia_mask) & nonrecovered_mask]
data_ar 	  	  = data[(~colombia_mask) & (~nonrecovered_mask)]
data 		  	  = pd.concat([data_cn, data_cr, data_an, data_ar])

# Use boolean variable for symptoms
data['symptomatic'] = (data['symptomatic'] == 'Symptomatic')

# Sort according to date
data.sort_values(by='date', inplace=True)
data.reset_index(inplace=True)

# Extract the day of confirmation and store its parity
days        = data['date'].apply(lambda x: x.split('-')[-1]).astype(int)
odd_day     = (days%2 != 0)
data['odd'] = odd_day

# Choose columns for the final database
columns = ['odd', 'country', 'age', 'gender', 'symptomatic', 'outcome']
data    = data[columns]

################################################################################
# Generate a dataset with the same number of recovery and non-recovery cases,
# and even and odd registration day
################################################################################
recovered    = data[data.outcome == 'Recovered']
nonrecovered = data[data.outcome == 'Death']

# There are more recovered than not, so the latter gives us the dataset size
count		 	 = len(nonrecovered) // 4
recovered_arg    = recovered[recovered.country == 'Argentina']
recovered_col    = recovered[recovered.country == 'Colombia']
balanced_dataset = pd.concat([nonrecovered,
						      recovered_arg[recovered_arg.odd].iloc[:count],
						      recovered_col[recovered_col.odd].iloc[:count],
						      recovered_arg[~recovered_arg.odd].iloc[:count],
						      recovered_col[~recovered_col.odd].iloc[:count]])
balanced_dataset.sort_index(inplace=True)

################################################################################
# Final simplifications
################################################################################
# Replace strings for booleans
balanced_dataset.replace({'country': {'Argentina': True, 'Colombia': False},
				          'gender':  {'Female': True, 'Male': False},
					  	  'outcome': {'Death': False, 'Recovered': True}},
		         		 inplace=True)

# More descriptive column names
balanced_dataset.rename(columns={'country': 'argentina',
								 'gender':  'female',
								 'outcome': 'recovered'},
						inplace=True)

# Final export
if not os.path.isdir('datasets'):
    os.mkdir('datasets')
balanced_dataset.to_csv('datasets/covid_argentina_colombia_until20210322.csv')
