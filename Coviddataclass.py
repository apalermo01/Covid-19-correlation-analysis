import pandas as pd
import numpy as np
import re
import us_state_abbrev as abrev

class Coviddataclass:

    def clean_data(self, df, df2):
        us_state_abbrev = abrev.us_state_abbrev
        abbrev_us_state = abrev.abbrev_us_state

        # pop territories from us_state_abbrev
        us_state_abbrev.pop('American Samoa')
        us_state_abbrev.pop('Guam')
        us_state_abbrev.pop('Northern Mariana Islands')
        us_state_abbrev.pop('Virgin Islands')
        us_state_abbrev.pop('Puerto Rico')

        abbrev_us_state.pop('AS')
        abbrev_us_state.pop('GU')
        abbrev_us_state.pop('MP')
        abbrev_us_state.pop('VI')
        abbrev_us_state.pop('PR')

        ############################################
        ### Cleaning case and death dataset (df) ###
        ############################################

        # rename columns
        df.rename(columns={
        'cumulative_cases_per_100_000': 'cumulative_cases_1e6',
        'cumulative_deaths_per_100_000': 'cumulative_deaths_1e6',
        'new_cases_per_100_000': 'new_cases_1e6',
        'new_deaths_per_100_000': 'new_deaths_1e6',
        'new_cases_7_day_rolling_avg': 'new_cases_7day',
        'new_deaths_7_day_rolling_avg': 'new_deaths_7day'}, inplace=True)

        # filter location types
        df = df.drop(df[~df['location_type'].isin(['county'])].index)

        # convert fips_code to ints
        df['fips_code'] = df['fips_code'].astype(np.int64)

        # concatenate full location name
        df['full_loc_name'] = df['location_name'] + ', ' + df['state']

        # change all location names to lowercase
        df['location_name'] = df['location_name'].str.lower()

        # drop Puerto Rico
        df.drop(df[df['state']=='Puerto Rico'].index, inplace=True)

        # convert dates to datetime
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

        # convert total_population to integers
        df['total_population'] = df['total_population'].astype(np.int64)

        # fill nulls in new_cases and new_deaths
        df['new_cases'].fillna(value=0, inplace=True)
        df['new_deaths'].fillna(value=0, inplace=True)

        # drop negative values in new_cases and new_deaths
        df['new_cases'].clip(lower=0, inplace=True)
        df['new_deaths'].clip(lower=0, inplace=True)

        # convert new_cases and new_deaths to ints
        df['new_cases'] = df['new_cases'].astype(np.int64)
        df['new_deaths'] = df['new_deaths'].astype(np.int64)

        # fill nulls and negatives in new_cases_1e6 and new_deaths_1e6
        df['new_cases_1e6'].clip(lower=0, inplace=True)
        df['new_cases_1e6'].fillna(value=0, inplace=True)
        df['new_deaths_1e6'].clip(lower=0, inplace=True)
        df['new_deaths_1e6'].fillna(value=0, inplace=True)

        # convert nulls in new_cases_7day and new_deaths_7day to in the first week of data collection to 0
        df['new_cases_7day'][df['date'] < pd.to_datetime('2020-01-30', format='%Y-%m-%d')] =\
            df['new_cases_7day'][df['date'] < pd.to_datetime('2020-01-30', format='%Y-%m-%d')].fillna(value=0)

        df['new_deaths_7day'][df['date'] < pd.to_datetime('2020-01-30', format='%Y-%m-%d')] =\
            df['new_deaths_7day'][df['date'] < pd.to_datetime('2020-01-30', format='%Y-%m-%d')].fillna(value=0)

        # convert the rest of the nulls to zero (figure out why there are so many nulls in the field in a future update)
        df['new_cases_7day'].fillna(value=0, inplace=True)
        df['new_deaths_7day'].fillna(value=0, inplace=True)

        # generate normalized 7 day rolling averages
        df["new_cases_7day_1e6"]  = df['new_cases_7day']   /(df['total_population']/1e5)
        df["new_deaths_7day_1e6"] = df['new_deaths_7day'] /(df['total_population']/1e5)

        ###############################
        ### Cleaning policy dataset ###
        ###############################

        # drop territories
        df2 = df2.drop(df2[~df2['state_id'].isin(abbrev_us_state)].index)

        # rename state_id
        df2.replace(to_replace=abbrev_us_state, inplace=True)

        # replace nulls in county with 'statewide'
        df2['county'].fillna(value='statewide', inplace=True)

        # convert county to lowercase
        df2['county'] = df2['county'].str.lower()

        # change names of some counties to match in df
        county_match    = re.compile(" county$")
        munici_match    = re.compile(" municipality$")
        city_match      = re.compile(" city$")
        Borough_match   = re.compile(" borough$")

        df2['county'].replace(to_replace= county_match, value='', inplace=True)
        df2['county'].replace(to_replace= munici_match, value='', inplace=True)
        df2['county'].replace(to_replace=   city_match, value='', inplace=True)
        df2['county'].replace(to_replace=Borough_match, value='', inplace=True)

        # convert dates to datetime
        df2['date'] = pd.to_datetime(df2['date'], format='%Y-%m-%d')

        # drop policies before start date
        start_date = min(df['date'])
        df2.drop(df2[df2['date'] < start_date].index, inplace=True)

        # rename some policy types
        policy_replacements_dict = {
        'Stop Initiation Of Evictions Overall Or Due To Covid Related Issues' : 'Stop Initiation Of Evictions',
        'Modify Medicaid Requirements With 1135 Waivers Date Of CMS Approval' : 'Modify Medicaid Requirements',
        'Stop Enforcement Of Evictions Overall Or Due To Covid Related Issues': 'Stop Enforcement Of Evictions',
        'Mandate Face Mask Use By All Individuals In Public Facing Businesses': 'Mandate Face Masks In Businesses',
        'Mandate Face Mask Use By All Individuals In Public Spaces'           : 'Mandate Face Masks In Public Spaces',
        'Reopened ACA Enrollment Using a Special Enrollment Period'           : 'ACA Special Enrollment Period',
        'Suspended Elective Medical Dental Procedures'                        : 'Suspend Elective Dental Procedures',
        'Allow Expand Medicaid Telehealth Coverage'                           : 'Expand Medicaid Telehealth Coverage',
        'Renter Grace Period Or Use Of Security Deposit To Pay Rent'          : 'Grace Period / Security Deposit for Rent'}

        for key in policy_replacements_dict.keys():
            df2['policy_type'].replace(to_replace=key, value=policy_replacements_dict[key], inplace=True)

        # drop non-specific policies
        policies_drop = ["phase 1", "phase 2", "phase 3", "phase 4", "phase 5", "new phase"]
        df2 = df2.drop(df2[df2['policy_type'].isin(policies_drop)].index)

        # fill nulls in total_phases with 0
        df2['total_phases'].fillna(value=0, inplace=True)

        # convert total_phases to integers
        df2['total_phases'] = df2['total_phases'].astype(np.int64)

        return df, df2


    def get_cases(self, df, level="county", county="orange", state="California"):

        """ A function which filters case_data to a specific county.
        inputs:
        level  -- "county" [default] - get county level data
                any other input - get state level data
        county -- desired county
        state  -- desired state
        df     -- DataFrame to use, case_data by default

        return:
        dataFrame -- case_data filtered to a specific county with index=date
        """

        if level == "county":
            return  df[(df["location_name"] == county) & (df["state"] == state)].set_index("date")
        else:
            # filter data to desired state
            df = df[df['state'] == state]

            # reindex on location name
            df = df.set_index(["location_name"])

            # get a list of all dates
            all_dates = pd.to_datetime(np.unique(df['date'].to_numpy()))

            # get the state population from county populations
            pop = sum([(pops/1e5) for pops in df[df['date']==all_dates[0]]['total_population']])

            # add up the case and death #s that have the same date and state (adding cover counties)
            state_cases       = [sum([(county_cases/pop) for county_cases in df[df['date'] == dates]['new_cases_1e6' ]])
                            for dates in all_dates]

            state_deaths      = [sum([(county_cases/pop) for county_cases in df[df['date'] == dates]['new_deaths_1e6']])
                            for dates in all_dates]

            state_cases_7day  = [sum([(county_cases/pop) for county_cases in df[df['date'] == dates]['new_cases_7day'   ]])
                            for dates in all_dates]

            state_deaths_7day = [sum([(county_cases/pop) for county_cases in df[df['date'] == dates]['new_deaths_7day'  ]])
                            for dates in all_dates]

            return pd.DataFrame(data={'date'    : all_dates,
                                'new_cases_1e6'  : state_cases,
                                'new_deaths_1e6' : state_deaths,
                                'new_cases_7day_1e6'    : state_cases_7day,
                                'new_deaths_7day_1e6'   : state_deaths_7day
                                }).set_index(["date"])