import pandas as pd
import numpy as np
import re
import us
from IPython.display import clear_output
from datetime import datetime, timedelta
import time

class Coviddataclass:

    def clean_data(self, df=None, df2=None):
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

        # filter location
        df = df.drop(df[~df['location_type'].isin(['county'])].index)
        states = [elem.name for elem in us.states.STATES]
        df = df.drop(df[~df['state'].isin(states)].index)

        # convert fips_code to ints
        df['fips_code'] = df['fips_code'].astype(np.int64)

        # concatenate full location name
        df['full_loc_name'] = df['location_name'] + ', ' + df['state']
        df['location_name'] = df['location_name'].str.lower()

        # re-order columns
        cols = df.columns.tolist()
        cols = cols[:6] + [cols[-1]] + cols[6:-1]
        df = df[cols]

        df.rename(columns={'location_name' : 'county'}, inplace=True)

        # convert dates to datetime
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

        # convert total_population to integers
        df['total_population'] = df['total_population'].astype(np.int64)

        # In new_cases (_1e6) & new_deaths (_1e6), fill nulls with zero and replace negatives with zero as well
        cols = ['new_cases', 'new_deaths', 'new_cases_1e6', 'new_deaths_1e6']

        for col in cols:
            df[col].fillna(value=0, inplace=True)
            df[col].clip(lower=0, inplace=True)

        cols2 = ["fips_code", "total_population", "new_cases", "new_deaths"]
        df[cols2] = df[cols2].astype(np.int64)

        # Re-calculate 7 day rolling averages.
        # Start timer.
        start = time.time()

        # Set rolling averages to zero in the first few days of measurements.
        df[['new_cases_7day', 'new_deaths_7day']] = df[['new_cases_7day', 'new_deaths_7day']].mask(
            (df['date'] < pd.to_datetime('2020-01-30', format='%Y-%m-%d')), df[['new_cases_7day', 'new_deaths_7day']].fillna(0)
            )

        # Get dataframes where normalized new_cases and new_deaths are null
        nulls_case7day = df[df['new_cases_7day' ].isnull()]
        nulls_death7day = df[df['new_deaths_7day'].isnull()]

        # Pre-define timedelta for efficiency.
        days_7 = timedelta(days=7)

        # Prepare some outputs since this function will take a while.
        i = 0
        num_elem = len(nulls_case7day)
        print("Re-calculating 7-day rolling averages")
        msg = "Looping through nulls in new_cases_7day. Please be patient, this may take a while"
        print(msg)

        # Loop through nulls in new_cases_7day
        for index, data in nulls_case7day.iterrows():

            # Output for inpatient people like me.
            i += 1
            if i%10 == 0:
                print(f"index: {i}/{num_elem}")

            if i%250 == 0:
                clear_output()
                print(msg)
                print(f"index: {i}/{num_elem}")

            # Calculate rolling average at the datapoint of interest
            df.loc[index, ['new_cases_7day']] = np.sum((
                [df['new_cases'][(df['full_loc_name'] == data.full_loc_name) &
                ((df['date'] <= data.date) & (df['date'] > data.date-days_7))].values])) / 7

        # Reset output.
        clear_output()
        print("Looping through nulls in new_deaths_7day. Please be patient, this may take a while")
        i = 0
        num_elem = len(nulls_death7day)

        # Loop through nulls in new_deaths_7day
        for index, data in nulls_death7day.iterrows():

            # Output status.
            i += 1
            if i%10 == 0:
                print(f"index: {i}/{num_elem}")

            # Calculate rolling average at datapoint of interest.
            df.loc[index, ['new_deaths_7day']] = np.sum(([df['new_deaths'][
                (df['full_loc_name'] == data.full_loc_name) &
                ((df['date'] <= data.date) & (df['date'] > data.date-days_7))].values])) / 7


        clear_output()
        print("Done re-calculating 7-day averages")
        end = time.time()
        print(f"time elapsed: {end-start}")

        # Generate normalized 7 day rolling averages.
        df["new_cases_7day_1e6"]  = df['new_cases_7day']   /(df['total_population']/1e5)
        df["new_deaths_7day_1e6"] = df['new_deaths_7day'] /(df['total_population']/1e5)

        ###############################
        ### Cleaning policy dataset ###
        ###############################

        # Drop territories, convert state abbreviations to full name, re-name 'state_id' to 'state',
        # drop 'total_phases', fill nulls in 'county' with 'statewide', convert 'county' to lowercase,
        # and convert 'date' to datetime.

        abbr = [elem.abbr for elem in us.states.STATES]

        df2 = df2.drop(df2[~df2['state_id'].isin(abbr)].index)                     # 1
        df2.replace(to_replace=us.states.mapping('abbr', 'name'), inplace=True)    # 2
        df2.rename(columns={'state_id' : 'state'}, inplace=True)                   # 3
        df2.drop('total_phases', axis=1, inplace=True)                             # 4
        df2['county'].fillna(value='statewide', inplace=True)                      # 5
        df2['county'] = df2['county'].str.lower()                                  # 6
        df2['date'] = pd.to_datetime(df2['date'], format='%Y-%m-%d')               # 7


        # Fill mismatches between df and df2
        county_match  = re.compile(" county$")
        munici_match  = re.compile(" municipality$")
        Borough_match = re.compile(" borough$")

        df2['county'].replace(to_replace=county_match,  value='', inplace=True)
        df2['county'].replace(to_replace=munici_match,  value='', inplace=True)
        df2['county'].replace(to_replace=Borough_match, value='', inplace=True)

        # Replace nulls in fips_code with 2 digit fips code for state for statewide policies.

        for index, data in df2.iterrows():
            if data.policy_level == 'state':
                df2.loc[index, 'fips_code'] = np.int64(us.states.lookup(data.state).fips)

        # Convert fips_code to integers
        df2['fips_code'] = df2['fips_code'].astype(np.int64)

        # Drop policies with invalid dates
        df2 = df2.drop(df2[(df2['date'] < min(df['date'])) | (df2['date'] > datetime.today())].index)

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

        return df, df2


        def get_cases(self, level="county", county="orange", state="California", df=None):
            """ Return the new_case and new_death numbers at the given level of aggregation (county, state, or national).

        Parameters
        ----------
        level : {'county', 'state', 'national'}
            If county, returns a DataFrame filtered to a specific county (default).
            If state, aggregates the DataFrame to the state level.
            If national, or any other input, returns the DataFrame aggregated to the national level.
        county : string
            desired county
        state : string
            desired state
        df : pandas DataFrame
            DataFrame to use, df by default

        Returns
        ----------
        DataFrame
            case_data filtered to a specific county or aggregated to the state / national level with index=date
        """
        # return county level data
        if level == "county":
            return_df = df[(df["county"] == county) & (df["state"] == state)].set_index("date")[['new_cases_1e6',
                                                                                                'new_deaths_1e6',
                                                                                                'new_cases_7day_1e6',
                                                                                                'new_deaths_7day_1e6',
                                                                                                ]]

            return return_df

        # If this is filtered at the state level, filter df to desired state. Otherwise, return national-level data.
        if level == "state":
            df = df[df['state'] == state]

        # Reindex on location name.
        df = df.set_index(["full_loc_name"])

        # Get a list of all dates.
        all_dates = df['date'].unique()

        # Get the total population from the county populations.
        total_population =  sum([(pops / 1e5) for pops in df[df['date'] == all_dates[0]]['total_population']])

        # Add up the case and death #s that have the same date.
        new_cases       = [sum([(county_cases / total_population)
                                for county_cases in df[df['date'] == dates]['new_cases_1e6']])
                                for dates in all_dates]

        new_deaths      = [sum([(county_cases / total_population)
                                for county_cases in df[df['date'] == dates]['new_deaths_1e6']])
                                for dates in all_dates]

        new_cases_7day  = [sum([(county_cases / total_population)
                                for county_cases in df[df['date'] == dates]['new_cases_7day_1e6']])
                                for dates in all_dates]

        new_deaths_7day = [sum([(county_cases / total_population)
                                for county_cases in df[df['date'] == dates]['new_deaths_7day_1e6']])
                                for dates in all_dates]


        return_df = pd.DataFrame(data={'date'               : all_dates,
                                    'new_cases_1e6'      : new_cases,
                                    'new_deaths_1e6'     : new_deaths,
                                    'new_cases_7day_1e6' : new_cases_7day,
                                    'new_deaths_7day_1e6': new_deaths_7day
                                    }).set_index(["date"])
        return return_df

    def plot_cases(self, level="county", county="orange", state="California", df=None, fade=0.75, style="whitegrid"):

        """ A function which plots the COVID-19 case/death data and 7 day average.

        Parameters
        ----------
        level : {'county', 'state', 'national'}
            Value to pass to get_cases()
        county : string
            desired county
        state : string
            desired state
        df : pandas DataFrame
            DataFrame to use, df by default
        style : string
            Seaborn plot style (default: "whitegrid")
        fade : float
            level of transparency for new_cases_1e6 and new_deaths_1e6 (default: 0.75)

        Returns
        ----------
        matplotlib.figure.Figure
        ndarray containing the two axis handles
        pandas DataFrame
        """

        # Get the data.
        cases = get_cases(level, county, state, df)

        # Set up plots.
        fig, ax = plt.subplots(2, 1, figsize=(10,5), sharex=True);
        plt.subplots_adjust(hspace=0.02)
        sns.set_style(style)

        # Plot cases.
        cases.plot(
            color=[f'{fade}', 'k'],
            y=["new_cases_1e6","new_cases_7day_1e6"],
            label=["Cases per capita", "7-day average"],
            ax=ax[0]
            );

        # Plot deaths.
        cases.plot(
            color=[f'{fade}', 'k'],
            y=["new_deaths_1e6","new_deaths_7day_1e6"],
            label=["Deaths per capita", "7-day average"],
            ax=ax[1]
            );

        # Format axis labels.
        ax[0].set_ylabel("cases per 100,000")
        ax[1].set_ylabel("deaths per 100,000")

        # Set plot title based on level of aggregeation (county, state, or national)
        if level == "county":
            ax[0].set_title(f"New COVID-19 cases and deaths per 100,000 in {county} County, {state}");
        elif level == "state":
            ax[0].set_title(f"New COVID-19 cases and deaths per 100,000 in {state}");
        else:
            ax[0].set_title("New COVID-19 cases and deaths per 100,000 in the United States")

        return fig, ax, cases

    def get_policy_data(self, state="California", county="statewide", state_policies=True, county_policies=True, df=None):
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

            return df[(df['state'] == state) &
                    ((df["county"] == county) | (df["county"] == "statewide"))]

        # state policies only
        elif state_policies and not county_policies:
            return df[ (df['state'] == state) & (df["county"] == "statewide")]

        # county policies only
        else:
            return df[ (df['state'] == state) & (df["county"] == county)]

    def plot_case_and_policies(
            self, county, state, df=None, df2=None, style="whitegrid", colors=['k', 'b', 'r', 'g'],
            policies=[
                "mandate face masks in public spaces",
                "mandate face masks in businesses",
                "shelter in place",
                "state of emergency"],

            labels=["face mask mandate (public spaces)",
                "face mask mandate (businesses)",
                "shelter in place",
                "state of emergency"],

            fade=0.75):


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
        plt.figure();
        fig, ax, df = plot_cases(level="county", county=county, state=state, fade=fade, style=style, df=df);


        # Get the policy data for the selected state and county.
        policy_data = get_policy_data(state, county, df=df2);

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
        legend_position = (1, 1)
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
                days_serial = (row.date - pd.Timestamp(year=1970, month=1, day=1)).days
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
                line = ax[i].axvline(x         = row.date,
                                    linestyle = styles_start_stop[row.start_stop],
                                    color     = styles_policy_type[row.policy_type],
                                    lw        = 2,
                                    ymin      = hmin,
                                    ymax      = hmax,
                                    )

                # Add the policy to the legend if it's not already in the list of labels
                if (row.policy_type not in labels):
                    labels.append(row.policy_type);

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
        leg2 = ax[0].legend(handles=legend_lines, bbox_to_anchor=legend_position)
        ax[0].add_artist(leg1)

        return ax

    def calculate_deltas(measure_period=14, filtered_policies=None, case_df=None, policy_df=None):
        """For every policy implementation at the state and county level, calculate the change in case and death numbers.

        Parameters
        ----------
        measure_period : int
            time to wait (in days) before measuring a change in new case or death numbers (14 by default)
        filtered_policies : array-like
            specify policies to select (defaul: None- calulate deltas for all policies)
        case_df : pandas DataFrame
            DataFrame with case / death information (default: df)
        policy_df : pandas DataFrame
            DataFrame with police information (default: df2)

        Returns
        ----------
        A copy of the covid policies df (df2) with 2 appended columns for the change in case and death numbers.
        """

        # Initialize wait period before measurement.
        wait_period = timedelta(days=measure_period)
        day_1 = timedelta(days=1)

        def sub_calc_deltas(ser, date, wait=wait_period):
            """Wrap repeated calculations in a sub function to avoid repetition."""
            day_1 = timedelta(days=1)

            start      = ser[ser.index==date].values[0]
            start_1day = ser[ser.index==date+day_1].values[0]

            end        = ser[ser.index==date+wait].values[0]
            end_1day   = ser[ser.index==date+wait+day_1].values[0]

            return [start, start_1day, end, end_1day]

        # If there we are only examining select policies, then filter those out.
        if filtered_policies is not None:
            policy_df = policy_df.loc[policy_df['policy_type'].isin(filtered_policies)]

        correlated_df = policy_df.copy()

        # Initially fill the delta column with nan.
        correlated_df.loc[:, f"case_{measure_period}_day_delta"] = np.nan
        correlated_df.loc[:, f"case_{measure_period}_day_accel"] = np.nan
        correlated_df.loc[:, f"death_{measure_period}_day_delta"] = np.nan
        correlated_df.loc[:, f"death_{measure_period}_day_accel"] = np.nan

        # Load all state-aggregated datasets into a dictionary. We expect to need all 50 states so let's take the time to aggregate
        # the state data now so we don't need to do it repeatedly in the loop.

        state_cases_dict = dict()
        for state in [elem.name for elem in us.states.STATES]:
            state_cases_dict[state]=get_cases(level="state", state=state);

        case_df = case_df.set_index('date')
        total_policies = len(policy_df)

        for index, data in policy_df.iterrows():

            # If this is a state-level policy, then we already have the DataFrame to use.
            if data.policy_level == 'state':
                state_df = state_cases_dict[data.state]
                ser_cases = state_df['new_cases_7day_1e6' ]
                ser_deaths = state_df['new_deaths_7day_1e6']

            # This must be a county level policy- filter the appropriate data.
            else:
                ser_cases = case_df['new_cases_7day_1e6' ][case_df['fips_code'] == data.fips_code]
                ser_deaths = case_df['new_deaths_7day_1e6'][case_df['fips_code'] == data.fips_code]

            # Get the case and death numbers at the appropriate days.
            c11, c12, c21, c22 = sub_calc_deltas(ser_cases, date=data.date)
            d11, d12, d21, d22 = sub_calc_deltas(ser_deaths, date=data.date)

            # Calculate the difference in new cases at the selected dates.
            correlated_df.at[index, f"case_{measure_period}_day_delta"] = c21 - c11
            correlated_df.at[index, f"death_{measure_period}_day_delta"] = d21 - d11

            # Calculate the change in curvature (aka acceleration) of the case / death plots at policy implementation and
            # measure_period days afterwards.

            correlated_df.at[index, f"case_{measure_period}_day_accel"] = ((c12-c11) - (c21-c22)) / measure_period
            correlated_df.at[index, f"death_{measure_period}_day_accel"] = ((d12-d11) - (d21-d22)) / measure_period

        return correlated_df

    def calc_delta_stats(deltas, measure_period=14, min_samples=10):
        """Take the deltas calculated with each policy and calculate the average and sd.
        Parameters
        ----------
        deltas : pandas DataFrame
            dataframe of policy deltas on which to do the calculations
        measure_period : int
            time to wait (in days) before measuring a change in new case or death numbers (14 by default)
        min_samples : int
            minimum number of samples that a policy must have for reporting of average and std (default: 10)

        Returns
        ----------
        A dataframe with a record for the start/stop of each policy type and the average / std of the change in
        case / death numbers measure_period days after implementation
        """

        # Generate a new list of policy types differentiating between start and stop.
        policy_types = ([elem + " - start" for elem in deltas['policy_type'].unique()]
                        + [elem + " - stop"  for elem in deltas['policy_type'].unique()])

        # Initialize empty arrays for the associated statistics.
        case_avg, death_avg, case_std, death_std, num_samples = [], [], [], [], []
        case_accel_avg, death_accel_avg, case_accel_std, death_accel_std = [], [], [], []

        # Loop through all the policy types.
        for policy in policy_types:

            # Determine whether this policy is the beginning or end.
            if policy.endswith("stop"):
                len_index = -7
                start_stop = "stop"
            else:
                len_index = -8
                start_stop = "start"

            # Get arrays of all the deltas for each type of policy
            case_data  = deltas[(deltas['policy_type'] == policy[:len_index]) &
                                (deltas['start_stop'] == start_stop)][f'case_{measure_period}_day_delta']

            death_data = deltas[(deltas['policy_type'] == policy[:len_index]) &
                                (deltas['start_stop'] == start_stop)][f'death_{measure_period}_day_delta']

            case_accel_data = deltas[(deltas['policy_type'] == policy[:len_index]) &
                                    (deltas['start_stop'] == start_stop)][f'case_{measure_period}_day_accel']

            death_accel_data = deltas[(deltas['policy_type'] == policy[:len_index]) &
                                    (deltas['start_stop'] == start_stop)][f'death_{measure_period}_day_accel']

            num_samples.append(len(case_data))

            # Calculate the averages and standard deviations for each policy
            case_avg.append(np.nanmean(case_data.to_numpy()))
            death_avg.append(np.nanmean(death_data.to_numpy()))

            case_std.append(np.nanstd(case_data.to_numpy()))
            death_std.append(np.nanstd(death_data.to_numpy()))

            case_accel_avg.append(np.nanmean(case_accel_data.to_numpy()))
            death_accel_avg.append(np.nanmean(death_accel_data.to_numpy()))

            case_accel_std.append(np.nanstd(case_accel_data.to_numpy()))
            death_accel_std.append(np.nanstd(death_accel_data.to_numpy()))

        # Construct the dataframe to tabulate the data.
        delta_stats = pd.DataFrame(np.transpose([case_avg, case_accel_avg, death_avg, death_accel_avg,
                                                case_std, case_accel_std, death_std, death_accel_std,
                                                num_samples]), index=policy_types,
                                columns=['case_avg', 'case_accel_avg', 'death_avg', 'death_accel_avg',
                                            'case_std', 'case_accel_std', 'death_std', 'death_accel_std',
                                            'num_samples']
                                )

        # Drop record with less than min_samples samples.
        delta_stats.drop(delta_stats[delta_stats['num_samples'] <= min_samples].index, inplace=True)

        return delta_stats