import fastf1
import pandas as pd
import os

fastf1.Cache.enable_cache('../cache')  # Adjust path as needed

def fetch_race_data(start_year=2022, end_year=None, output_csv='raw_f1_data.csv'):
    if end_year is None:
        from datetime import datetime
        end_year = datetime.now().year
    all_data = []
    for year in range(start_year, end_year+1):
        schedule = fastf1.get_event_schedule(year)
        for _, event in schedule.iterrows():
            gp = event['EventName']
            round_num = event['RoundNumber']
            try:
                # Get qualifying session
                quali = fastf1.get_session(year, gp, 'Q')
                quali.load()
                # Get race session
                race = fastf1.get_session(year, gp, 'R')
                race.load()
            except Exception as e:
                print(f"Skipping {year} {gp}: {e}")
                continue
            # Qualifying positions
            quali_results = quali.results[['DriverNumber', 'Abbreviation', 'TeamName', 'Position']].copy()
            quali_results.rename(columns={'Position': 'QualifyingPosition'}, inplace=True)
            # Race results
            race_results = race.results[['DriverNumber', 'Abbreviation', 'TeamName', 'GridPosition', 'Position']].copy()
            race_results.rename(columns={'Position': 'FinishingPosition'}, inplace=True)
            # Merge on DriverNumber
            merged = pd.merge(race_results, quali_results[['DriverNumber', 'QualifyingPosition']], on='DriverNumber', how='left')
            # Weather info
            weather = race.weather_data
            air_temp = weather['AirTemp'].mean() if not weather.empty else None
            track_temp = weather['TrackTemp'].mean() if not weather.empty else None
            rainfall_sum = weather['Rainfall'].sum() if not weather.empty else None
            if rainfall_sum is None:
                weather_category = 'Unknown'
            elif rainfall_sum == 0:
                weather_category = 'Dry'
            elif 0 < rainfall_sum < 5:
                weather_category = 'Mixed'
            else:
                weather_category = 'Wet'
            # Pit stops
            pitstops = race.pit_stops.groupby('Driver')['Stop'].count().to_dict()
            # Stint strategy
            stints = race.stints.groupby('Driver')['Compound'].apply(list).to_dict()
            # Number of race laps
            try:
                num_laps = race.total_laps
            except AttributeError:
                num_laps = len(race.laps) if hasattr(race, 'laps') else None
            for _, row in merged.iterrows():
                driver = row['Abbreviation']
                all_data.append({
                    'Year': year,
                    'Round': round_num,
                    'Track': gp,
                    'Driver': driver,
                    'Team': row['TeamName'],
                    'QualifyingPosition': row['QualifyingPosition'],
                    'GridPosition': row['GridPosition'],
                    'FinishingPosition': row['FinishingPosition'],
                    'AirTemp': air_temp,
                    'TrackTemp': track_temp,
                    'Weather': weather_category,
                    'WeatherCategory': weather_category,
                    'PitStopCount': pitstops.get(driver, 0),
                    'StintStrategy': stints.get(driver, []),
                    'NumLaps': num_laps,
                })
    df = pd.DataFrame(all_data)
    df.to_csv(output_csv, index=False)
    print(f"Saved data to {output_csv}")

if __name__ == "__main__":
    fetch_race_data()
