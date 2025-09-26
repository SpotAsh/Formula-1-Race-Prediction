import fastf1
import pandas as pd
import os

fastf1.Cache.enable_cache('cache')  # Adjust path as needed

def fetch_race_data(start_year=2022, end_year=2024, output_csv='raw_f1_data.csv'):
    if end_year is None:
        from datetime import datetime
        end_year = datetime.now().year
    all_data = []
    for year in range(start_year, end_year+1):
        try:
            schedule = fastf1.get_event_schedule(year)
            if schedule.empty:
                print(f"No events found for {year}")
                continue
        except Exception as e:
            print(f"Failed to get event schedule for {year}: {e}")
            continue
            
        for _, event in schedule.iterrows():
            try:
                gp = event['EventName']
                round_num = event['RoundNumber']
            except (KeyError, TypeError) as e:
                print(f"Invalid event data for {year}: {e}")
                continue
                
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
            try:
                quali_results = quali.results[['DriverNumber', 'Abbreviation', 'TeamName', 'Position']].copy()
                quali_results.rename(columns={'Position': 'QualifyingPosition'}, inplace=True)
            except Exception as e:
                print(f"Skipping {year} {gp} due to missing qualifying results: {e}")
                continue
            # Race results
            try:
                race_results = race.results[['DriverNumber', 'Abbreviation', 'TeamName', 'GridPosition', 'Position']].copy()
                race_results.rename(columns={'Position': 'FinishingPosition'}, inplace=True)
            except Exception as e:
                print(f"Skipping {year} {gp} due to missing race results: {e}")
                continue
            # Merge on DriverNumber
            try:
                merged = pd.merge(race_results, quali_results[['DriverNumber', 'QualifyingPosition']], on='DriverNumber', how='left')
                if merged.empty:
                    print(f"Skipping {year} {gp} due to empty merged results")
                    continue
            except Exception as e:
                print(f"Skipping {year} {gp} due to merge error: {e}")
                continue
            # Weather info
            try:
                weather = race.weather_data
                air_temp = weather['AirTemp'].mean() if not weather.empty else None
                track_temp = weather['TrackTemp'].mean() if not weather.empty else None
                rainfall_sum = weather['Rainfall'].sum() if not weather.empty else None
            except Exception:
                weather = pd.DataFrame()
                air_temp = None
                track_temp = None
                rainfall_sum = None
            if rainfall_sum is None:
                weather_category = 'Unknown'
            elif rainfall_sum == 0:
                weather_category = 'Dry'
            elif 0 < rainfall_sum < 5:
                weather_category = 'Mixed'
            else:
                weather_category = 'Wet'
            # Number of race laps
            try:
                num_laps = race.total_laps
            except (AttributeError, Exception):
                if hasattr(race, 'laps') and race.laps is not None and hasattr(race.laps, '__len__'):
                    num_laps = len(race.laps)
                else:
                    num_laps = None
            for _, row in merged.iterrows():
                try:
                    driver = row['Abbreviation']
                    all_data.append({
                        'Year': year,
                        'Round': round_num,
                        'Track': gp,
                        'Driver': driver,
                        'Team': row.get('TeamName', 'Unknown'),
                        'QualifyingPosition': row.get('QualifyingPosition', -1),
                        'GridPosition': row.get('GridPosition', -1),
                        'FinishingPosition': row.get('FinishingPosition', -1),
                        'AirTemp': air_temp,
                        'TrackTemp': track_temp,
                        'Weather': weather_category,
                        'WeatherCategory': weather_category,
                        'NumLaps': num_laps,
                    })
                except Exception as e:
                    print(f"Error processing driver data for {year} {gp}: {e}")
                    continue
    
    if not all_data:
        print("No data collected. Check your parameters and network connection.")
        return
        
    try:
        df = pd.DataFrame(all_data)
        df.to_csv(output_csv, index=False)
        print(f"Saved {len(df)} records to {output_csv}")
    except Exception as e:
        print(f"Failed to save data: {e}")

if __name__ == "__main__":
    fetch_race_data()
