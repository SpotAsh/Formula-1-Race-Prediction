# TODO: Encode categorical features (Driver, Team, Weather, Track) and normalize numerical features before modeling.
import pandas as pd
import ast

def process_f1_data(input_csv='raw_f1_data.csv', output_csv='clean_f1_data.csv'):
    df = pd.read_csv(input_csv)
    # Convert columns
    df['QualifyingPosition'] = pd.to_numeric(df['QualifyingPosition'], errors='coerce')
    df['GridPosition'] = pd.to_numeric(df['GridPosition'], errors='coerce')
    df['FinishingPosition'] = pd.to_numeric(df['FinishingPosition'], errors='coerce')
    df['AirTemp'] = pd.to_numeric(df['AirTemp'], errors='coerce')
    df['TrackTemp'] = pd.to_numeric(df['TrackTemp'], errors='coerce')
    df['PitStopCount'] = pd.to_numeric(df['PitStopCount'], errors='coerce')
    df['NumLaps'] = pd.to_numeric(df['NumLaps'], errors='coerce')
    # Convert StintStrategy from string to list (if needed)
    def parse_stint(x):
        if isinstance(x, list):
            return x
        if isinstance(x, str) and x.startswith('['):
            try:
                return ast.literal_eval(x)
            except Exception:
                return []
        return []
    df['StintStrategy'] = df['StintStrategy'].apply(parse_stint)
    # Add StintCount column
    df['StintCount'] = df['StintStrategy'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    # Fill missing values
    df['QualifyingPosition'].fillna(-1, inplace=True)
    df['GridPosition'].fillna(-1, inplace=True)
    df['FinishingPosition'].fillna(-1, inplace=True)
    df['AirTemp'].fillna(df['AirTemp'].mean(), inplace=True)
    df['TrackTemp'].fillna(df['TrackTemp'].mean(), inplace=True)
    df['PitStopCount'].fillna(0, inplace=True)
    df['NumLaps'].fillna(df['NumLaps'].mean(), inplace=True)
    df['Weather'].fillna('Unknown', inplace=True)
    df['WeatherCategory'].fillna('Unknown', inplace=True)
    # Save cleaned data
    df.to_csv(output_csv, index=False)
    print(f"Saved cleaned data to {output_csv}")

if __name__ == "__main__":
    process_f1_data()
