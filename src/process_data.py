# TODO: Encode categorical features (Driver, Team, Weather, Track) and normalize numerical features before modeling.
import pandas as pd
import ast
from sklearn.preprocessing import LabelEncoder
import pickle
import os

def safe_encode_column(column_data, encoder, unseen_value="Unknown"):
    """
    Encode a column with handling for unseen categories.
    Maps unseen categories to unseen_value.
    """
    try:
        return encoder.transform(column_data)
    except ValueError:
        # Handle unseen categories by mapping them to unseen_value
        encoded = []
        for value in column_data:
            try:
                encoded.append(encoder.transform([value])[0])
            except ValueError:
                # Map unseen categories to the "Unknown" label
                encoded.append(encoder.transform([unseen_value])[0])
        return encoded

def fit_encoders(df, categorical_columns):
    """
    Fit LabelEncoders for categorical columns and save them for future use.
    """
    encoders = {}
    encoder_dir = 'encoders'
    
    # Create encoders directory if it doesn't exist
    if not os.path.exists(encoder_dir):
        os.makedirs(encoder_dir)
    
    for col in categorical_columns:
        if col in df.columns:
            encoder = LabelEncoder()
            # Fit on non-null values plus "Unknown" for unseen category handling
            non_null_data = df[col].dropna()
            if len(non_null_data) > 0:
                # Add "Unknown" to the data to ensure it's included as a valid class
                data_with_unknown = list(non_null_data.unique()) + ["Unknown"]
                encoder.fit(data_with_unknown)
                encoders[col] = encoder
                
                # Save encoder for future use
                with open(f'{encoder_dir}/{col}_encoder.pkl', 'wb') as f:
                    pickle.dump(encoder, f)
                print(f"Fitted and saved encoder for {col} (includes 'Unknown' class)")
    
    return encoders

def load_encoders(categorical_columns):
    """
    Load previously fitted encoders for categorical columns.
    """
    encoders = {}
    encoder_dir = 'encoders'
    
    for col in categorical_columns:
        encoder_path = f'{encoder_dir}/{col}_encoder.pkl'
        if os.path.exists(encoder_path):
            with open(encoder_path, 'rb') as f:
                encoders[col] = pickle.load(f)
            print(f"Loaded encoder for {col}")
        else:
            print(f"No saved encoder found for {col}")
    
    return encoders

def encode_categorical_features(df, categorical_columns, fit_encoders_flag=True):
    """
    Encode categorical features using LabelEncoder.
    If fit_encoders_flag is True, fit new encoders. Otherwise, load existing ones.
    """
    if fit_encoders_flag:
        encoders = fit_encoders(df, categorical_columns)
    else:
        encoders = load_encoders(categorical_columns)
    
    # Encode each categorical column
    for col in categorical_columns:
        if col in df.columns and col in encoders:
            # Fill missing values with 'Unknown' before encoding
            df[col] = df[col].fillna('Unknown')
            
            # Encode the column
            df[col] = safe_encode_column(df[col], encoders[col])
            print(f"Encoded {col} with {len(encoders[col].classes_)} unique values")
        elif col in df.columns:
            print(f"Warning: No encoder found for {col}, skipping encoding")
    
    return df

def process_f1_data(input_csv='raw_f1_data.csv', output_csv='new_clean_f1_data.csv', fit_encoders=True):
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
    df['QualifyingPosition'] = df['QualifyingPosition'].fillna(-1)
    df['GridPosition'] = df['GridPosition'].fillna(-1)
    df['FinishingPosition'] = df['FinishingPosition'].fillna(-1)
    df['AirTemp'] = df['AirTemp'].fillna(df['AirTemp'].mean())
    df['TrackTemp'] = df['TrackTemp'].fillna(df['TrackTemp'].mean())
    df['PitStopCount'] = df['PitStopCount'].fillna(df['PitStopCount'].median())
    df['NumLaps'] = df['NumLaps'].fillna(df['NumLaps'].mean())
    df['Weather'] = df['Weather'].fillna('Unknown')
    df['WeatherCategory'] = df['WeatherCategory'].fillna('Unknown')
    
    # Define categorical columns for encoding (removed Weather to avoid redundancy with WeatherCategory)
    categorical_columns = ['Driver', 'Team', 'Track', 'WeatherCategory']
    
    # Encode categorical features
    print("Encoding categorical features...")
    df = encode_categorical_features(df, categorical_columns, fit_encoders)
    
    # Drop Weather column to avoid redundancy with WeatherCategory
    if 'Weather' in df.columns:
        df = df.drop('Weather', axis=1)
        print("Dropped Weather column (redundant with WeatherCategory)")
    
    # Ensure target column (FinishingPosition) remains numeric and is not encoded
    df['FinishingPosition'] = pd.to_numeric(df['FinishingPosition'], errors='coerce')
    df['FinishingPosition'] = df['FinishingPosition'].fillna(-1)
    
    # Logging improvements - show processed data preview
    print("\n" + "="*50)
    print("PROCESSED DATA PREVIEW:")
    print("="*50)
    print(df.head())
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Categorical columns encoded: {categorical_columns}")
    print(f"Target column: FinishingPosition (numeric)")
    
    # Print unique class counts for each encoded categorical column
    print("\n" + "="*50)
    print("ENCODED CATEGORICAL COLUMNS - UNIQUE CLASS COUNTS:")
    print("="*50)
    for col in categorical_columns:
        if col in df.columns:
            unique_count = df[col].nunique()
            print(f"{col}: {unique_count} unique classes")
    
    # Save XGBoost-ready data
    df.to_csv(output_csv, index=False)
    print(f"\nSaved XGBoost-ready data to {output_csv}")

if __name__ == "__main__":
    process_f1_data()
