import fastf1
import pandas as pd

def get_race_laps(year=2025, gp='China', session_type='R'):
    fastf1.Cache.enable_cache('cache')  # create a 'cache/' directory for downloaded files
    session = fastf1.get_session(year, gp, session_type)
    session.load()
    return session.laps

if __name__ == "__main__":
    laps = get_race_laps()

    # Clean up and convert laptimes to seconds
    laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()

    # Show basic info
    print("🏁 Total laps:", len(laps)/20)
    print("\n📊 Drivers in race:", laps['Driver'].unique())
    print("\n🚦 First 10 lap entries:")
    print(laps[['Driver', 'LapNumber', 'LapTimeSeconds', 'Position', 'Compound', 'PitOutTime', 'PitInTime']].head(10))