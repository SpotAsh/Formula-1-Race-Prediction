import fastf1

def get_race_laps(year=2023, gp='Monza', session_type='R'):
    session = fastf1.get_session(year, gp, session_type)  # R = Race
    session.load()
    laps = session.laps
    return laps
