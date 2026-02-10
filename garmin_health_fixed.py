"""
Garmin Health Data Scraper - FIXED for garth >= 0.6
Stahuje HRV, spánek, stress, body battery, denní souhrn
"""

import os
import json
from datetime import datetime, timedelta, date
from pathlib import Path
from dotenv import load_dotenv
import garth
import pandas as pd
from collections import defaultdict

load_dotenv()


class GarminHealthScraper:
    def __init__(self):
        self.email = os.getenv('GARMIN_EMAIL')
        self.password = os.getenv('GARMIN_PASSWORD')
        self.data_folder = Path(os.getenv('DATA_FOLDER', 'garmin_data'))
        self.data_folder.mkdir(exist_ok=True)

        self.health_folder = self.data_folder / "health"
        self.health_folder.mkdir(exist_ok=True)

        self.token_store = self.data_folder / ".garth"

    def login(self):
        try:
            if self.token_store.exists():
                import shutil
                if self.token_store.is_file():
                    self.token_store.unlink()
                else:
                    shutil.rmtree(self.token_store)

            print("Logging in...")
            garth.login(self.email, self.password)
            garth.save(self.token_store)
            print(f"Logged in as {self.email}")
            return True
        except Exception as err:
            print(f"Login failed: {err}")
            return False

    def get_date_range(self, days_back=90):
        end = date.today()
        start = end - timedelta(days=days_back)
        dates = []
        current = start
        while current <= end:
            dates.append(current)
            current += timedelta(days=1)
        return dates

    def fetch_sleep_data(self):
        print("\n[SLEEP] Fetching sleep data...")
        sleep_data = []
        for d in self.get_date_range(90):
            try:
                resp = garth.connectapi(
                    "/wellness-service/wellness/dailySleepData",
                    params={"date": d.strftime('%Y-%m-%d')}
                )
                if not resp or not isinstance(resp, dict):
                    continue
                # Sleep data may be at top level or nested in dailySleepDTO
                data = resp.get('dailySleepDTO', resp)
                if not isinstance(data, dict) or not data.get('sleepTimeSeconds'):
                    continue
                sleep_scores = data.get('sleepScores', {})
                overall = sleep_scores.get('overall', {}) if isinstance(sleep_scores, dict) else {}
                sleep_data.append({
                    'date': d.strftime('%Y-%m-%d'),
                    'sleepTimeSeconds': data.get('sleepTimeSeconds'),
                    'sleepTimeHours': round(data.get('sleepTimeSeconds', 0) / 3600, 2),
                    'deepSleepSeconds': data.get('deepSleepSeconds'),
                    'lightSleepSeconds': data.get('lightSleepSeconds'),
                    'remSleepSeconds': data.get('remSleepSeconds'),
                    'awakeSleepSeconds': data.get('awakeSleepSeconds'),
                    'sleepScore': overall.get('value') if isinstance(overall, dict) else None,
                })
            except:
                continue
        print(f"  Got {len(sleep_data)} nights")
        return sleep_data

    def fetch_stress_data(self):
        print("\n[STRESS] Fetching stress data...")
        stress_data = []
        for d in self.get_date_range(90):
            try:
                data = garth.connectapi(
                    f"/wellness-service/wellness/dailyStress/{d.strftime('%Y-%m-%d')}"
                )
                if data and isinstance(data, dict):
                    stress_data.append({
                        'date': d.strftime('%Y-%m-%d'),
                        'avgStressLevel': data.get('avgStressLevel'),
                        'maxStressLevel': data.get('maxStressLevel'),
                        'restStressLevel': data.get('restStressLevel'),
                    })
            except:
                continue
        print(f"  Got {len(stress_data)} days")
        return stress_data

    def fetch_heart_rate_data(self):
        print("\n[HEART RATE] Fetching resting heart rate...")
        hr_data = []
        for d in self.get_date_range(90):
            try:
                data = garth.connectapi(
                    "/wellness-service/wellness/dailyHeartRate",
                    params={"date": d.strftime('%Y-%m-%d')}
                )
                if data and isinstance(data, dict) and data.get('restingHeartRate'):
                    hr_data.append({
                        'date': d.strftime('%Y-%m-%d'),
                        'restingHeartRate': data.get('restingHeartRate'),
                        'minHeartRate': data.get('minHeartRate'),
                        'maxHeartRate': data.get('maxHeartRate'),
                        'weeklyAvgRestingHR': data.get('lastSevenDaysAvgRestingHeartRate'),
                    })
            except:
                continue
        print(f"  Got {len(hr_data)} days")
        return hr_data

    def fetch_hrv_data(self):
        print("\n[HRV] Fetching HRV data...")
        hrv_data = []
        dates = self.get_date_range(90)
        start_str = dates[0].strftime('%Y-%m-%d')
        end_str = dates[-1].strftime('%Y-%m-%d')
        try:
            data = garth.connectapi(
                f"/hrv-service/hrv/daily/{start_str}/{end_str}"
            )
            if data and isinstance(data, dict):
                for entry in data.get('hrvSummaries', []):
                    hrv_data.append({
                        'date': entry.get('calendarDate'),
                        'hrvWeeklyAvg': entry.get('weeklyAvg'),
                        'hrvLastNightAvg': entry.get('lastNightAvg'),
                        'hrvLastNight5MinHigh': entry.get('lastNight5MinHigh'),
                        'hrvStatus': entry.get('status'),
                    })
        except Exception as e:
            print(f"  Error: {e}")
        print(f"  Got {len(hrv_data)} days")
        return hrv_data

    def fetch_training_readiness(self):
        print("\n[TRAINING READINESS] Fetching training readiness...")
        tr_data = []
        for d in self.get_date_range(90):
            try:
                resp = garth.connectapi(
                    f"/metrics-service/metrics/trainingreadiness/{d.strftime('%Y-%m-%d')}"
                )
                if resp and isinstance(resp, list) and resp:
                    r = resp[0]
                    if r.get('score') is not None:
                        tr_data.append({
                            'date': d.strftime('%Y-%m-%d'),
                            'readinessScore': r.get('score'),
                            'readinessLevel': r.get('level'),
                            'recoveryTimeMin': r.get('recoveryTime'),
                            'acuteLoad': r.get('acuteLoad'),
                            'acwrPercent': r.get('acwrFactorPercent'),
                            'sleepScoreFactor': r.get('sleepScoreFactorPercent'),
                            'stressHistoryFactor': r.get('stressHistoryFactorPercent'),
                            'hrvFactor': r.get('hrvFactorPercent'),
                            'sleepHistoryFactor': r.get('sleepHistoryFactorPercent'),
                            'recoveryTimeFactor': r.get('recoveryTimeFactorPercent'),
                        })
            except:
                continue
        print(f"  Got {len(tr_data)} days")
        return tr_data

    def fetch_endurance_hill_scores(self):
        print("\n[SCORES] Fetching endurance & hill scores...")
        score_data = []
        for d in self.get_date_range(90):
            ds = d.strftime('%Y-%m-%d')
            row = {'date': ds}
            found = False
            try:
                resp = garth.connectapi(
                    '/metrics-service/metrics/endurancescore',
                    params={'calendarDate': ds}
                )
                if resp and isinstance(resp, dict) and resp.get('overallScore') is not None:
                    row['enduranceScore'] = resp.get('overallScore')
                    row['enduranceClass'] = resp.get('classification')
                    found = True
            except:
                pass
            try:
                resp = garth.connectapi(
                    '/metrics-service/metrics/hillscore',
                    params={'calendarDate': ds}
                )
                if resp and isinstance(resp, dict) and resp.get('overallScore') is not None:
                    row['hillScore'] = resp.get('overallScore')
                    row['hillStrength'] = resp.get('strengthScore')
                    row['hillEndurance'] = resp.get('enduranceScore')
                    found = True
            except:
                pass
            if found:
                score_data.append(row)
        print(f"  Got {len(score_data)} days")
        return score_data

    def fetch_vo2max_history(self):
        print("\n[VO2 MAX] Fetching VO2 Max history...")
        vo2_data = []
        dates = self.get_date_range(90)
        start_str = dates[0].strftime('%Y-%m-%d')
        end_str = dates[-1].strftime('%Y-%m-%d')
        try:
            data = garth.connectapi(
                f"/metrics-service/metrics/maxmet/daily/{start_str}/{end_str}"
            )
            if data and isinstance(data, list):
                for entry in data:
                    generic = entry.get('generic', {})
                    if generic:
                        vo2_data.append({
                            'date': generic.get('calendarDate'),
                            'vo2MaxPrecise': generic.get('vo2MaxPreciseValue'),
                            'vo2Max': generic.get('vo2MaxValue'),
                        })
        except Exception as e:
            print(f"  Error: {e}")
        print(f"  Got {len(vo2_data)} readings")
        return vo2_data

    def fetch_daily_summary(self):
        print("\n[DAILY SUMMARY] Fetching daily summaries...")
        summary_data = []
        for d in self.get_date_range(90):
            try:
                data = garth.connectapi(
                    "/usersummary-service/usersummary/daily",
                    params={"calendarDate": d.strftime('%Y-%m-%d')}
                )
                if data and isinstance(data, dict) and data.get('totalSteps') is not None:
                    summary_data.append({
                        'date': d.strftime('%Y-%m-%d'),
                        'steps': data.get('totalSteps'),
                        'distance_km': round(data.get('totalDistanceMeters', 0) / 1000, 2) if data.get('totalDistanceMeters') else None,
                        'activeCalories': data.get('activeKilocalories'),
                        'totalCalories': data.get('totalKilocalories'),
                        'highlyActiveMin': round(data.get('highlyActiveSeconds', 0) / 60, 1) if data.get('highlyActiveSeconds') else None,
                        'activeMin': round(data.get('activeSeconds', 0) / 60, 1) if data.get('activeSeconds') else None,
                        'sedentaryMin': round(data.get('sedentarySeconds', 0) / 60, 1) if data.get('sedentarySeconds') else None,
                        'floorsAscended': data.get('floorsAscended'),
                        'bodyBatteryHighest': data.get('bodyBatteryHighestValue'),
                        'bodyBatteryLowest': data.get('bodyBatteryLowestValue'),
                        'bodyBatteryAtWake': data.get('bodyBatteryAtWakeTime'),
                        'bodyBatteryCharged': data.get('bodyBatteryChargedValue'),
                        'bodyBatteryDrained': data.get('bodyBatteryDrainedValue'),
                        'avgSpo2': data.get('averageSpo2'),
                        'lowestSpo2': data.get('lowestSpo2'),
                        'avgRespiration': data.get('avgWakingRespirationValue'),
                        'moderateIntensityMin': data.get('moderateIntensityMinutes'),
                        'vigorousIntensityMin': data.get('vigorousIntensityMinutes'),
                    })
            except:
                continue
        print(f"  Got {len(summary_data)} days")
        return summary_data

    def save_health_data(self, all_data):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        json_file = self.health_folder / f"health_data_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, indent=2, ensure_ascii=False)
        print(f"\nJSON: {json_file.name}")

        saved_files = []
        for key, data in all_data.items():
            if data and isinstance(data, list) and len(data) > 0:
                df = pd.DataFrame(data)
                csv_file = self.health_folder / f"{key}_{timestamp}.csv"
                df.to_csv(csv_file, index=False, encoding='utf-8')
                print(f"CSV: {key} -> {csv_file.name} ({len(data)} records)")
                saved_files.append(csv_file)

        return saved_files

    def create_master_dataset(self, all_data):
        print("\n--- CREATING MASTER DATASET ---")

        try:
            # Load activities
            activity_files = sorted(self.data_folder.glob("activities_*.csv"))
            if activity_files:
                activities_df = pd.read_csv(activity_files[-1])
                print(f"  Activities: {len(activities_df)} records from {activity_files[-1].name}")
            else:
                activities_df = pd.DataFrame()
                print("  No activities file found")

            # Build master by date
            master = defaultdict(dict)

            # Add activities aggregated by day
            if not activities_df.empty:
                for datum, group in activities_df.groupby('datum'):
                    master[datum]['training_distance_km'] = round(group['vzdalenost_km'].sum(), 2)
                    master[datum]['training_time_min'] = round(group['cas_min'].sum(), 1)
                    master[datum]['training_avg_hr'] = round(group['prumer_tep'].mean(), 0) if group['prumer_tep'].mean() > 0 else None
                    master[datum]['training_max_hr'] = group['max_tep'].max() if group['max_tep'].max() > 0 else None
                    master[datum]['training_calories'] = round(group['kalorie'].sum(), 0)
                    master[datum]['training_count'] = len(group)

            # Add health data from all sources
            for key, records in all_data.items():
                if isinstance(records, list):
                    for record in records:
                        d = record.get('date')
                        if d:
                            for k, v in record.items():
                                if k != 'date' and v is not None:
                                    master[d][k] = v

            # Build DataFrame
            rows = []
            for d, values in sorted(master.items()):
                row = {'date': d}
                row.update(values)
                rows.append(row)

            master_df = pd.DataFrame(rows)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            master_file = self.data_folder / f"master_dataset_{timestamp}.csv"
            master_df.to_csv(master_file, index=False, encoding='utf-8')

            print(f"  Saved: {master_file.name}")
            print(f"  Days: {len(master_df)}, Columns: {len(master_df.columns)}")
            print(f"  Columns: {list(master_df.columns)}")

            return master_df

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    print("=" * 60)
    print("GARMIN HEALTH SCRAPER v2")
    print("=" * 60)

    scraper = GarminHealthScraper()

    if not scraper.email:
        print("Set GARMIN_EMAIL in .env!")
        return

    print(f"Email: {scraper.email}\n")

    if not scraper.login():
        return

    print(f"\nFetching 90 days of data...")

    all_data = {}
    all_data['sleep'] = scraper.fetch_sleep_data()
    all_data['stress'] = scraper.fetch_stress_data()
    all_data['heart_rate'] = scraper.fetch_heart_rate_data()
    all_data['hrv'] = scraper.fetch_hrv_data()
    all_data['training_readiness'] = scraper.fetch_training_readiness()
    all_data['scores'] = scraper.fetch_endurance_hill_scores()
    all_data['vo2max'] = scraper.fetch_vo2max_history()
    all_data['daily_summary'] = scraper.fetch_daily_summary()

    print("\n" + "=" * 60)
    print("SAVING")
    print("=" * 60)

    scraper.save_health_data(all_data)

    print("\n" + "=" * 60)
    print("MASTER DATASET")
    print("=" * 60)

    scraper.create_master_dataset(all_data)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total = 0
    for key, data in all_data.items():
        if isinstance(data, list):
            count = len(data)
            total += count
            status = "OK" if count > 0 else "EMPTY"
            print(f"  [{status}] {key}: {count} records")

    print(f"\n  Total: {total} records")
    print(f"  Folder: {scraper.health_folder}")

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
