"""
Garmin Connect Sync - FIXED VERZE
Správně zpracovává různé formáty odpovědí z API
"""

import os
import json
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
import garth
import pandas as pd

load_dotenv()

class GarminSync:
    def __init__(self):
        self.email = os.getenv('GARMIN_EMAIL')
        self.password = os.getenv('GARMIN_PASSWORD')
        self.data_folder = Path(os.getenv('DATA_FOLDER', 'garmin_data'))
        self.data_folder.mkdir(exist_ok=True)
        self.token_store = self.data_folder / ".garth"
        
    def login(self):
        """Přihlášení"""
        try:
            # Smažeme staré tokeny
            if self.token_store.exists():
                import shutil
                if self.token_store.is_file():
                    self.token_store.unlink()
                else:
                    shutil.rmtree(self.token_store)
            
            print("🔐 Přihlašuji se...")
            garth.login(self.email, self.password)
            garth.save(self.token_store)
            
            try:
                profile = garth.connectapi("/userprofile-service/userprofile")
                print(f"✅ Přihlášen: {profile.get('displayName', 'Uživatel')}")
            except:
                print(f"✅ Přihlášen!")
            
            return True
            
        except Exception as err:
            print(f"❌ Chyba: {err}")
            return False
    
    def extract_activities(self, response):
        """Extrahuje seznam aktivit z různých formátů odpovědi"""
        
        # Pokud je to přímo seznam
        if isinstance(response, list):
            return response
        
        # Pokud je to slovník, zkusíme různé klíče
        if isinstance(response, dict):
            # Zkusíme běžné klíče
            for key in ['activityList', 'activities', 'data', 'results']:
                if key in response:
                    return response[key]
            
            # Pokud slovník obsahuje jen 1 klíč, vrátíme jeho hodnotu
            if len(response) == 1:
                return list(response.values())[0]
        
        return None
    
    def get_activities(self):
        """Stažení aktivit - zkouší různé metody"""
        print(f"\n📥 Stahuji aktivity...")
        
        # Seznam metod k vyzkoušení
        methods = [
            # Metoda 1: Základní endpoint
            {
                'name': 'Základní endpoint',
                'url': '/activitylist-service/activities/search/activities',
                'params': {'limit': 50}
            },
            # Metoda 2: S paginací
            {
                'name': 'S paginací',
                'url': '/activitylist-service/activities/search/activities',
                'params': {'start': 0, 'limit': 50}
            },
            # Metoda 3: Date range
            {
                'name': 'Date range (90 dní)',
                'url': '/activitylist-service/activities/search/activities',
                'params': {
                    'startDate': (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d'),
                    'endDate': datetime.now().strftime('%Y-%m-%d'),
                    'limit': 100
                }
            },
            # Metoda 4: Bez parametrů
            {
                'name': 'Bez parametrů',
                'url': '/activitylist-service/activities/search/activities',
                'params': {}
            },
        ]
        
        for idx, method in enumerate(methods, 1):
            try:
                print(f"\n[Pokus {idx}] {method['name']}...")
                
                response = garth.connectapi(method['url'], params=method['params'])
                
                # Zjistíme typ odpovědi
                print(f"   Typ odpovědi: {type(response).__name__}")
                
                # Extrahujeme aktivity
                activities = self.extract_activities(response)
                
                if activities and len(activities) > 0:
                    print(f"   ✅ ÚSPĚCH! Nalezeno {len(activities)} aktivit")
                    return activities
                else:
                    print(f"   ⚠️ Prázdný seznam")
                    
            except Exception as e:
                print(f"   ❌ Selhalo: {type(e).__name__}: {e}")
                continue
        
        print("\n❌ Nepodařilo se stáhnout aktivity")
        return None
    
    def process_activities(self, activities):
        """Zpracování aktivit do DataFrame"""
        if not activities:
            return pd.DataFrame()
        
        processed = []
        
        print(f"\n📝 Zpracovávám {len(activities)} aktivit...")
        
        for idx, act in enumerate(activities):
            try:
                # Debug: ukážeme první aktivitu
                if idx == 0:
                    print(f"\n🔍 Debug - První aktivita:")
                    print(f"   Typ: {type(act)}")
                    if isinstance(act, dict):
                        print(f"   Klíče: {list(act.keys())[:10]}")
                
                # Bezpečné získání hodnot
                distance = act.get('distance', 0) if isinstance(act, dict) else 0
                duration = act.get('duration', 0) if isinstance(act, dict) else 0
                
                # Typ aktivity
                act_type = act.get('activityType', {}) if isinstance(act, dict) else {}
                if isinstance(act_type, dict):
                    type_key = act_type.get('typeKey', 'unknown')
                elif isinstance(act_type, str):
                    type_key = act_type
                else:
                    type_key = 'unknown'
                
                # Výpočet tempa
                pace = 0
                if distance and duration and distance > 0:
                    pace = round((duration / (distance / 1000)) / 60, 2)
                
                # Získání data
                start_time = act.get('startTimeLocal', '') if isinstance(act, dict) else ''
                datum = start_time[:10] if len(start_time) >= 10 else ''
                cas = start_time[11:16] if len(start_time) >= 16 else ''
                
                processed.append({
                    'datum': datum,
                    'cas': cas,
                    'nazev': act.get('activityName', 'Bez názvu') if isinstance(act, dict) else 'Bez názvu',
                    'typ': type_key,
                    'vzdalenost_km': round(distance / 1000, 2) if distance else 0,
                    'cas_min': round(duration / 60, 1) if duration else 0,
                    'tempo_min_km': pace,
                    'prumer_tep': act.get('averageHR', 0) if isinstance(act, dict) else 0,
                    'max_tep': act.get('maxHR', 0) if isinstance(act, dict) else 0,
                    'kalorie': act.get('calories', 0) if isinstance(act, dict) else 0,
                    'training_load': act.get('activityTrainingLoad', None) if isinstance(act, dict) else None,
                    'aerobic_te': act.get('aerobicTrainingEffect', None) if isinstance(act, dict) else None,
                    'anaerobic_te': act.get('anaerobicTrainingEffect', None) if isinstance(act, dict) else None,
                    'vo2max': act.get('vO2MaxValue', None) if isinstance(act, dict) else None,
                    'avg_power': act.get('avgPower', None) if isinstance(act, dict) else None,
                    'norm_power': act.get('normPower', None) if isinstance(act, dict) else None,
                    'avg_cadence': act.get('averageRunningCadenceInStepsPerMinute', None) if isinstance(act, dict) else None,
                    'avg_stride_length': act.get('avgStrideLength', None) if isinstance(act, dict) else None,
                    'avg_ground_contact_time': act.get('avgGroundContactTime', None) if isinstance(act, dict) else None,
                    'avg_vertical_oscillation': act.get('avgVerticalOscillation', None) if isinstance(act, dict) else None,
                    'avg_vertical_ratio': act.get('avgVerticalRatio', None) if isinstance(act, dict) else None,
                    'elevation_gain': act.get('elevationGain', None) if isinstance(act, dict) else None,
                    'elevation_loss': act.get('elevationLoss', None) if isinstance(act, dict) else None,
                    'fastest_1km': round(act.get('fastestSplit_1000', 0) / 60, 2) if isinstance(act, dict) and act.get('fastestSplit_1000') else None,
                    'fastest_5km': round(act.get('fastestSplit_5000', 0) / 60, 2) if isinstance(act, dict) and act.get('fastestSplit_5000') else None,
                    'steps': act.get('steps', None) if isinstance(act, dict) else None,
                    'body_battery_diff': act.get('differenceBodyBattery', None) if isinstance(act, dict) else None,
                    'moderate_intensity_min': act.get('moderateIntensityMinutes', None) if isinstance(act, dict) else None,
                    'vigorous_intensity_min': act.get('vigorousIntensityMinutes', None) if isinstance(act, dict) else None,
                    'training_effect_label': act.get('trainingEffectLabel', None) if isinstance(act, dict) else None,
                    'water_estimated_ml': act.get('waterEstimated', None) if isinstance(act, dict) else None,
                })
                
            except Exception as e:
                print(f"   ⚠️ Chyba u aktivity {idx}: {e}")
                continue
        
        print(f"✅ Zpracováno {len(processed)} aktivit")
        
        return pd.DataFrame(processed) if processed else pd.DataFrame()
    
    def save_data(self, activities):
        """Uložení dat"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # JSON - surová data
        json_file = self.data_folder / f"activities_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(activities, f, indent=2, ensure_ascii=False)
        print(f"\n💾 JSON: {json_file.name}")
        
        # DataFrame a CSV
        df = self.process_activities(activities)
        
        if not df.empty:
            csv_file = self.data_folder / f"activities_{timestamp}.csv"
            df.to_csv(csv_file, index=False, encoding='utf-8')
            print(f"📊 CSV: {csv_file.name}")
            
            return df
        
        return None

def main():
    print("=" * 60)
    print("🏃 GARMIN SYNC - FIXED VERZE")
    print("=" * 60)
    print()
    
    sync = GarminSync()
    
    if not sync.email or sync.email == "tvuj_email@example.com":
        print("❌ Nastav email v .env!")
        return
    
    print(f"📧 Email: {sync.email}\n")
    
    # Přihlášení
    if not sync.login():
        return
    
    # Stažení aktivit
    activities = sync.get_activities()
    
    if not activities:
        print("\n💡 Tip: Zkontroluj složku garmin_data/")
        return
    
    # Uložení
    df = sync.save_data(activities)
    
    if df is not None and not df.empty:
        print("\n" + "=" * 60)
        print("📈 STATISTIKY")
        print("=" * 60)
        
        print(f"\n📊 Celkem aktivit: {len(df)}")
        print(f"🏃 Celková vzdálenost: {df['vzdalenost_km'].sum():.2f} km")
        print(f"⏱️  Celkový čas: {df['cas_min'].sum()/60:.1f} hod")
        
        if df[df['tempo_min_km'] > 0]['tempo_min_km'].mean() > 0:
            avg_pace = df[df['tempo_min_km'] > 0]['tempo_min_km'].mean()
            print(f"⚡ Průměrné tempo: {avg_pace:.2f} min/km")
        
        if df['prumer_tep'].mean() > 0:
            print(f"❤️  Průměrný tep: {df['prumer_tep'].mean():.0f} bpm")
        
        print("\n📋 Podle typu aktivity:")
        type_stats = df.groupby('typ').agg({
            'vzdalenost_km': 'sum',
            'cas_min': 'sum'
        }).round(2)
        
        for typ, data in type_stats.iterrows():
            if data['vzdalenost_km'] > 0:
                print(f"   {typ}: {data['vzdalenost_km']:.2f} km, {data['cas_min']:.0f} min")
        
        print("\n📅 Poslední aktivity:")
        for idx, row in df.head(10).iterrows():
            if row['vzdalenost_km'] > 0:
                print(f"   {row['datum']} {row['cas']} - {row['nazev']}: {row['vzdalenost_km']:.2f} km")
    
    print("\n" + "=" * 60)
    print("✅ HOTOVO!")
    print("=" * 60)
    print(f"\n💾 Všechna data jsou ve složce: garmin_data/")

if __name__ == "__main__":
    main()
