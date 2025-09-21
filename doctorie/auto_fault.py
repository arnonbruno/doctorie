import pandas as pd
import numpy as np
import os
import glob
import warnings
import re
from typing import Tuple, Dict, List, Any, Optional
from dataclasses import dataclass
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
import joblib

warnings.filterwarnings('ignore')

@dataclass
class FaultDetectionConfig:
    """Configuration for the fault detection system."""
    context_max_rpm: float = 1100
    context_max_load: float = 35.0
    base_features: Tuple[str, ...] = (
        'coolant_temperature', 'calculated_load', 'engine_rpm', 'altitude',
        'oxygen_sensor_bank1_sensor1', 'short_term_fuel_trim_bank1_sensor1',
        'long_term_fuel_trim_bank1', 'map_sensor_pressure', 'battery_voltage'
    )
    stft_fault_range: Tuple[float, float] = (-15.0, -10.0)
    oxygen_rich_range: Tuple[float, float] = (0.8, 0.9)
    low_battery_threshold: float = 11.5
    random_state: int = 42
    n_estimators: int = 200

class DomainFeatureEngineer(BaseEstimator, TransformerMixin):
    """Generates interaction features based on automotive domain knowledge."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            return X
        X_transformed = X.copy()
        if 'map_sensor_pressure' in X_transformed.columns and 'engine_rpm' in X_transformed.columns:
            X_transformed['map_rpm_ratio'] = X_transformed['map_sensor_pressure'] / (X_transformed['engine_rpm'] + 1e-6)
        stft = 'short_term_fuel_trim_bank1_sensor1'
        ltft = 'long_term_fuel_trim_bank1'
        if stft in X_transformed.columns and ltft in X_transformed.columns:
            X_transformed['fuel_trim_diff'] = X_transformed[stft] - X_transformed[ltft]
        return X_transformed

class _PipelineDFWrapper(BaseEstimator, TransformerMixin):
    """Wraps a DataFrame-based transformer for compatibility with NumPy-based pipeline steps."""
    def __init__(self, transformer, feature_names):
        self.transformer = transformer
        self.feature_names = feature_names
        self.feature_names_out_ = None
    def fit(self, X, y=None):
        X_df = pd.DataFrame(X, columns=self.feature_names) if isinstance(X, np.ndarray) else X
        self.transformer.fit(X_df, y)
        X_transformed = self.transformer.transform(X_df.head(1))
        self.feature_names_out_ = X_transformed.columns.tolist()
        return self
    def transform(self, X):
        X_df = pd.DataFrame(X, columns=self.feature_names) if isinstance(X, np.ndarray) else X
        X_transformed = self.transformer.transform(X_df)
        return X_transformed.values
    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_

class AutomotiveFaultDetector:
    """Main class for automotive fault detection using Isolation Forest."""
    def __init__(self, config: FaultDetectionConfig = None):
        self.config = config or FaultDetectionConfig()
        self.pipeline: Optional[Pipeline] = None
        self.column_mapping = self._get_column_mapping()
        self.base_features: List[str] = list(self.config.base_features)
        self.final_features: Optional[List[str]] = None
        self.optimized_threshold: Optional[float] = None

    def _get_column_mapping(self) -> Dict[str, str]:
        """Map Portuguese column names to English."""
        return {
            'Temperatura do líquido de arrefecimento do motor - CTS': 'coolant_temperature',
            'Carga calculada do motor': 'calculated_load',
            'Rotação do motor - RPM': 'engine_rpm',
            'Altitude': 'altitude',
            'Sonda lambda - Banco 1, sensor 1': 'oxygen_sensor_bank1_sensor1',
            'Ajuste de combustível de curto prazo - Banco 1, sensor 1': 'short_term_fuel_trim_bank1_sensor1',
            'Pressão no coletor de admissão - MAP': 'map_sensor_pressure',
            'Tensão do módulo': 'battery_voltage',
            'Nº de falhas na memória': 'trouble_codes_count',
            'Ajuste de combustível de longo prazo - Banco 1': 'long_term_fuel_trim_bank1',
            'Estado do sistema de combustível': 'fuel_system_status',
            'Temperatura do ar ambiente': 'ambient_air_temperature',
            'Temperatura do ar admitido - ACT': 'intake_air_temperature',
            'Ajuste de combustível de curto prazo - Banco 1': 'short_term_fuel_trim_bank1',
            'Sonda lambda - Banco 1, sensor 2': 'oxygen_sensor_bank1_sensor2',
            'Pressão barométrica': 'barometric_pressure',
            'Carga absoluta do motor': 'absolute_engine_load',
            'Posição relativa da borboleta - TPS': 'relative_throttle_position',
            'Posição absoluta da borboleta - TPS': 'absolute_throttle_position_tps',
            'Posição absoluta da borboleta - Sensor B': 'absolute_throttle_position_sensor_b'
        }

    def load_and_process_data(self, data_path: str) -> pd.DataFrame:
        """Load, clean, validate, and apply contextual filtering."""
        print("Starting data processing...")
        df = self._load_data(data_path)
        df_cleaned = self._clean_data(df)
        df_valid = self._filter_valid_data(df_cleaned)
        df_contextual = self._apply_context_filter(df_valid)
        return df_contextual

    def _load_data(self, data_path: str) -> pd.DataFrame:
        """Load data from CSV files."""
        reference_files = glob.glob(os.path.join(data_path, "*REFERENCIA*.csv"))
        fault_file = os.path.join(data_path, "fault_example.csv")
        all_data = []
        for file in reference_files:
            try:
                df = pd.read_csv(file)
                df['is_fault'] = False
                all_data.append(df)
            except Exception as e:
                print(f"Error loading {file}: {e}")
        if os.path.exists(fault_file):
            try:
                fault_df = pd.read_csv(fault_file)
                fault_df['is_fault'] = True
                all_data.append(fault_df)
            except Exception as e:
                print(f"Error loading {fault_file}: {e}")
        if not all_data:
            raise FileNotFoundError(f"No data files found in {data_path}")
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"Loaded {len(combined_df):,} total records.")
        return combined_df

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply cleaning and transformation."""
        print("Cleaning and transforming data...")
        df_clean = df.copy()
        df_clean.rename(columns=self.column_mapping, inplace=True)
        df_clean = self._handle_fuel_system_status(df_clean)
        df_clean = self._clean_numeric_columns_with_units(df_clean)
        return df_clean

    def _filter_valid_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter data based on requirements (no trouble codes for reference)."""
        if 'trouble_codes_count' in df.columns:
            df['trouble_codes_count'] = pd.to_numeric(df['trouble_codes_count'], errors='coerce').fillna(0)
        else:
            df['trouble_codes_count'] = 0
        valid_data = df[
            (df['is_fault'] == True) |
            ((df['is_fault'] == False) & (df['trouble_codes_count'] == 0))
        ].copy()
        print(f"After validation filtering: {len(valid_data):,} records")
        return valid_data

    def _apply_context_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter data to the specific operating context (idle/low load)."""
        print("Applying contextual filter (idle/low load)...")
        df_filtered = df.copy()
        required_cols = ['engine_rpm', 'calculated_load']
        if not all(col in df_filtered.columns for col in required_cols):
            print("Warning: Missing context columns. Cannot apply context filter.")
            return df_filtered
        df_filtered['engine_rpm'] = df_filtered['engine_rpm'].fillna(9999)
        df_filtered['calculated_load'] = df_filtered['calculated_load'].fillna(100)
        mask = (
            (df_filtered['engine_rpm'] <= self.config.context_max_rpm) &
            (df_filtered['calculated_load'] <= self.config.context_max_load)
        )
        df_contextual = df[mask].copy()
        print(f"Data remaining in context: {len(df_contextual):,} records")
        print(f" - Normal samples: {len(df_contextual[df_contextual['is_fault'] == False]):,}")
        print(f" - Fault samples: {len(df_contextual[df_contextual['is_fault'] == True]):,}")
        if len(df_contextual) == 0:
            raise ValueError("Context filter is too restrictive. No data remains.")
        return df_contextual

    def _handle_fuel_system_status(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encodes the fuel system status column."""
        col = 'fuel_system_status'
        if col in df.columns:
            status_series = df[col].astype(str).str.lower().str.strip()
            encoded_col_name = col + '_encoded'
            df[encoded_col_name] = np.nan
            df.loc[status_series.str.contains('aberta'), encoded_col_name] = 0
            df.loc[status_series.str.contains('fechada'), encoded_col_name] = 1
            df = df.drop(columns=[col])
        return df

    def _clean_numeric_columns_with_units(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean numeric columns by removing unit suffixes and noise."""
        exclude_columns = ['source_file', 'is_fault', 'fuel_system_status_encoded']
        columns_to_clean = [col for col in df.columns if col not in exclude_columns]
        cleaning_pattern = r'[^0-9.\-+]'
        for col in columns_to_clean:
            if col in df.columns:
                series = df[col].astype(str)
                series = series.str.replace(cleaning_pattern, '', regex=True)
                series = series.str.replace(r'\.(?=.*\.)', '', regex=True)
                series = series.replace(['', 'nan', 'NaN', '.', '+', '-'], np.nan)
                df[col] = pd.to_numeric(series, errors='coerce')
        return df

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare the base feature set for modeling."""
        print("Preparing base feature set...")
        available_features = []
        missing_features = []
        for feature in self.base_features:
            if feature in df.columns:
                available_features.append(feature)
            else:
                missing_features.append(feature)
        if missing_features:
            print(f"Warning: The following base features are missing: {missing_features}")
        self.base_features = available_features
        if not self.base_features:
            raise ValueError("No features available for modeling.")
        X = df[self.base_features]
        y = df['is_fault']
        print(f"Using {len(self.base_features)} base features.")
        return X, y

    def train_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the fault detection model using the enhanced pipeline."""
        print("Training anomaly detection model...")
        X_train = X[y == False].copy()
        if len(X_train) == 0:
            raise ValueError("No normal data available for training within the defined context.")
        self.pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('feature_engineer', _PipelineDFWrapper(DomainFeatureEngineer(), self.base_features)),
            ('scaler', RobustScaler()),
            ('model', IsolationForest(
                contamination='auto',
                random_state=self.config.random_state,
                n_estimators=self.config.n_estimators,
                n_jobs=-1
            ))
        ])
        self.pipeline.fit(X_train)
        self.final_features = self.pipeline.named_steps['feature_engineer'].get_feature_names_out()
        print(f"Model training completed on {len(X_train)} samples.")
        print(f"Total features after engineering: {len(self.final_features)}")
        self.optimize_threshold(X, y)

    def optimize_threshold(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Optimize the decision threshold by maximizing F1-score."""
        print("Optimizing decision threshold...")
        scores = -self.pipeline.decision_function(X)
        y_true = y.astype(int)
        if y_true.nunique() < 2:
            print("Optimization skipped: Only one class present. Using default threshold (0.0).")
            self.optimized_threshold = 0.0
            return
        precision, recall, thresholds = precision_recall_curve(y_true, scores)
        denominator = precision + recall
        f1_scores = np.where(denominator == 0, 0, (2 * precision * recall) / denominator)
        best_index = np.argmax(f1_scores[:-1])
        self.optimized_threshold = thresholds[best_index]
        p_best = precision[best_index]
        r_best = recall[best_index]
        f1_best = f1_scores[best_index]
        print(f"Optimization results:")
        print(f"Best F1-score: {f1_best:.4f}")
        print(f"Recall: {r_best:.4f}")
        print(f"Precision: {p_best:.4f}")
        print(f"Optimized threshold: {self.optimized_threshold:.4f}")

    def detect_faults(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Detect faults using the trained pipeline and optimized threshold."""
        if self.pipeline is None or self.optimized_threshold is None:
            raise ValueError("Model not trained or threshold not optimized.")
        scores = -self.pipeline.decision_function(X)
        fault_predictions = (scores >= self.optimized_threshold).astype(int)
        return fault_predictions, scores

    def evaluate_model(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Evaluate the model performance."""
        print("Evaluating model performance...")
        fault_predictions, scores = self.detect_faults(X)
        y_true = y.astype(int)
        if y_true.nunique() < 2:
            print("Evaluation skipped: Only one class present.")
            return {}
        print("Classification report:")
        print(classification_report(y_true, fault_predictions, target_names=['Normal', 'Fault'], zero_division=0))
        cm = confusion_matrix(y_true, fault_predictions)
        print("Confusion matrix:")
        print(f" Predicted")
        print(f"Actual Normal Fault")
        try:
            print(f"Normal {cm[0,0]:6d} {cm[0,1]:5d}")
            print(f"Fault {cm[1,0]:6d} {cm[1,1]:5d}")
        except IndexError:
            print(cm)
        precision = precision_score(y_true, fault_predictions, zero_division=0)
        recall = recall_score(y_true, fault_predictions, zero_division=0)
        f1 = f1_score(y_true, fault_predictions, zero_division=0)
        print(f"Metrics summary:")
        print(f"Precision: {precision:.3f} | Recall: {recall:.3f} | F1-score: {f1:.3f}")
        return {'precision': precision, 'recall': recall, 'f1_score': f1}

    def save_model(self, filepath: str) -> None:
        """Save the trained model pipeline and configuration."""
        model_data = {
            'pipeline': self.pipeline,
            'config': self.config,
            'column_mapping': self.column_mapping,
            'base_features': self.base_features,
            'final_features': self.final_features,
            'optimized_threshold': self.optimized_threshold
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to: {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load a trained model pipeline and configuration."""
        model_data = joblib.load(filepath)
        self.pipeline = model_data['pipeline']
        self.config = model_data['config']
        self.column_mapping = model_data['column_mapping']
        self.base_features = model_data['base_features']
        self.final_features = model_data['final_features']
        self.optimized_threshold = model_data['optimized_threshold']
        print(f"Model loaded from: {filepath}")

class FaultDetectionInterface:
    """Interface for fault detection predictions using the trained pipeline."""
    def __init__(self, detector: AutomotiveFaultDetector):
        self.detector = detector
        if self.detector.pipeline is None:
            raise ValueError("Detector model is not trained.")
    def predict_single_scan(self, scan_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict fault for a single scan input."""
        print("Analyzing new scan...")
        df = pd.DataFrame([scan_data])
        df_cleaned = self._apply_etl(df)
        if not self._is_in_context(df_cleaned):
            return {
                'fault_detected': False,
                'status': 'Scan outside modeled context (e.g., high RPM or high load).'
            }
        df_features = self._ensure_features(df_cleaned)
        X = df_features[self.detector.base_features]
        try:
            predictions, scores = self.detector.detect_faults(X)
            fault_detected = bool(predictions[0])
            result = {
                'fault_detected': fault_detected,
                'anomaly_score': float(scores[0]),
                'threshold': self.detector.optimized_threshold,
                'status': 'Fault detected' if fault_detected else 'No fault detected'
            }
            if fault_detected:
                result['fault_details'] = self._analyze_fault_details(df_cleaned.iloc[0])
            return result
        except Exception as e:
            print(f"Error details: {e}")
            return {
                'fault_detected': None,
                'status': 'Error in prediction.'
            }
    def _apply_etl(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the same ETL pipeline as training."""
        df = df.rename(columns=self.detector.column_mapping)
        df = self.detector._handle_fuel_system_status(df)
        df = self.detector._clean_numeric_columns_with_units(df)
        return df
    def _is_in_context(self, df_cleaned: pd.DataFrame) -> bool:
        """Check if the single scan input fits the modeled operating context."""
        config = self.detector.config
        sample = df_cleaned.iloc[0]
        try:
            rpm = sample.get('engine_rpm')
            load = sample.get('calculated_load')
            if pd.notna(rpm) and rpm > config.context_max_rpm:
                return False
            if pd.notna(load) and load > config.context_max_load:
                return False
            return True
        except Exception as e:
            print(f"Error checking context: {e}. Proceeding with prediction.")
            return True
    def _ensure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all base features required by the model are present."""
        for feature in self.detector.base_features:
            if feature not in df.columns:
                df[feature] = np.nan
        return df
    def _analyze_fault_details(self, sample: pd.Series) -> List[str]:
        """Apply rule-based diagnostics to a sample flagged as faulty."""
        details = []
        config = self.detector.config
        stft_col = 'short_term_fuel_trim_bank1_sensor1'
        if stft_col in sample.index and pd.notna(sample[stft_col]):
            stft = sample[stft_col]
            if config.stft_fault_range[0] <= stft <= config.stft_fault_range[1]:
                details.append(f"Rich air-fuel mixture detected (STFT: {stft:.2f}%)")
        o2_col = 'oxygen_sensor_bank1_sensor1'
        if o2_col in sample.index and pd.notna(sample[o2_col]):
            o2 = sample[o2_col]
            if config.oxygen_rich_range[0] <= o2 <= config.oxygen_rich_range[1]:
                details.append(f"Rich mixture indicated by O2 sensor (Voltage: {o2:.2f}V)")
        battery_col = 'battery_voltage'
        if battery_col in sample.index and pd.notna(sample[battery_col]):
            battery = sample[battery_col]
            if battery < config.low_battery_threshold:
                details.append(f"Low battery voltage detected ({battery:.2f}V)")
        map_col = 'map_sensor_pressure'
        if map_col in sample.index and pd.notna(sample[map_col]):
            map_pressure = sample[map_col]
            details.append(f"MAP sensor reading: {map_pressure:.2f} mbar")
        if not details:
            details.append("Anomaly detected by model; specific diagnosis inconclusive.")
        return details

def main():
    """Main execution script demonstrating the full pipeline."""
    print("Automotive Fault Detection System")
    config = FaultDetectionConfig()
    detector = AutomotiveFaultDetector(config)
    data_path = "./datasets"
    model_path = 'models/automotive_fault_pipeline_final.pkl'
    if not os.path.exists(data_path):
        print(f"Warning: Data directory not found at {os.path.abspath(data_path)}")
    try:
        print("Training and evaluation phase")
        df = detector.load_and_process_data(data_path)
        X, y = detector.prepare_features(df)
        detector.train_model(X, y)
        metrics = detector.evaluate_model(X, y)
        os.makedirs('models', exist_ok=True)
        detector.save_model(model_path)
        print("Demonstration phase")
        test_detector = AutomotiveFaultDetector()
        test_detector.load_model(model_path)
        interface = FaultDetectionInterface(test_detector)
        print("Case 1: Normal operation (idle, warm engine)")
        normal_scan = {
            'Temperatura do líquido de arrefecimento do motor - CTS': '85 °C',
            'Carga calculada do motor': '25 %',
            'Rotação do motor - RPM': '800 rpm',
            'Altitude': '500 m',
            'Sonda lambda - Banco 1, sensor 1': '0.45 V',
            'Ajuste de combustível de curto prazo - Banco 1, sensor 1': '2 %',
            'Ajuste de combustível de longo prazo - Banco 1': '1 %',
            'Pressão no coletor de admissão - MAP': '350 mbar',
            'Tensão do módulo': '13.8 V'
        }
        result1 = interface.predict_single_scan(normal_scan)
        print(f"Result: {result1['status']}")
        if 'anomaly_score' in result1:
            print(f"Score: {result1['anomaly_score']:.4f} (Threshold: {result1['threshold']:.4f})")
        print("Case 2: Fault condition (rich mixture, low battery)")
        fault_scan = {
            'Temperatura do líquido de arrefecimento do motor - CTS': '50 °C',
            'Carga calculada do motor': '25 %',
            'Rotação do motor - RPM': '900 rpm',
            'Altitude': '500 m',
            'Sonda lambda - Banco 1, sensor 1': '0.85 V',
            'Ajuste de combustível de curto prazo - Banco 1, sensor 1': '-12 %',
            'Ajuste de combustível de longo prazo - Banco 1': '-5 %',
            'Pressão no coletor de admissão - MAP': '450 mbar',
            'Tensão do módulo': '11.0 V'
        }
        result2 = interface.predict_single_scan(fault_scan)
        print(f"Result: {result2['status']}")
        if 'anomaly_score' in result2:
            print(f"Score: {result2['anomaly_score']:.4f} (Threshold: {result2['threshold']:.4f})")
        if result2.get('fault_detected') and 'fault_details' in result2:
            print("Diagnosis details:")
            for detail in result2['fault_details']:
                print(f"- {detail}")
        print("Case 3: Outside context (driving)")
        driving_scan = {
            'Carga calculada do motor': '60 %',
            'Rotação do motor - RPM': '3000 rpm'
        }
        result3 = interface.predict_single_scan(driving_scan)
        print(f"Result: {result3['status']}")
        print("Case 4: Missing data (robustness check)")
        missing_data_scan = {
            'Carga calculada do motor': '25 %',
            'Rotação do motor - RPM': '800 rpm'
        }
        result4 = interface.predict_single_scan(missing_data_scan)
        print(f"Result: {result4['status']}")
        if 'anomaly_score' in result4:
            print(f"Score: {result4['anomaly_score']:.4f} (Threshold: {result4['threshold']:.4f})")
        print("System demonstration completed successfully.")
    except FileNotFoundError as e:
        print(f"Execution failed due to missing data: {e}")
    except Exception as e:
        print(f"Critical error during execution: {e}")

if __name__ == "__main__":
    main()