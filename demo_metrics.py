"""
Demo script to show metric output format
"""
import numpy as np
from config import Config

def demo_metrics_output():
    """Demonstrate what the metric output will look like"""
    
    config = Config()
    
    print("\n" + "="*80)
    print("EVALUATION METRICS (DENORMALIZED WITH UNITS)")
    print("="*80)
    
    # Simulated metrics (these would come from actual model evaluation)
    print("\n" + "-"*80)
    print("PER-FEATURE PERFORMANCE:")
    print("-"*80)
    
    # Example metrics for each feature
    example_metrics = {
        'temperature': {'MAE': 2.45, 'RMSE': 3.12, 'R2': 0.89, 'MAPE': 18.3, 'unit': '°C'},
        'windspeed': {'MAE': 3.21, 'RMSE': 4.67, 'R2': 0.76, 'MAPE': 24.1, 'unit': 'knots'},
        'winddirection': {'MAE': 28.45, 'RMSE': 42.31, 'R2': 0.65, 'unit': 'degrees', 'is_circular': True},
        'visibility': {'MAE': 1.23, 'RMSE': 1.89, 'R2': 0.82, 'MAPE': 15.7, 'unit': 'SM'},
        'pressure': {'MAE': 5.67, 'RMSE': 8.23, 'R2': 0.94, 'MAPE': 0.58, 'unit': 'hPa'}
    }
    
    for feature_name in config.FEATURE_COLS:
        if feature_name in example_metrics:
            m = example_metrics[feature_name]
            unit = m['unit']
            is_circular = m.get('is_circular', False)
            
            print(f"\n{feature_name.upper()} [{unit}]:")
            if is_circular:
                print(f"  MAE (Circular):  {m['MAE']:.4f} {unit}")
                print(f"  RMSE (Circular): {m['RMSE']:.4f} {unit}")
            else:
                print(f"  MAE:   {m['MAE']:.4f} {unit}")
                print(f"  RMSE:  {m['RMSE']:.4f} {unit}")
                if 'MAPE' in m:
                    print(f"  MAPE:  {m['MAPE']:.2f}%")
            print(f"  R²:    {m['R2']:.4f}")
    
    # Overall metrics
    print("\n" + "-"*80)
    print("OVERALL (averaged across all features):")
    print("-"*80)
    print(f"  Average MAE:   {8.20:.4f}")
    print(f"  Average RMSE:  {12.04:.4f}")
    print(f"  Average R²:    {0.81:.4f}")
    
    print("\n" + "="*80)
    
    print("\nNOTE: These are example metrics. To see real metrics from your trained model:")
    print("  1. The model has been trained and saved to: checkpoints/best_model.pt")
    print("  2. Run a full training with: python main_train.py --split random --epochs 50")
    print("  3. The evaluator will automatically print metrics in this format after each epoch")
    print("  4. All metrics are in the proper units as requested:")
    print("     - Temperature: °C")
    print("     - Wind Speed: knots")
    print("     - Wind Direction: degrees (circular)")
    print("     - Visibility: SM (Statute Miles)")
    print("     - Pressure: hPa")

if __name__ == "__main__":
    demo_metrics_output()
