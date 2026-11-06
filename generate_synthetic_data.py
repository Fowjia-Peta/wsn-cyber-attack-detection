"""Creates a small SYNTHETIC wireless-sensor-network dataset so the project runs end to end.
For real results, replace data/wsn_data.csv with a real dataset (e.g. WSN-DS) and keep the last column named 'label'."""
import numpy as np, pandas as pd, os

rng = np.random.default_rng(42)
N = 4000

def make(n, attack):
    if not attack:
        return pd.DataFrame({
            "packet_size": rng.normal(200, 50, n).clip(40, 1500),
            "packet_rate": rng.normal(20, 5, n).clip(1, None),
            "avg_delay_ms": rng.normal(30, 8, n).clip(1, None),
            "residual_energy": rng.uniform(0.4, 1.0, n),
            "hop_count": rng.integers(1, 6, n),
            "rssi": rng.normal(-60, 6, n),
            "label": "normal"})
    return pd.DataFrame({
        "packet_size": rng.normal(320, 120, n).clip(40, 1500),
        "packet_rate": rng.normal(55, 20, n).clip(1, None),
        "avg_delay_ms": rng.normal(70, 25, n).clip(1, None),
        "residual_energy": rng.uniform(0.05, 0.7, n),
        "hop_count": rng.integers(1, 9, n),
        "rssi": rng.normal(-72, 10, n),
        "label": "attack"})

df = pd.concat([make(int(N*0.7), False), make(int(N*0.3), True)]).sample(frac=1, random_state=1)
os.makedirs("data", exist_ok=True)
df.to_csv("data/wsn_data.csv", index=False)
print("Saved data/wsn_data.csv", df.shape)
