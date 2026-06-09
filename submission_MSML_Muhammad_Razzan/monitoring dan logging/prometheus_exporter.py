import time
import requests
import pandas as pd
from prometheus_client import start_http_server, Counter, Histogram

REQUEST_SUCCESS = Counter('prediction_success_total', 'Total prediksi yang sukses')
REQUEST_ERROR = Counter('prediction_error_total', 'Total prediksi yang gagal/error')

LATENCY_HISTOGRAM = Histogram('inference_latency_seconds', 'Distribusi waktu respons inferensi (detik)')

def run_inference_simulator():
    try:
        df = pd.read_csv("loan_data_clean.csv")
        
        if 'Loan_Status' in df.columns:
            df = df.drop(columns=['Loan_Status'])
        
        sample_data = df.head(1)
        payload = {"dataframe_split": sample_data.to_dict(orient="split")}
        print("Payload nyata berhasil dibuat dari dataset.")
    except Exception as e:
        print(f"Gagal memuat dataset: {e}")
        return

    print("Memulai pengiriman request nyata ke http://127.0.0.1:5002/invocations ...")
    
    while True:
        start_time = time.time()
        try:
            response = requests.post(
                "http://127.0.0.1:5002/invocations", 
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            latency = time.time() - start_time
            LATENCY_HISTOGRAM.observe(latency)
            
            if response.status_code == 200:
                REQUEST_SUCCESS.inc()
                print(f"[SUKSES] Latency: {latency:.4f}s | Prediksi: {response.json()}")
            else:
                REQUEST_ERROR.inc()
                print(f"[ERROR] HTTP {response.status_code} | Pesan: {response.text}")
                
        except requests.exceptions.RequestException as e:
            REQUEST_ERROR.inc()
            print(f"[GAGAL KONEKSI] Pastikan mlflow serve berjalan di port 5002! Error: {e}")
        time.sleep(5)

if __name__ == '__main__':
    start_http_server(8000)
    print("Prometheus metrics server berjalan di http://localhost:8000")
    run_inference_simulator()