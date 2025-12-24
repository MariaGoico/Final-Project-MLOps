import gradio as gr
import requests

API_URL = "https://mlops-finalproject-latest.onrender.com"

def test_endpoint(endpoint):
    try:
        response = requests.get(f"{API_URL}{endpoint}", timeout=5)
        response.raise_for_status()
        data = response.json()
        return data.get("message", "No message returned")
    except Exception as e:
        return f"Error connecting to API: {str(e)}"

iface = gr.Interface(
    fn=test_endpoint,
    inputs=gr.Dropdown(
        choices=["/", "/predict"],
        value="/",
        label="Endpoint to test",
    ),
    outputs=gr.Textbox(label="API response"),
    title="FastAPI – Render connection test",
    description="Checks whether the Render API endpoints are reachable",
)

if __name__ == "__main__":
    iface.launch()
