import io
import base64
import joblib
import numpy as np
import pandas as pd
import lightkurve as lk
import matplotlib
import matplotlib.pyplot as plt
import requests
import os
from flask import Flask, request, jsonify
from flask_cors import CORS

matplotlib.use('Agg')

model = joblib.load("planet_habitability_pipeline.joblib")

app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])
        
        prediction = model.predict(df)[0]
        print(prediction)
        probability = model.predict_proba(df)[0].tolist()

        return jsonify({
            "prediction": "Habitable" if prediction == 1 else "Not Habitable",
            "probability": probability
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def save_plot_to_base64(plot_func, *args, **kwargs):
    buf = io.BytesIO()
    plot_func(*args, **kwargs)
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close()
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    return encoded


@app.route("/lightcurve", methods=["POST"])
def lightcurve():
    try:
        data = request.get_json()
        target = data.get("target_star")

        if not target:
            return jsonify({"error": "No target star provided"}), 400

        search = lk.search_targetpixelfile(target, quarter=16)
        
        if len(search) == 0:
            return jsonify({"error": f"No data found for {target}"}), 404

        pixel_file = search.download()
        lc = pixel_file.to_lightcurve(aperture_mask="all")

        plots = {}

        plots["raw_lightcurve"] = save_plot_to_base64(lc.plot)

        flat_lc = lc.flatten(window_length=401)
        plots["flattened_lightcurve"] = save_plot_to_base64(flat_lc.plot)

        periodogram = flat_lc.to_periodogram(minimum_period=1, maximum_period=5)
        plots["periodogram"] = save_plot_to_base64(periodogram.plot)

        period = periodogram.period_at_max_power.value
        print(f"Detected orbital period: {period:.3f} days")

        folded_lc = flat_lc.fold(period=period)
        plots["folded_lightcurve"] = save_plot_to_base64(folded_lc.plot)

        binned_lc = folded_lc.bin(time_bin_size=0.01)
        plots["binned_lightcurve"] = save_plot_to_base64(binned_lc.plot)

        buf = io.BytesIO()
        pixel_file.plot(frame=42)
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        plt.close()
        buf.seek(0)
        plots["pixel_frame"] = base64.b64encode(buf.read()).decode("utf-8")

        
        flux = folded_lc.flux.value
        phase = folded_lc.time.value
        
        baseline_flux = np.median(flux[np.abs(phase) > 0.3])
        min_flux = np.min(flux)
        transit_depth = np.nan
        if not np.isnan(baseline_flux) and not np.isnan(min_flux) and baseline_flux > 0:
            transit_depth = (baseline_flux - min_flux) / baseline_flux
        
        transit_duration_hours = np.nan
        if not np.isnan(transit_depth) and not np.isnan(baseline_flux) and not np.isnan(period):
            transit_mask = flux < (baseline_flux - transit_depth * baseline_flux / 2)
            if np.any(transit_mask):
                transit_phases = phase[transit_mask]
                transit_duration_phase = np.max(transit_phases) - np.min(transit_phases)
                if not np.isnan(transit_duration_phase) and transit_duration_phase > 0:
                    transit_duration_hours = transit_duration_phase * period * 24
            else:
                if period > 0:
                    transit_duration_hours = 0.1 * period * 24
        
        planet_radius_rearth = np.nan
        if transit_depth > 0 and not np.isnan(transit_depth):
            planet_radius_rearth = np.sqrt(transit_depth) * 109.1
        
        planet_mass_mearth = np.nan
        if not np.isnan(planet_radius_rearth) and planet_radius_rearth > 0:
            if planet_radius_rearth < 4:
                planet_mass_mearth = planet_radius_rearth ** 2.06
            else:
                planet_mass_mearth = planet_radius_rearth ** 1.3
        
        planet_density_gcc = np.nan
        if (not np.isnan(planet_mass_mearth) and not np.isnan(planet_radius_rearth) and 
            planet_mass_mearth > 0 and planet_radius_rearth > 0):
            planet_volume_earth = planet_radius_rearth ** 3
            if planet_volume_earth > 0:
                planet_density_gcc = planet_mass_mearth / planet_volume_earth * 5.51
        
        orbital_eccentricity = 0.0
        
        parameters = {}
        
        if not np.isnan(planet_radius_rearth) and planet_radius_rearth > 0:
            parameters["planet_radius"] = round(planet_radius_rearth, 2)
        
        if not np.isnan(planet_mass_mearth) and planet_mass_mearth > 0:
            parameters["planet_mass"] = round(planet_mass_mearth, 1)
        
        if not np.isnan(planet_density_gcc) and planet_density_gcc > 0:
            parameters["planet_density"] = round(planet_density_gcc, 2)
        
        if not np.isnan(period) and period > 0:
            parameters["orbital_period"] = round(period, 3)
        
        if not np.isnan(transit_depth) and transit_depth >= 0:
            parameters["transit_depth"] = round(transit_depth * 100, 3)
        
        if not np.isnan(transit_duration_hours) and transit_duration_hours > 0:
            parameters["transit_duration_hours"] = round(transit_duration_hours, 2)
        
        print(f"Extracted parameters: {parameters}")

        return jsonify({
            "plots": plots,
            "parameters": parameters
        })

    except Exception as e:
        print(f"Error in lightcurve analysis: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/explain_plots", methods=["POST"])
def explain_plots():
    try:
        data = request.get_json()
        periodogram_b64 = data.get("periodogram")
        folded_lightcurve_b64 = data.get("folded_lightcurve")
        target_star = data.get("target_star", "Unknown")
        
        if not periodogram_b64 or not folded_lightcurve_b64:
            return jsonify({"error": "Both periodogram and folded lightcurve images are required"}), 400
        
        import requests
        import os
        
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            return jsonify({"error": "OpenAI API key not configured"}), 500
        
        analysis_prompt = f"""You are analyzing exoplanet lightcurve data for {target_star}. Two scientific plots are provided:

PLOT 1: PERIODOGRAM
- This shows the power spectrum of the lightcurve
- The x-axis is period (in days) - how long it takes for the planet to orbit
- The y-axis is power - how strong the signal is at each period
- Look for the highest peak - this indicates the most likely orbital period
- Explain what the peak means in simple terms

PLOT 2: FOLDED LIGHTCURVE  
- This shows the lightcurve folded at the detected period
- The x-axis is phase (0 to 1) - one complete orbit
- The y-axis is normalized flux - the star's brightness
- Look for the dip in brightness - this is the planet transiting in front of the star
- The depth of the dip tells us about the planet's size
- The width of the dip tells us about the transit duration

Please explain both plots in simple, beginner-friendly language. Make assumptions about what we can learn from these plots and what they tell us about the potential exoplanet(like period,planet radius, maybe temperature). Keep your explanation educational and accessible to someone new to astronomy. Return without any * signs and headers just plain text with divided lines and nothing else."""

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key}"
        }
        
        payload = {
            "model": "gpt-4o-2024-08-06",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": analysis_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{periodogram_b64}",
                                "detail": "high"
                            }
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{folded_lightcurve_b64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 1500,
            "temperature": 0.3
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            return jsonify({"error": f"OpenAI API error: {response.text}"}), 500
        
        result = response.json()
        explanation = result["choices"][0]["message"]["content"]
        
        return jsonify({
            "explanation": explanation,
            "target_star": target_star
        })
        
    except Exception as e:
        print(f"Error in plot explanation: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "service": "HomeSeeker API"})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)