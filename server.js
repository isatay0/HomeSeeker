import express from "express";
import fetch from "node-fetch";
import cors from "cors";
import dotenv from "dotenv";

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json({ limit: '30mb' }));

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;

const systemPrompt = `
You are HomeSeeker, a professional scientist in the field of 
astronomy and astrophysics who is passionate about exoplanets.
You help anyone who is interested and has questions about exoplanets and their analysis.`;

app.post("/chat", async (req, res) => {
  const userMessage = req.body.message;

  if (!OPENAI_API_KEY) {
    return res.status(500).json({ reply: "OpenAI API key not configured" });
  }

  try {
    const response = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${OPENAI_API_KEY}`
      },
      body: JSON.stringify({
        model: "gpt-4o-mini",
        messages: [
          { role: "system", content: systemPrompt },
          { role: "user", content: userMessage }
        ]
      })
    });

    if (!response.ok) {
      const errText = await response.text();
      console.error("OpenAI API error:", errText);
      return res.status(500).json({ reply: "OpenAI API error" });
    }

    const data = await response.json();
    const reply = data.choices[0].message.content;
    res.json({ reply });
  } catch (err) {
    console.error("Server error:", err);
    res.status(500).json({ reply: "Error connecting to AI." });
  }
});

app.post('/analyze_images', async (req, res) => {
  try {
    const { target = 'Unknown', images } = req.body || {};
    
    if (!images || !images.periodogram || !images.folded_lightcurve) {
      return res.status(400).json({ 
        error: 'Require images.periodogram and images.folded_lightcurve' 
      });
    }

    if (!OPENAI_API_KEY) {
      return res.status(500).json({ 
        error: 'OpenAI API key not configured. Set OPENAI_API_KEY in .env file' 
      });
    }

    console.log(`Analyzing images for target: ${target}`);

    let periodogramB64 = images.periodogram.replace(/^data:image\/\w+;base64,/, '');
    let foldedB64 = images.folded_lightcurve.replace(/^data:image\/\w+;base64,/, '');

    console.log('Periodogram base64 length:', periodogramB64.length);
    console.log('Folded base64 length:', foldedB64.length);

    if (periodogramB64.length < 100 || foldedB64.length < 100) {
      return res.status(400).json({ 
        error: 'Images appear to be invalid or too small' 
      });
    }

    const analysisPrompt = `You are analyzing exoplanet lightcurve data for ${target}. Two images are provided:

IMAGE 1: Periodogram showing Power vs Period (days)
IMAGE 2: Folded lightcurve showing Normalized Flux vs Phase

YOUR TASK:
1. Read the period value (in days) from the highest peak in the periodogram
2. Measure the transit depth (dip in flux) from the folded lightcurve
3. Calculate planetary parameters from these measurements

Use these standard formulas:
- Planet Radius = sqrt(transit_depth) × stellar_radius (assume 1 R_sun)
- Semi-Major Axis ≈ (period_days²)^(1/3) × 0.1 AU
- Temperature ≈ 5800K × sqrt(stellar_radius / (2 × semi_major_axis))

Respond with ONLY this format (no markdown, no asterisks):

Planet Radius (Earth radii): [number]
Planet Mass (Earth masses): [number]  
Planet Density (g/cm³): [number]
Equilibrium Temperature (K): [number]
Semi-Major Axis (AU): [number]
Orbital Eccentricity: [number]
Orbital Period (days): [number]
Stellar Temperature (K): [number]
Stellar Mass (Solar masses): [number]
Stellar Radius (Solar radii): [number]
Stellar Luminosity (Solar luminosities): [number]

Analysis: [Brief description of planet type and characteristics]

Confidence: [High/Medium/Low]`;

    const response = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${OPENAI_API_KEY}`,
      },
      body: JSON.stringify({
        model: 'gpt-4o-2024-08-06',
        messages: [
          {
            role: 'user',
            content: [
              {
                type: 'text',
                text: analysisPrompt
              },
              {
                type: 'image_url',
                image_url: {
                  url: `data:image/png;base64,${periodogramB64}`,
                  detail: 'high'
                }
              },
              {
                type: 'image_url',
                image_url: {
                  url: `data:image/png;base64,${foldedB64}`,
                  detail: 'high'
                }
              }
            ]
          }
        ],
        max_tokens: 1000,
        temperature: 0.0
      })
    });

    if (!response.ok) {
      const errText = await response.text();
      console.error('OpenAI Vision API error:', errText);
      return res.status(500).json({ 
        error: 'OpenAI Vision API error', 
        details: errText 
      });
    }

    const data = await response.json();
    let reply = data?.choices?.[0]?.message?.content || 'No analysis generated';

    console.log('Raw reply from GPT-4o:', reply);
    console.log('Reply length:', reply.length);

    reply = reply
      .replace(/\*+/g, '')
      .replace(/#+\s*/g, '')
      .replace(/`+/g, '')
      .replace(/^\s*[-•]\s*/gm, '')
      .replace(/^\s*\d+\.\s*/gm, '')
      .replace(/_+/g, '')            // Remove underscores
      .replace(/~+/g, '')            // Remove strikethrough
      .replace(/\[|\]/g, '')         // Remove brackets
      .split('\n')
      .map(line => line.trim())      // Trim each line
      .filter(line => line.length > 0) // Remove empty lines
      .join('\n')
      .trim();

    console.log('Cleaned reply:', reply);
    console.log('Cleaned length:', reply.length);
    console.log('Analysis completed successfully');
    
    return res.json({ reply });

  } catch (err) {
    console.error('Server error in /analyze_images:', err);
    return res.status(500).json({ 
      error: 'Server error', 
      details: err.message 
    });
  }
});

app.get('/health', (req, res) => {
  res.json({ 
    status: 'healthy', 
    service: 'HomeSeeker API Server',
    hasApiKey: !!OPENAI_API_KEY
  });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`✅ HomeSeeker Server running on http://localhost:${PORT}`);
  console.log(`✅ API Key configured: ${OPENAI_API_KEY ? 'Yes' : 'No'}`);
});