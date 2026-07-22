import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# --- MODEL LOADING ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'LogisticRegression.pkl')

try:
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
        
    if hasattr(model, 'feature_names_in_'):
        feature_names = model.feature_names_in_.tolist()
    else:
        feature_names = [
            'Student_ID', 'Age', 'Gender', 'Family_Income', 'Internet_Access', 
            'Study_Hours_per_Day', 'Attendance_Rate', 'Assignment_Delay_Days', 
            'Travel_Time_Minutes', 'Part_Time_Job', 'Scholarship', 'Stress_Index', 
            'GPA', 'Semester_GPA', 'CGPA', 'Semester', 'Department', 'Parental_Education'
        ]
        
    feature_importance = model.coef_[0].tolist() if hasattr(model, 'coef_') else [0] * len(feature_names)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    feature_names = []
    feature_importance = []

# --- HTML / UI TEMPLATE ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Student Retention Analytics | Enterprise AI</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        /* --- THEME SYSTEM --- */
        :root {
            --bg-color: #080c14;
            --surface-glass: rgba(18, 26, 43, 0.65);
            --surface-border: rgba(255, 255, 255, 0.08);
            --surface-hover: rgba(255, 255, 255, 0.12);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent-primary: #3b82f6; 
            --accent-secondary: #06b6d4; 
            --accent-glow: rgba(59, 130, 246, 0.4);
            --danger: #ef4444;
            --danger-glow: rgba(239, 68, 68, 0.3);
            --success: #10b981;
            --success-glow: rgba(16, 185, 129, 0.3);
            --font-family: 'Plus Jakarta Sans', system-ui, sans-serif;
            --bg-gradient: radial-gradient(at 0% 0%, rgba(59, 130, 246, 0.12) 0px, transparent 50%),
                           radial-gradient(at 100% 100%, rgba(6, 182, 212, 0.1) 0px, transparent 50%);
        }

        body.theme-cyberpunk {
            --bg-color: #05030a;
            --surface-glass: rgba(23, 10, 36, 0.7);
            --surface-border: rgba(255, 0, 124, 0.2);
            --accent-primary: #ff007c; 
            --accent-secondary: #00f0ff; 
            --accent-glow: rgba(255, 0, 124, 0.5);
            --danger: #ff3355;
            --danger-glow: rgba(255, 51, 85, 0.4);
            --success: #00ff9d;
            --success-glow: rgba(0, 255, 157, 0.4);
            --bg-gradient: radial-gradient(at 0% 0%, rgba(255, 0, 124, 0.18) 0px, transparent 50%),
                           radial-gradient(at 100% 100%, rgba(0, 240, 255, 0.15) 0px, transparent 50%);
        }

        body.theme-forest {
            --bg-color: #040d08;
            --surface-glass: rgba(12, 31, 20, 0.7);
            --surface-border: rgba(46, 160, 67, 0.2);
            --text-secondary: #94a89a;
            --accent-primary: #10b981; 
            --accent-secondary: #84cc16; 
            --accent-glow: rgba(16, 185, 129, 0.4);
            --danger: #f43f5e;
            --danger-glow: rgba(244, 63, 94, 0.3);
            --success: #22c55e;
            --success-glow: rgba(34, 197, 94, 0.3);
            --bg-gradient: radial-gradient(at 0% 0%, rgba(16, 185, 129, 0.15) 0px, transparent 50%),
                           radial-gradient(at 100% 100%, rgba(132, 204, 22, 0.1) 0px, transparent 50%);
        }

        body.theme-ice {
            --bg-color: #020912;
            --surface-glass: rgba(11, 27, 48, 0.7);
            --surface-border: rgba(56, 189, 248, 0.2);
            --text-secondary: #7dd3fc;
            --accent-primary: #38bdf8; 
            --accent-secondary: #818cf8; 
            --accent-glow: rgba(56, 189, 248, 0.4);
            --danger: #fb7185;
            --danger-glow: rgba(251, 113, 133, 0.3);
            --success: #38bdf8;
            --success-glow: rgba(56, 189, 248, 0.3);
            --bg-gradient: radial-gradient(at 0% 0%, rgba(56, 189, 248, 0.15) 0px, transparent 50%),
                           radial-gradient(at 100% 100%, rgba(129, 140, 248, 0.12) 0px, transparent 50%);
        }

        body.theme-magic {
            --bg-color: #0d0518;
            --surface-glass: rgba(30, 14, 51, 0.7);
            --surface-border: rgba(168, 85, 247, 0.25);
            --text-secondary: #cbd5e1;
            --accent-primary: #a855f7; 
            --accent-secondary: #f43f5e; 
            --accent-glow: rgba(168, 85, 247, 0.4);
            --danger: #f43f5e;
            --danger-glow: rgba(244, 63, 94, 0.3);
            --success: #c084fc;
            --success-glow: rgba(192, 132, 252, 0.3);
            --bg-gradient: radial-gradient(at 0% 0%, rgba(168, 85, 247, 0.18) 0px, transparent 50%),
                           radial-gradient(at 100% 100%, rgba(244, 63, 94, 0.12) 0px, transparent 50%);
        }

        body.theme-solar {
            --bg-color: #120400;
            --surface-glass: rgba(41, 14, 4, 0.7);
            --surface-border: rgba(249, 115, 22, 0.25);
            --text-secondary: #fdba74;
            --accent-primary: #f97316; 
            --accent-secondary: #eab308; 
            --accent-glow: rgba(249, 115, 22, 0.4);
            --danger: #ef4444;
            --danger-glow: rgba(239, 68, 68, 0.3);
            --success: #84cc16;
            --success-glow: rgba(132, 204, 22, 0.3);
            --bg-gradient: radial-gradient(at 0% 0%, rgba(249, 115, 22, 0.18) 0px, transparent 50%),
                           radial-gradient(at 100% 100%, rgba(234, 179, 8, 0.12) 0px, transparent 50%);
        }

        body.theme-void {
            --bg-color: #000000;
            --surface-glass: rgba(18, 18, 18, 0.85);
            --surface-border: rgba(255, 255, 255, 0.15);
            --text-secondary: #a3a3a3;
            --accent-primary: #ffffff; 
            --accent-secondary: #737373; 
            --accent-glow: rgba(255, 255, 255, 0.2);
            --danger: #a3a3a3;
            --danger-glow: rgba(163, 163, 163, 0.2);
            --success: #e5e5e5;
            --success-glow: rgba(229, 229, 229, 0.2);
            --bg-gradient: none;
        }

        body.theme-synthwave {
            --bg-color: #0b0716;
            --surface-glass: rgba(26, 14, 48, 0.75);
            --surface-border: rgba(236, 72, 153, 0.3);
            --text-secondary: #d8b4fe;
            --accent-primary: #ec4899; 
            --accent-secondary: #8b5cf6; 
            --accent-glow: rgba(236, 72, 153, 0.5);
            --danger: #f43f5e;
            --danger-glow: rgba(244, 63, 94, 0.4);
            --success: #06b6d4;
            --success-glow: rgba(6, 182, 212, 0.4);
            --bg-gradient: radial-gradient(at 0% 0%, rgba(236, 72, 153, 0.18) 0px, transparent 50%),
                           radial-gradient(at 100% 100%, rgba(139, 92, 246, 0.15) 0px, transparent 50%);
        }

        /* --- GLOBAL STYLES --- */
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: var(--font-family);
            background-color: var(--bg-color);
            background-image: var(--bg-gradient);
            background-attachment: fixed;
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            overflow-x: hidden;
            transition: background-color 0.5s cubic-bezier(0.4, 0, 0.2, 1), background-image 0.5s ease;
        }

        /* --- HEADER & NAVIGATION --- */
        .top-nav { display: flex; justify-content: space-between; align-items: center; padding: 1.5rem 3rem; }
        .brand { display: flex; align-items: center; gap: 0.75rem; font-weight: 700; font-size: 1.25rem; letter-spacing: -0.03em; }
        .brand-badge { background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary)); padding: 0.25rem 0.6rem; border-radius: 6px; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em; color: white; font-weight: 800; }

        .theme-selector-wrapper { position: relative; }
        .theme-selector { 
            background: rgba(0, 0, 0, 0.4); 
            backdrop-filter: blur(12px);
            border: 1px solid var(--surface-border); 
            color: var(--text-primary); 
            padding: 0.6rem 1.2rem; 
            border-radius: 10px; 
            font-family: var(--font-family); 
            font-size: 0.85rem;
            font-weight: 500;
            outline: none; 
            cursor: pointer; 
            transition: all 0.3s ease; 
        }
        .theme-selector:hover { border-color: var(--accent-primary); box-shadow: 0 0 15px var(--accent-glow); }
        .theme-selector option { background: #080c14; color: #fff; }

        /* --- LAYOUT STRUCTURE --- */
        .container { width: 95%; max-width: 1440px; margin: 0 auto 3rem auto; display: grid; grid-template-columns: 380px 1fr; gap: 2rem; }
        
        .glass-card { 
            background: var(--surface-glass); 
            backdrop-filter: blur(20px); 
            -webkit-backdrop-filter: blur(20px); 
            border: 1px solid var(--surface-border); 
            border-radius: 20px; 
            padding: 2rem; 
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37); 
            transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1); 
            position: relative;
            overflow: hidden;
        }
        .glass-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; height: 1px;
            background: linear-gradient(90deg, transparent, var(--surface-border), transparent);
        }
        .glass-card:hover { transform: translateY(-4px); border-color: var(--surface-hover); box-shadow: 0 16px 48px 0 rgba(0, 0, 0, 0.5); }
        
        h1, h2, h3 { font-weight: 600; letter-spacing: -0.03em; }
        
        .app-header { text-align: center; padding: 1rem 1rem 2.5rem 1rem; animation: fadeInDown 0.8s cubic-bezier(0.16, 1, 0.3, 1); }
        .app-header h1 { 
            font-size: 2.75rem; 
            background: linear-gradient(135deg, #ffffff 0%, var(--text-secondary) 100%); 
            -webkit-background-clip: text; 
            -webkit-text-fill-color: transparent; 
        }
        .app-header p { color: var(--text-secondary); margin-top: 0.5rem; font-size: 1.05rem; font-weight: 400; }

        /* --- FORM STYLES --- */
        .input-group { margin-bottom: 1.1rem; }
        .input-group label { display: block; font-size: 0.75rem; color: var(--text-secondary); margin-bottom: 0.4rem; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600; }
        .input-group input { 
            width: 100%; 
            padding: 0.75rem 1rem; 
            background: rgba(0, 0, 0, 0.3); 
            border: 1px solid var(--surface-border); 
            border-radius: 10px; 
            color: var(--text-primary); 
            font-family: var(--font-family);
            font-size: 0.95rem; 
            transition: all 0.3s ease; 
        }
        .input-group input:focus { 
            outline: none; 
            border-color: var(--accent-primary); 
            box-shadow: 0 0 0 3px var(--accent-glow);
            background: rgba(0, 0, 0, 0.5);
        }
        
        .form-container { max-height: 68vh; overflow-y: auto; padding-right: 10px; }
        .form-container::-webkit-scrollbar { width: 5px; }
        .form-container::-webkit-scrollbar-track { background: transparent; }
        .form-container::-webkit-scrollbar-thumb { background: var(--surface-border); border-radius: 10px; }
        
        /* --- PREMIUM BUTTON ANIMATIONS --- */
        .btn-predict { 
            width: 100%; 
            padding: 1.1rem; 
            background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%); 
            border: none; 
            border-radius: 12px; 
            color: white; 
            font-family: var(--font-family);
            font-size: 1rem; 
            font-weight: 700; 
            cursor: pointer; 
            transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1); 
            text-transform: uppercase; 
            letter-spacing: 0.1em; 
            position: relative; 
            overflow: hidden;
            box-shadow: 0 4px 20px var(--accent-glow);
            margin-top: 1rem;
        }
        .btn-predict::after {
            content: '';
            position: absolute;
            top: -50%; left: -50%; width: 200%; height: 200%;
            background: linear-gradient(60deg, transparent, rgba(255, 255, 255, 0.25), transparent);
            transform: rotate(30deg) translateY(-100%);
            transition: transform 0.8s ease;
        }
        .btn-predict:hover { 
            transform: translateY(-2px) scale(1.01); 
            box-shadow: 0 8px 30px var(--accent-glow); 
        }
        .btn-predict:hover::after {
            transform: rotate(30deg) translateY(100%);
        }
        .btn-predict:active { transform: translateY(1px) scale(0.99); }

        /* --- DASHBOARD & METRICS --- */
        .dashboard-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 1.5rem; }
        .full-width { grid-column: 1 / -1; }
        
        .result-display { text-align: center; padding: 2.5rem 2rem; display: flex; flex-direction: column; justify-content: center; align-items: center; }
        .result-status { font-size: 2.25rem; font-weight: 700; margin-top: 0.75rem; transition: all 0.5s ease; display: flex; align-items: center; gap: 0.75rem; }
        
        .logo-container svg { width: 42px; height: 42px; stroke: currentColor; stroke-width: 2.2; filter: drop-shadow(0 0 8px currentColor); }
        
        .metric-cards { display: flex; justify-content: space-around; width: 100%; margin-top: 2rem; gap: 1.25rem; }
        .metric { 
            text-align: center; 
            padding: 1.25rem 1rem; 
            background: rgba(0, 0, 0, 0.25); 
            border-radius: 14px; 
            flex: 1; 
            border: 1px solid var(--surface-border); 
            transition: all 0.3s ease; 
        }
        .metric:hover { border-color: var(--accent-primary); transform: translateY(-2px); }
        .metric-value { font-size: 1.75rem; font-weight: 700; color: var(--accent-secondary); transition: color 0.5s ease;}
        .metric-label { font-size: 0.7rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.08em; margin-top: 0.25rem; font-weight: 600;}
        
        .chart-container { position: relative; height: 280px; width: 100%; margin-top: 1rem; }
        .radar-container { position: relative; height: 320px; width: 100%; margin-top: 1rem; }
        
        /* --- ANIMATIONS --- */
        @keyframes fadeInDown { from { opacity: 0; transform: translateY(-30px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes fadeInUp { from { opacity: 0; transform: translateY(30px); } to { opacity: 1; transform: translateY(0); } }
        .animate-up { animation: fadeInUp 0.8s cubic-bezier(0.16, 1, 0.3, 1) forwards; opacity: 0; }
        .delay-1 { animation-delay: 0.15s; }
        .delay-2 { animation-delay: 0.3s; }
        
        @media (max-width: 1024px) { 
            .container { grid-template-columns: 1fr; } 
            .dashboard-grid { grid-template-columns: 1fr; } 
            .metric-cards { flex-direction: column; }
        }
    </style>
</head>
<body class="theme-default">

    <nav class="top-nav animate-up">
        <div class="brand">
            <span>PREDICT</span><span class="brand-badge">PRO AI</span>
        </div>
        <div class="theme-selector-wrapper">
            <select id="themeSelector" class="theme-selector" onchange="changeTheme(this.value)">
                <option value="theme-default">Midnight Glass</option>
                <option value="theme-cyberpunk">Cyberpunk</option>
                <option value="theme-forest">Deep Forest</option>
                <option value="theme-ice">Glacial Ice</option>
                <option value="theme-magic">Magic Purple</option>
                <option value="theme-solar">Solar Flare</option>
                <option value="theme-void">Abyssal Void</option>
                <option value="theme-synthwave">Neon Synthwave</option>
            </select>
        </div>
    </nav>

    <header class="app-header">
        <h1>Student Dropout Analytics</h1>
        <p>Real-time machine learning inference & behavioral evaluation platform</p>
    </header>

    <div class="container">
        <!-- Sidebar Input Form -->
        <aside class="glass-card animate-up delay-1">
            <h2 style="font-size: 1.25rem; margin-bottom: 1.25rem;">Student Profile Parameters</h2>
            <div class="form-container">
                <form id="predictionForm">
                    <script>
                        const formFeatures = [
                            {id: 'Student_ID', label: 'Student ID', type: 'number', step: '1', val: 1001},
                            {id: 'Age', label: 'Age', type: 'number', step: '1', val: 20},
                            {id: 'Gender', label: 'Gender (0=M, 1=F)', type: 'number', step: '1', val: 1},
                            {id: 'Family_Income', label: 'Family Income ($)', type: 'number', step: '1000', val: 45000},
                            {id: 'Internet_Access', label: 'Internet Access (0=No, 1=Yes)', type: 'number', step: '1', val: 1},
                            {id: 'Study_Hours_per_Day', label: 'Study Hours/Day', type: 'number', step: '0.1', val: 3.5},
                            {id: 'Attendance_Rate', label: 'Attendance Rate (%)', type: 'number', step: '0.1', val: 85.0},
                            {id: 'Assignment_Delay_Days', label: 'Avg Assignment Delay', type: 'number', step: '1', val: 2},
                            {id: 'Travel_Time_Minutes', label: 'Travel Time (Mins)', type: 'number', step: '1', val: 30},
                            {id: 'Part_Time_Job', label: 'Part Time Job (0=No, 1=Yes)', type: 'number', step: '1', val: 0},
                            {id: 'Scholarship', label: 'Scholarship (0=No, 1=Yes)', type: 'number', step: '1', val: 1},
                            {id: 'Stress_Index', label: 'Stress Index (1-10)', type: 'number', step: '0.1', val: 5.5},
                            {id: 'GPA', label: 'Current GPA', type: 'number', step: '0.01', val: 3.2},
                            {id: 'Semester_GPA', label: 'Last Semester GPA', type: 'number', step: '0.01', val: 3.1},
                            {id: 'CGPA', label: 'Cumulative GPA', type: 'number', step: '0.01', val: 3.15},
                            {id: 'Semester', label: 'Current Semester', type: 'number', step: '1', val: 4},
                            {id: 'Department', label: 'Department Code', type: 'number', step: '1', val: 2},
                            {id: 'Parental_Education', label: 'Parental Education', type: 'number', step: '1', val: 3}
                        ];
                        formFeatures.forEach(f => {
                            document.write(
                                '<div class="input-group">' +
                                '<label for="' + f.id + '">' + f.label + '</label>' +
                                '<input type="' + f.type + '" id="' + f.id + '" name="' + f.id + '" step="' + f.step + '" value="' + f.val + '" required>' +
                                '</div>'
                            );
                        });
                    </script>
                    <button type="submit" class="btn-predict" id="submitBtn">Execute Model</button>
                </form>
            </div>
        </aside>

        <!-- Analytics Dashboard -->
        <main class="dashboard-grid animate-up delay-2">
            <div class="glass-card full-width result-display">
                <h3 style="color: var(--text-secondary); font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.1em;">Inference Status Output</h3>
                <div id="resultOutput" class="result-status">
                    <span style="font-size: 1.1rem; color: var(--text-secondary); font-weight: 400;">Awaiting Input Parameters...</span>
                </div>
                <div class="metric-cards">
                    <div class="metric"><div id="probStay" class="metric-value">--%</div><div class="metric-label">Retention Probability</div></div>
                    <div class="metric"><div id="probDrop" class="metric-value">--%</div><div class="metric-label">Dropout Probability</div></div>
                    <div class="metric"><div id="riskLevel" class="metric-value" style="color: #fff;">--</div><div class="metric-label">Assessed Risk Tier</div></div>
                </div>
            </div>

            <div class="glass-card">
                <h3 style="font-size: 1.1rem;">Prediction Confidence</h3>
                <div class="chart-container"><canvas id="donutChart"></canvas></div>
            </div>

            <div class="glass-card">
                <h3 style="font-size: 1.1rem;">Behavioral Radar Profile</h3>
                <div class="radar-container"><canvas id="radarChart"></canvas></div>
            </div>

            <div class="glass-card full-width">
                <h3 style="font-size: 1.1rem;">Feature Coefficients Impact (Model Weights)</h3>
                <div class="chart-container" style="height: 320px;"><canvas id="barChart"></canvas></div>
            </div>
        </main>
    </div>

    <script>
        const globalFeatures = {{ features | tojson | safe }};
        const globalImportances = {{ importance | tojson | safe }};
        let donutChart, radarChart, barChart;

        const SVG_DROPOUT = '<div class="logo-container"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke-linecap="round" stroke-linejoin="round"><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><polyline points="16 17 21 12 16 7"/><line x1="21" y1="12" x2="9" y2="12"/></svg></div>';
        const SVG_RETENTION = '<div class="logo-container"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke-linecap="round" stroke-linejoin="round"><path d="M22 10v6M2 10l10-5 10 5-10 5z"/><path d="M6 12v5c3 3 9 3 12 0v-5"/></svg></div>';

        const themePalette = {
            'theme-default': { primary: '#3b82f6', secondary: '#06b6d4', danger: '#ef4444', success: '#10b981', grid: 'rgba(255,255,255,0.08)' },
            'theme-cyberpunk': { primary: '#ff007c', secondary: '#00f0ff', danger: '#ff3355', success: '#00ff9d', grid: 'rgba(255,0,124,0.15)' },
            'theme-forest': { primary: '#10b981', secondary: '#84cc16', danger: '#f43f5e', success: '#22c55e', grid: 'rgba(46,160,67,0.15)' },
            'theme-ice': { primary: '#38bdf8', secondary: '#818cf8', danger: '#fb7185', success: '#38bdf8', grid: 'rgba(56,189,248,0.15)' },
            'theme-magic': { primary: '#a855f7', secondary: '#f43f5e', danger: '#f43f5e', success: '#c084fc', grid: 'rgba(168,85,247,0.15)' },
            'theme-solar': { primary: '#f97316', secondary: '#eab308', danger: '#ef4444', success: '#84cc16', grid: 'rgba(249,115,22,0.15)' },
            'theme-void': { primary: '#ffffff', secondary: '#737373', danger: '#a3a3a3', success: '#e5e5e5', grid: 'rgba(255,255,255,0.08)' },
            'theme-synthwave': { primary: '#ec4899', secondary: '#8b5cf6', danger: '#f43f5e', success: '#06b6d4', grid: 'rgba(236,72,153,0.15)' }
        };

        function hexToRgba(hex, alpha) {
            let r = parseInt(hex.slice(1, 3), 16),
                g = parseInt(hex.slice(3, 5), 16),
                b = parseInt(hex.slice(5, 7), 16);
            return 'rgba(' + r + ', ' + g + ', ' + b + ', ' + alpha + ')';
        }

        Chart.defaults.color = '#94a3b8';
        Chart.defaults.font.family = "'Plus Jakarta Sans', sans-serif";
        Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(8, 12, 20, 0.9)';
        Chart.defaults.plugins.tooltip.titleColor = '#fff';
        Chart.defaults.plugins.tooltip.padding = 12;
        Chart.defaults.plugins.tooltip.cornerRadius = 8;

        function initCharts() {
            const currentTheme = themePalette['theme-default'];

            const ctxDonut = document.getElementById('donutChart').getContext('2d');
            donutChart = new Chart(ctxDonut, {
                type: 'doughnut',
                data: { labels: ['Will Not Dropout', 'Will Dropout'], datasets: [{ data: [50, 50], backgroundColor: [currentTheme.success, currentTheme.danger], borderWidth: 0 }] },
                options: { responsive: true, maintainAspectRatio: false, cutout: '78%', plugins: { legend: { position: 'bottom', labels: { boxWidth: 12, padding: 20 } } }, animation: { duration: 1000 } }
            });

            const ctxRadar = document.getElementById('radarChart').getContext('2d');
            radarChart = new Chart(ctxRadar, {
                type: 'radar',
                data: {
                    labels: ['Attendance', 'GPA', 'Study Hours', 'Stress', 'Delay Index'],
                    datasets: [{ label: 'Student Profile', data: [0,0,0,0,0], backgroundColor: hexToRgba(currentTheme.primary, 0.25), borderColor: currentTheme.primary, pointBackgroundColor: currentTheme.secondary, borderWidth: 2.5 }]
                },
                options: { responsive: true, maintainAspectRatio: false, scales: { r: { angleLines: { color: currentTheme.grid }, grid: { color: currentTheme.grid }, pointLabels: { color: '#94a3b8', font: { size: 11 } }, ticks: { display: false } } } }
            });

            const bgColors = globalImportances.map(val => val < 0 ? hexToRgba(currentTheme.success, 0.75) : hexToRgba(currentTheme.danger, 0.75));
            const borderColors = globalImportances.map(val => val < 0 ? currentTheme.success : currentTheme.danger);
            const ctxBar = document.getElementById('barChart').getContext('2d');
            barChart = new Chart(ctxBar, {
                type: 'bar',
                data: { labels: globalFeatures, datasets: [{ label: 'Feature Impact', data: globalImportances, backgroundColor: bgColors, borderColor: borderColors, borderWidth: 1, borderRadius: 6 }] },
                options: { responsive: true, maintainAspectRatio: false, scales: { y: { grid: { color: 'rgba(255, 255, 255, 0.05)' } }, x: { grid: { display: false }, ticks: { maxRotation: 45, minRotation: 45 } } }, plugins: { legend: { display: false } } }
            });
        }

        function changeTheme(themeClass) {
            document.body.className = themeClass;
            const colors = themePalette[themeClass];

            donutChart.data.datasets[0].backgroundColor = [colors.success, colors.danger];
            donutChart.update();

            radarChart.data.datasets[0].backgroundColor = hexToRgba(colors.primary, 0.25);
            radarChart.data.datasets[0].borderColor = colors.primary;
            radarChart.data.datasets[0].pointBackgroundColor = colors.secondary;
            radarChart.options.scales.r.angleLines.color = colors.grid;
            radarChart.options.scales.r.grid.color = colors.grid;
            radarChart.update();

            const newBgColors = globalImportances.map(val => val < 0 ? hexToRgba(colors.success, 0.75) : hexToRgba(colors.danger, 0.75));
            const newBorderColors = globalImportances.map(val => val < 0 ? colors.success : colors.danger);
            barChart.data.datasets[0].backgroundColor = newBgColors;
            barChart.data.datasets[0].borderColor = newBorderColors;
            barChart.update();
            
            const riskSpan = document.getElementById('riskLevel');
            const resOut = document.getElementById('resultOutput');
            
            if(riskSpan.textContent === 'High') { 
                riskSpan.style.color = colors.danger; 
                resOut.style.color = colors.danger;
            }
            else if(riskSpan.textContent === 'Low') { 
                riskSpan.style.color = colors.success; 
                resOut.style.color = colors.success;
            }
        }

        document.getElementById('predictionForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            const dataObj = {};
            formData.forEach((value, key) => { dataObj[key] = parseFloat(value); });

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(dataObj)
                });
                const result = await response.json();
                if(response.ok) { updateDashboard(result, dataObj); } else { alert('Error: ' + result.error); }
            } catch (error) { alert('Connection error: ' + error.message); }
        });

        function updateDashboard(apiData, inputData) {
            const currentTheme = document.body.className || 'theme-default';
            const colors = themePalette[currentTheme];

            const resOut = document.getElementById('resultOutput');
            
            if (apiData.prediction_code === 1) {
                resOut.innerHTML = SVG_DROPOUT + '<div>' + apiData.prediction + '</div>';
                resOut.className = 'result-status'; 
                resOut.style.color = colors.danger;
            } else {
                resOut.innerHTML = SVG_RETENTION + '<div>' + apiData.prediction + '</div>';
                resOut.className = 'result-status'; 
                resOut.style.color = colors.success;
            }
            
            document.getElementById('probStay').textContent = apiData.probability_stay + '%';
            document.getElementById('probDrop').textContent = apiData.probability_dropout + '%';
            
            const riskSpan = document.getElementById('riskLevel');
            riskSpan.textContent = apiData.risk_level;
            riskSpan.style.color = apiData.risk_level === 'High' ? colors.danger : colors.success;

            donutChart.data.datasets[0].data = [apiData.probability_stay, apiData.probability_dropout];
            donutChart.update();

            radarChart.data.datasets[0].data = [
                (inputData.Attendance_Rate || 0) / 100 * 100,
                ((inputData.GPA || 0) / 4.0) * 100,
                ((inputData.Study_Hours_per_Day || 0) / 12) * 100,
                ((inputData.Stress_Index || 0) / 10) * 100,
                (20 - (inputData.Assignment_Delay_Days || 0)) / 20 * 100 
            ];
            radarChart.update();
        }

        window.addEventListener('DOMContentLoaded', initCharts);
    </script>
</body>
</html>
"""

# --- ROUTES ---
@app.route('/')
def home():
    """Renders the inline HTML template."""
    return render_template_string(
        HTML_TEMPLATE, 
        features=feature_names,
        importance=feature_importance
    )

@app.route('/predict', methods=['POST'])
def predict():
    """Handles model inference."""
    if not model:
        return jsonify({'error': 'Model failed to load.'}), 500

    try:
        data = request.get_json()
        
        input_data = []
        for feature in feature_names:
            val = float(data.get(feature, 0.0))
            input_data.append(val)
            
        features_array = np.array(input_data).reshape(1, -1)
        
        prediction_val = int(model.predict(features_array)[0])
        probabilities = model.predict_proba(features_array)[0]
        
        if prediction_val == 1:
            result_text = "Student Will Dropout"
            risk_level = "High"
        else:
            result_text = "Student Will Retain"
            risk_level = "Low"
            
        return jsonify({
            'prediction': result_text,
            'prediction_code': prediction_val,
            'probability_dropout': round(probabilities[1] * 100, 2),
            'probability_stay': round(probabilities[0] * 100, 2),
            'risk_level': risk_level,
            'input_echo': input_data
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
