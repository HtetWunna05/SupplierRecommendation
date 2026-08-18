import hashlib
import html
import io
import secrets
import string
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

px.defaults.template = "plotly_white"
px.defaults.width = None

from database import (
    COLLECTIONS,
    append_dataframe_unique,
    clear_supplier_data,
    connect_to_mongodb,
    dataframe_from_collection,
    ensure_indexes,
    log_activity,
    replace_collection_from_dataframe,
)


st.set_page_config(
    page_title="SupplyLogix • AI Supplier Intelligence",
    page_icon="🚚",
    layout="wide",
)



def inject_custom_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
        :root{
          --canvas:#f7f6f2; --paper:#ffffff; --ink:#111111; --muted:#77746f;
          --line:#dedbd3; --nav:#f4f2ed; --accent:#273d70; --accent2:#466a72;
          --lav:#d7c6ff; --green:#809b87; --shadow:0 18px 45px rgba(31,31,28,.08);
        }
        html,body,[class*="css"]{font-family:'DM Sans',sans-serif;color:var(--ink)}
        .stApp{background:var(--canvas)!important;position:relative;min-height:100vh}
        .stApp>header{background:rgba(247,246,242,.96);border-bottom:1px solid #ebe8e1;position:relative;z-index:10}
        section.main{position:relative;z-index:5}
        .main .block-container{position:relative;z-index:6}
        .block-container{max-width:none;width:100%;padding:2.2rem 3.2rem 5rem}
        section.main>div{max-width:none}

        .login-feature-outside{
            margin-top:-.15rem;
            min-height:115px;
            box-sizing:border-box;
            background:rgba(255,255,255,.86);
            border:1px solid rgba(30,38,50,.08);
            box-shadow:0 12px 30px rgba(30,38,50,.06);
        }
        .login-feature-outside span{color:#17213b}
        .login-feature-outside small{color:#77756f}
        .login-panel{margin-top:1.15rem}
        .login-ai-score{margin-top:1rem}
        .login-metric{margin-bottom:.75rem}
        .login-how{margin-top:1rem}
        .login-how-item{min-height:88px;padding:.25rem 0}
        @media (max-width:900px){.login-feature-outside{margin-top:.5rem}}

        /* ============================================================
           SUPPLYLOGIX — SMART LOGIN PAGE
           ============================================================ */
        .login-page-marker{display:none}
        body:has(.login-page-marker) .stApp{
            background:
                radial-gradient(circle at 8% 5%, rgba(121,149,154,.18), transparent 28%),
                radial-gradient(circle at 92% 12%, rgba(201,182,237,.20), transparent 30%),
                #f5f4f0!important;
        }
        body:has(.login-page-marker) .block-container{max-width:1380px;padding:0 2rem 4rem}
        .login-topbar{height:76px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid rgba(30,38,50,.09);margin-bottom:1.4rem}
        .login-top-brand{display:flex;align-items:center;gap:.65rem;font-family:'Manrope',sans-serif;font-size:1.08rem;font-weight:850;letter-spacing:-.025em;color:#17213b}
        .login-top-brand-icon{width:36px;height:36px;display:grid;place-items:center;border-radius:11px;background:#17213b;color:#fff;box-shadow:0 8px 20px rgba(23,33,59,.18)}
        .login-top-note{font-size:.62rem;letter-spacing:.16em;font-weight:850;color:#77756f}
        .login-hero{position:relative;min-height:430px;overflow:hidden;border-radius:30px;background:#17213b;box-shadow:0 25px 65px rgba(29,38,53,.16)}
        .login-hero-photo{position:absolute;inset:0;background-image:linear-gradient(90deg,rgba(12,20,34,.94) 0%,rgba(12,20,34,.82) 40%,rgba(12,20,34,.35) 72%,rgba(12,20,34,.18) 100%),url("https://images.unsplash.com/photo-1553413077-190dd305871c?auto=format&fit=crop&w=2000&q=90");background-size:cover;background-position:center;transform:scale(1.01)}
        .login-hero-overlay{position:absolute;inset:0;background:radial-gradient(circle at 82% 30%,rgba(126,149,154,.28),transparent 28%),linear-gradient(135deg,transparent,rgba(23,33,59,.25))}
        .login-hero-content{position:relative;z-index:2;max-width:700px;padding:4.2rem 4rem;color:#fff}
        .login-eyebrow{display:inline-flex;align-items:center;gap:.45rem;padding:.42rem .7rem;border:1px solid rgba(255,255,255,.22);border-radius:999px;background:rgba(255,255,255,.09);backdrop-filter:blur(10px);font-size:.58rem;letter-spacing:.16em;font-weight:850}
        .login-hero-content h1{margin:1.15rem 0 .75rem!important;font-family:'Manrope',sans-serif;font-size:clamp(2.5rem,5vw,4.4rem)!important;line-height:.98!important;letter-spacing:-.055em;color:#fff!important}
        .login-hero-content h1 span{color:#c9b6ed}
        .login-hero-content>p{max-width:570px;margin:0;color:rgba(255,255,255,.78);font-size:1rem;line-height:1.65}
        .login-feature-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:.75rem;margin-top:2rem}
        .login-feature{min-height:100px;padding:1rem;border:1px solid rgba(255,255,255,.15);border-radius:17px;background:rgba(255,255,255,.085);backdrop-filter:blur(14px)}
        .login-feature b{display:block;margin-bottom:.55rem;color:#c9b6ed;font-size:.62rem;letter-spacing:.12em}.login-feature span{display:block;color:#fff;font-size:.84rem;font-weight:800}.login-feature small{display:block;margin-top:.25rem;color:rgba(255,255,255,.60);font-size:.68rem;line-height:1.35}
        .login-workspace{display:grid;grid-template-columns:1.02fr .98fr;gap:1.35rem;margin-top:1.35rem}
        .login-panel,.login-intelligence{border:1px solid rgba(30,38,50,.09);border-radius:27px;background:rgba(255,255,255,.82);backdrop-filter:blur(18px);box-shadow:0 18px 45px rgba(30,38,50,.07)}
        .login-panel{padding:1.6rem}.login-welcome-card{padding:.25rem .15rem 1rem}.login-card-kicker{margin-bottom:.45rem;color:#806e9f;font-size:.59rem;font-weight:900;letter-spacing:.17em}.login-welcome-card h2{margin:0!important;font-family:'Manrope',sans-serif;font-size:2rem!important;letter-spacing:-.045em;color:#17213b!important}.login-welcome-card p{margin:.45rem 0 0;color:#74736d;font-size:.84rem}
        body:has(.login-page-marker) .stTabs{border:0;background:transparent;padding:0;box-shadow:none}
        body:has(.login-page-marker) [data-baseweb="tab-list"]{padding:.25rem;border-radius:14px;background:#eeede9}
        body:has(.login-page-marker) [data-baseweb="tab"]{min-height:40px;border-radius:10px;color:#77756f!important;font-size:.76rem;font-weight:800}
        body:has(.login-page-marker) [data-baseweb="tab"][aria-selected="true"]{background:#17213b!important;color:#fff!important}
        body:has(.login-page-marker) div[data-testid="stForm"]{border:0!important;background:transparent!important;padding:1rem .05rem .1rem!important;box-shadow:none!important}
        body:has(.login-page-marker) label{color:#44484f!important;font-size:.72rem!important;font-weight:750!important}
        body:has(.login-page-marker) div[data-baseweb="input"]{min-height:49px;border:1px solid #deded9!important;border-radius:13px!important;background:#fff!important;transition:all .18s ease}
        body:has(.login-page-marker) div[data-baseweb="input"]:focus-within{border-color:#687d91!important;box-shadow:0 0 0 3px rgba(104,125,145,.12)!important}
        body:has(.login-page-marker) div[data-baseweb="input"] input{color:#17213b!important}
        body:has(.login-page-marker) div[data-testid="stFormSubmitButton"] button{min-height:51px!important;border:0!important;border-radius:14px!important;background:#17213b!important;color:#fff!important;font-weight:850!important;box-shadow:0 12px 25px rgba(23,33,59,.18)!important;transition:all .18s ease}
        body:has(.login-page-marker) div[data-testid="stFormSubmitButton"] button:hover{background:#273d70!important;transform:translateY(-1px)}
        body:has(.login-page-marker) .stSelectbox div[data-baseweb="select"]>div{background:#fff!important;border:1px solid #deded9!important;border-radius:13px!important}
        body:has(.login-page-marker) .stAlert{border-radius:13px!important}
        .login-intelligence{position:relative;overflow:hidden;padding:1.7rem}.login-intelligence:before{content:"";position:absolute;width:240px;height:240px;top:-110px;right:-80px;border-radius:50%;background:radial-gradient(circle,rgba(201,182,237,.55),transparent 68%)}
        .login-intelligence-kicker{position:relative;color:#806e9f;font-size:.59rem;letter-spacing:.16em;font-weight:900}.login-intelligence h3{position:relative;margin:.45rem 0 .3rem;font-family:'Manrope',sans-serif;font-size:1.55rem;line-height:1.05;letter-spacing:-.04em;color:#17213b}.login-intelligence-subtitle{position:relative;color:#77756f;font-size:.78rem;line-height:1.5}
        .login-ai-score{position:relative;display:flex;align-items:center;gap:1rem;margin:1.25rem 0;padding:1rem;border-radius:19px;background:linear-gradient(120deg,#f4f1fa,#eef4f3);border:1px solid #e0ddd8}.login-ai-orb{width:58px;height:58px;min-width:58px;display:grid;place-items:center;border-radius:50%;background:radial-gradient(circle at 30% 25%,#c9b6ed,#78959a);color:#fff;font-size:1.25rem;box-shadow:0 10px 25px rgba(71,77,95,.18)}.login-ai-score strong{display:block;font-family:'Manrope',sans-serif;font-size:1.35rem;color:#17213b}.login-ai-score span{color:#77756f;font-size:.7rem}
        .login-metrics{position:relative;display:grid;grid-template-columns:repeat(2,1fr);gap:.7rem;margin-top:1rem}.login-metric{padding:.95rem;border:1px solid #e4e1db;border-radius:16px;background:rgba(250,249,246,.85)}.login-metric strong{display:block;font-family:'Manrope',sans-serif;font-size:1.35rem;color:#17213b}.login-metric span{display:block;margin-top:.15rem;color:#7b7973;font-size:.65rem}.login-metric small{display:block;margin-top:.45rem;color:#4e765e;font-size:.62rem;font-weight:750}
        .login-how{margin-top:1.35rem;padding:1.35rem 1.5rem;border:1px solid #e1ded8;border-radius:22px;background:#fbfaf7}.login-how-title{margin-bottom:.8rem;color:#17213b;font-family:'Manrope',sans-serif;font-size:.9rem;font-weight:850}.login-how-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem}.login-how-item{display:flex;gap:.65rem;align-items:flex-start}.login-how-number{width:27px;height:27px;min-width:27px;display:grid;place-items:center;border-radius:9px;background:#17213b;color:#fff;font-size:.62rem;font-weight:850}.login-how-item b{display:block;color:#353a43;font-size:.7rem}.login-how-item span{display:block;margin-top:.15rem;color:#7b7973;font-size:.62rem;line-height:1.35}
        /* Smart Recommendation visual refresh */
        body:has(.login-page-marker) .stApp{background:radial-gradient(circle at 12% 10%,rgba(121,149,154,.16),transparent 25%),radial-gradient(circle at 88% 8%,rgba(201,182,237,.22),transparent 28%),#f7f6f2}
        .login-hero{min-height:465px;border-radius:34px;background:#111b33;box-shadow:0 28px 75px rgba(23,33,59,.20)}
        .login-hero-photo{background-image:linear-gradient(90deg,rgba(10,19,35,.96) 0%,rgba(10,19,35,.82) 34%,rgba(10,19,35,.40) 68%,rgba(10,19,35,.16) 100%),url("https://images.unsplash.com/photo-1553413077-190dd305871c?auto=format&fit=crop&w=2200&q=92");background-position:center 48%;}
        .login-hero-overlay{background:radial-gradient(circle at 78% 24%,rgba(201,182,237,.34),transparent 25%),radial-gradient(circle at 66% 90%,rgba(120,149,154,.25),transparent 28%)}
        .login-hero-content{max-width:760px;padding:4.6rem 4.2rem}
        .login-eyebrow{background:rgba(201,182,237,.13);border-color:rgba(201,182,237,.34);color:#e5dbf5}
        .login-hero-content h1{font-size:clamp(2.8rem,5.5vw,4.8rem)!important}
        .login-hero-content h1 span{color:#d6c4ef}
        .login-feature-grid{grid-template-columns:repeat(3,1fr);gap:.85rem}
        .login-feature{background:rgba(255,255,255,.075);border-color:rgba(255,255,255,.16);border-radius:19px}
        .login-feature b{color:#d6c4ef}
        .login-panel,.login-intelligence{border-radius:30px;box-shadow:0 22px 55px rgba(30,38,50,.075)}
        .login-intelligence{background:linear-gradient(145deg,rgba(255,255,255,.92),rgba(248,246,252,.92))}
        .login-ai-orb{background:conic-gradient(from 210deg,#17213b,#78959a,#c9b6ed,#17213b);box-shadow:0 12px 30px rgba(83,82,110,.20)}
        .login-metric strong{font-size:1.42rem}
        .login-how{background:linear-gradient(135deg,#fbfaf7,#f4f1fa)}

        @media(max-width:900px){body:has(.login-page-marker) .block-container{padding:0 .8rem 2.5rem}.login-topbar{height:65px}.login-top-note{display:none}.login-hero{min-height:510px;border-radius:23px}.login-hero-photo{background-position:65% center}.login-hero-content{padding:2.5rem 1.5rem}.login-hero-content h1{font-size:2.55rem!important}.login-feature-grid{grid-template-columns:1fr;gap:.55rem}.login-feature{min-height:auto;padding:.75rem}.login-workspace{grid-template-columns:1fr}.login-how-grid{grid-template-columns:1fr}}
        @media(max-width:520px){.login-panel,.login-intelligence{padding:1.15rem;border-radius:21px}.login-metrics{grid-template-columns:1fr 1fr}.login-welcome-card h2{font-size:1.7rem!important}}
        
        /* Sidebar: editorial / premium SaaS navigation */
        section[data-testid="stSidebar"]{background:#f1f0eb;border-right:1px solid #dcd9d1}
        section[data-testid="stSidebar"][aria-expanded="true"]{min-width:355px!important;width:355px!important}
        section[data-testid="stSidebar"][aria-expanded="false"]{min-width:0!important;width:0!important;margin-left:0!important;border-right:0!important}
        section[data-testid="stSidebar"][aria-expanded="false"]>div:first-child{display:none!important}
        section[data-testid="stSidebar"]>div:first-child{padding:1.65rem 1.3rem 1.5rem}
        section[data-testid="stSidebar"] *{color:var(--ink)!important}
        .sidebar-brand{display:flex;align-items:center;gap:1rem;padding:.45rem .45rem 1.2rem;border-bottom:1px solid #d9d6ce;margin-bottom:1.35rem}
        .brand-icon{width:58px;height:58px;display:grid;place-items:center;border-radius:19px;background:linear-gradient(145deg,#f5eee2,#dce8df);border:1px solid #d3cec2;box-shadow:inset 0 1px 0 #fff,0 9px 22px rgba(45,43,36,.08);font-size:0}
        .brand-icon:after{content:'◆';font-size:20px;color:#141414;transform:scale(.82)}
        .brand-name{font-family:'Manrope',sans-serif;font-size:1.38rem;font-weight:800;letter-spacing:-.05em}
        .brand-name span{color:#718875!important}
        .brand-subtitle{font-size:.68rem!important;font-weight:800;letter-spacing:.13em;text-transform:uppercase;margin-top:.32rem;color:#171717!important}
        .sidebar-role-badge{background:#fff;border:1px solid #ddd9d0;border-radius:21px;padding:1rem 1rem;margin:.3rem 0 1rem;box-shadow:0 9px 25px rgba(36,35,31,.045)}
        .sidebar-role-badge small{display:block!important;color:#77746f!important;margin-top:.25rem;font-size:.75rem}
        section[data-testid="stSidebar"] .stButton>button{width:auto!important;border-radius:13px!important;background:#fff!important;color:#151515!important;border:1px solid #d6d1c8!important;padding:.55rem 1rem!important;box-shadow:none!important}
        section[data-testid="stSidebar"] .stButton>button:hover{background:#f8f6f1!important;border-color:#b9b4aa!important}
        .smart-nav{margin:.35rem 0 0}
        .smart-nav-section{display:flex;align-items:center;justify-content:space-between;padding:.35rem .75rem .55rem}
        .smart-nav-section span:first-child{font-size:.66rem;font-weight:800;letter-spacing:.15em;text-transform:uppercase;color:#817d75!important}
        .smart-nav-section span:last-child{font-size:.58rem;color:#9a958c!important}
        .smart-nav-item{display:flex;align-items:center;gap:.75rem;padding:.72rem .78rem;border-radius:15px;margin:.16rem 0;color:#26241f!important;background:transparent;border:1px solid transparent;transition:.18s ease}
        .smart-nav-item:hover{background:#e7e5df;border-color:#ddd9d0;transform:translateX(2px)}
        .smart-nav-item .nav-icon{width:32px;height:32px;display:grid;place-items:center;border-radius:10px;background:#ebe9e3;font-size:15px;flex:0 0 32px}
        .smart-nav-item .nav-copy{min-width:0;flex:1}
        .smart-nav-item .nav-title{font-size:.91rem;font-weight:650;line-height:1.1}
        .smart-nav-item .nav-desc{font-size:.68rem;color:#858179!important;margin-top:.2rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
        .smart-nav-item .nav-arrow{font-size:.9rem;color:#a29d94!important;opacity:0;transition:.18s}
        .smart-nav-item:hover .nav-arrow{opacity:1;transform:translateX(2px)}
        .smart-nav-active{background:#111!important;color:#fff!important;border-color:#111!important;box-shadow:0 12px 25px rgba(0,0,0,.12)}
        .smart-nav-active .nav-icon{background:rgba(255,255,255,.12)!important;color:#fff!important}
        .smart-nav-active .nav-title{color:#fff!important}
        .smart-nav-active .nav-desc{color:#bdbdbd!important}
        .smart-nav-active .nav-arrow{opacity:1;color:#fff!important}
        section[data-testid="stSidebar"] .smart-nav-button .stButton>button{width:100%!important;text-align:left!important;border:0!important;border-radius:15px!important;background:transparent!important;color:#26241f!important;padding:.72rem .78rem!important;box-shadow:none!important;font-size:.91rem!important;font-weight:650!important;min-height:52px!important;transition:.18s ease!important}
        section[data-testid="stSidebar"] .smart-nav-button .stButton>button:hover{background:#e7e5df!important;border-color:#ddd9d0!important;transform:translateX(2px)!important}
        .sidebar-footer{position:relative;margin-top:2.5rem;padding:1rem .7rem;border-top:1px solid #d9d6ce;color:#25231f!important}
        .sidebar-footer b{font-size:.7rem;letter-spacing:.12em}.sidebar-footer span{display:block;font-size:.72rem;color:#6e6b65!important;margin-top:.3rem}

        /* Main page / hero */
        .page-hero{position:relative;overflow:hidden;border-radius:32px;padding:2.55rem 2.75rem;margin:.8rem 0 1.55rem;background:linear-gradient(115deg,#283b70 0%,#304775 56%,#3d666a 100%);box-shadow:0 22px 50px rgba(41,58,96,.16);min-height:270px;display:flex;align-items:center}
        .page-hero:before{content:'';position:absolute;width:420px;height:420px;right:-55px;top:-130px;border-radius:50%;background:radial-gradient(circle,rgba(188,179,247,.42),rgba(132,170,183,.17) 38%,transparent 68%)}
        .page-hero:after{content:'✦';position:absolute;right:130px;top:73px;width:124px;height:124px;border-radius:50%;display:grid;place-items:center;color:#fff;font-size:38px;background:linear-gradient(145deg,#a58ad3,#6f9299);box-shadow:0 0 0 1px rgba(255,255,255,.18),0 0 0 52px rgba(255,255,255,.035),0 0 0 1px rgba(255,255,255,.12) inset}
        .page-hero-top{position:relative;z-index:2;display:block;max-width:720px}
        .page-hero-icon{display:none}
        .ui-kicker{display:block;background:none;border:0;padding:0;color:#ddd5ff!important;font-size:.72rem;font-weight:800;letter-spacing:.16em;margin-bottom:1.05rem}
        .page-hero h1{font-family:'Manrope',sans-serif;color:#dfd2ff!important;font-size:3rem!important;line-height:1.03!important;letter-spacing:-.065em!important;margin:0!important;font-weight:800!important;max-width:680px}
        .page-hero p{color:#e7e9f2!important;margin:.9rem 0 0!important;font-size:1rem;line-height:1.65;max-width:690px}
        .ui-status{display:none}
        h1,h2,h3{font-family:'Manrope',sans-serif;color:#111!important;letter-spacing:-.045em}
        h1{font-weight:800!important}h2,h3{font-weight:750!important}
        h2::before,h3::before{display:none}
        .section-kicker{font-size:.7rem;font-weight:800;letter-spacing:.17em;text-transform:uppercase;color:#7560a0!important;margin-top:2rem}

        /* Metrics / cards */
        div[data-testid="stMetric"]{background:#fff;border:1px solid #dedbd4;border-top:4px solid #111;border-radius:20px;padding:1.1rem 1.2rem;box-shadow:0 12px 30px rgba(30,29,26,.055);min-height:105px}
        div[data-testid="stMetric"]:hover{transform:translateY(-2px);box-shadow:var(--shadow)}
        div[data-testid="stMetricLabel"] p{color:#2d2b27!important;font-size:.9rem!important;font-weight:500!important;text-transform:none;letter-spacing:0}
        div[data-testid="stMetricValue"]{color:#111!important;font-family:'Manrope';font-size:1.7rem;font-weight:800}
        div[data-testid="stMetricDelta"]{color:#708a76!important}
        .metric-card{background:#fff;border:1px solid #dedbd4;border-top:4px solid #111;border-radius:20px;padding:1.1rem 1.2rem;box-shadow:0 12px 30px rgba(30,29,26,.055)}
        .metric-card .label{font-size:.9rem;color:#2d2b27}.metric-card .value{font-family:'Manrope';font-size:1.75rem;font-weight:800;margin-top:.3rem}.metric-card .subvalue{font-size:.76rem;color:#77746f;margin-top:.28rem;font-weight:650}

        /* Tables / charts / controls */
        div[data-testid="stDataFrame"]{background:#fff;border:1px solid #dedbd4;border-radius:20px;overflow:hidden;box-shadow:0 12px 30px rgba(30,29,26,.05)}
        div[data-testid="stDataFrame"] div[role="columnheader"]{background:#f2f0eb!important;color:#24231f!important;font-weight:700!important}
        div[data-testid="stPlotlyChart"]{background:#fff;border:1px solid #dedbd4;border-radius:20px;padding:.6rem;box-shadow:0 12px 30px rgba(30,29,26,.05)}
        div[data-testid="stForm"],div[data-testid="stExpander"]{background:#fff;border:1px solid #dedbd4;border-radius:22px;padding:1.05rem;box-shadow:0 12px 30px rgba(30,29,26,.05)}
        .stTabs [data-baseweb="tab-list"]{background:#eeece6;border:1px solid #dedbd4;border-radius:15px;padding:.25rem;gap:.2rem}
        .stTabs [data-baseweb="tab"]{border-radius:11px;color:#6b6861!important;font-weight:700}
        .stTabs [aria-selected="true"]{background:#111;color:#fff!important;box-shadow:0 8px 18px rgba(0,0,0,.12)}
        .stButton>button,.stDownloadButton>button,button[kind="secondary"]{border-radius:13px!important;background:#fff!important;color:#151515!important;border:1px solid #d3cec4!important;font-weight:700!important;transition:.18s}
        .stButton>button:hover,.stDownloadButton>button:hover{transform:translateY(-1px);border-color:#9c978e!important;box-shadow:0 8px 18px rgba(0,0,0,.06)}
        .stButton>button[kind="primary"]{background:#111!important;color:#fff!important;border-color:#111!important;box-shadow:0 9px 20px rgba(0,0,0,.14)}
        div[data-baseweb="input"],div[data-baseweb="select"]>div,div[data-baseweb="textarea"]{background:#fff!important;border-radius:12px!important;border:1px solid #d7d3ca!important;color:#171717!important}
        div[data-baseweb="input"] input,div[data-baseweb="textarea"] textarea{color:#111!important}
        .stSelectbox label,.stTextInput label,.stNumberInput label,.stSlider label,.stMultiSelect label{color:#34312c!important;font-weight:600!important}
        .stAlert{border-radius:15px}
        hr{border-color:#dedbd4}
        .stCaption,small{color:#77746f!important}
        .discover-title{font-family:'Manrope';font-weight:800;font-size:1rem;letter-spacing:-.02em;margin-top:.25rem}
        .discover-kicker{color:#7661a0!important;font-size:.7rem;font-weight:800;letter-spacing:.16em;text-transform:uppercase;margin-top:2.2rem}
        .info-card{background:#fff;border:1px solid #dedbd4;border-radius:20px;padding:1.25rem;box-shadow:0 12px 30px rgba(30,29,26,.05)}

        /* Smart requirement comparison */
        .match-summary{position:relative;overflow:hidden;margin:.2rem 0 .75rem;padding:1.25rem 1.35rem;border:1px solid rgba(39,61,112,.16);border-radius:22px;background:linear-gradient(135deg,#ffffff 0%,#f6f3ff 55%,#eef6f2 100%);box-shadow:0 14px 35px rgba(39,61,112,.07)}
        .match-summary:after{content:'✦';position:absolute;right:22px;top:12px;font-size:3.4rem;color:#c8b7eb;opacity:.45}
        .match-summary-title{font-family:'Manrope',sans-serif;font-size:1.35rem;font-weight:800;letter-spacing:-.04em;color:#17213b}
        .match-summary-sub{margin-top:.35rem;color:#666a76;font-size:.88rem;line-height:1.5}
        .match-summary-sub strong{color:#24375e}
        .match-table{width:100%;border:1px solid #dfe2e8;border-radius:20px;overflow:hidden;background:#fff;box-shadow:0 14px 35px rgba(25,35,55,.07);margin:.7rem 0 1rem}
        .match-row{display:grid;grid-template-columns:1.25fr .9fr .9fr;align-items:center;min-height:62px;padding:0 1.2rem;border-top:1px solid #e8e9ed;gap:1rem}
        .match-row:first-child{border-top:0}
        .match-head{min-height:48px;background:#17213b;color:#fff;border-top:0;font-size:.68rem;font-weight:800;letter-spacing:.12em}
        .match-head div:nth-child(2),.match-head div:nth-child(3){text-align:right}
        .match-label{font-size:.88rem;font-weight:650;color:#333a4b}
        .match-yours{text-align:right;font-size:.9rem;font-weight:700;color:#596071}
        .match-supplier{text-align:right;font-size:.94rem;font-weight:800;color:#2f7057}
        .match-good{background:linear-gradient(90deg,#fff 0%,#fbfcfd 58%,#f2faf5 100%)}
        .match-good:hover{background:linear-gradient(90deg,#fff 0%,#f8f7ff 55%,#edf8f1 100%)}
        .smart-preview.good{display:grid;grid-template-columns:auto 1fr auto;align-items:center;gap:1rem;padding:1rem 1.2rem;border-radius:18px;background:linear-gradient(100deg,#f3faf6,#eef6f2);border:1px solid #c9dfd2;box-shadow:0 10px 26px rgba(58,111,83,.07)}
        .smart-preview.good .preview-score{min-width:78px;width:78px;height:58px;border-radius:16px;background:#fff;border:1px solid #d8e7dd;justify-content:center;align-items:center;gap:.3rem;box-shadow:0 6px 15px rgba(48,95,70,.06)}
        .smart-preview.good .preview-score strong{color:#3b7d5e;font-size:1.65rem}
        .smart-preview.good .preview-score span{text-transform:uppercase;letter-spacing:.07em;font-size:.56rem;font-weight:800;color:#668072}
        .smart-preview.good b{font-size:.88rem;color:#244d3b}
        .smart-preview.good p{font-size:.76rem;color:#6b776f}
        .smart-preview.good .preview-pill{background:#263d35;padding:.55rem .85rem;font-size:.72rem;box-shadow:0 7px 15px rgba(25,51,41,.12)}

        /* Premium login page */
        body:has(.login-page-marker) .stApp{
            background-image:linear-gradient(135deg,rgba(246,243,236,.96),rgba(237,241,239,.94)),url('https://images.unsplash.com/photo-1556761175-b413da4baf72?auto=format&fit=crop&w=1800&q=85');
            background-size:cover;background-position:center;background-attachment:fixed;
        }
        body:has(.login-page-marker) .block-container{max-width:1220px;padding:2.2rem 2.2rem 4rem}
        body:has(.login-page-marker) h1{font-size:2rem!important;color:#18243d!important;margin-bottom:.1rem!important}
        body:has(.login-page-marker) .stCaption{color:#6e746f!important}
        body:has(.login-page-marker) div[data-testid="stImage"]{border-radius:28px;overflow:hidden;box-shadow:0 25px 60px rgba(30,42,58,.18);margin-bottom:1.1rem}
        body:has(.login-page-marker) div[data-testid="stImage"] img{border-radius:28px;max-height:310px;object-fit:cover}
        body:has(.login-page-marker) div[data-testid="stMetric"]{background:rgba(255,255,255,.86);backdrop-filter:blur(12px);border:1px solid rgba(255,255,255,.8);border-top:3px solid #273d70;border-radius:18px;min-height:92px;padding:.8rem 1rem;box-shadow:0 12px 30px rgba(30,42,58,.07)}
        body:has(.login-page-marker) .stTabs{background:rgba(255,255,255,.82);backdrop-filter:blur(16px);border:1px solid rgba(255,255,255,.85);border-radius:26px;padding:1.25rem;box-shadow:0 25px 60px rgba(30,42,58,.12)}
        body:has(.login-page-marker) [data-baseweb="tab-list"]{background:#ececf0;border-radius:14px;padding:.25rem}
        body:has(.login-page-marker) [data-baseweb="tab"]{border-radius:10px;color:#6e7180!important;font-weight:750}
        body:has(.login-page-marker) [data-baseweb="tab"][aria-selected="true"]{background:#17213b!important;color:#fff!important}
        body:has(.login-page-marker) div[data-testid="stForm"]{border:0!important;background:transparent!important;padding:.8rem .15rem .1rem!important;box-shadow:none!important}
        body:has(.login-page-marker) div[data-baseweb="input"]{background:#fff!important;border:1px solid #d9dce2!important;border-radius:13px!important;min-height:48px}
        body:has(.login-page-marker) div[data-baseweb="input"]:focus-within{border-color:#536c91!important;box-shadow:0 0 0 3px rgba(83,108,145,.12)!important}
        body:has(.login-page-marker) div[data-baseweb="input"] input{color:#17213b!important}
        body:has(.login-page-marker) div[data-testid="stFormSubmitButton"] button{background:#17213b!important;color:#fff!important;border:0!important;border-radius:14px!important;min-height:50px!important;font-weight:800!important;box-shadow:0 12px 25px rgba(23,33,59,.18)!important}
        body:has(.login-page-marker) div[data-testid="stFormSubmitButton"] button:hover{background:#273d70!important;transform:translateY(-1px)!important}
        body:has(.login-page-marker) .stSelectbox div[data-baseweb="select"]>div{background:#fff!important;border:1px solid #d9dce2!important;border-radius:13px!important}
        body:has(.login-page-marker) .stMarkdown h2{font-size:2rem!important;color:#17213b!important}
        body:has(.login-page-marker) .stAlert{border-radius:14px!important}

        /* Smart Recommendation visual refresh */
        body:has(.login-page-marker) .stApp{background:radial-gradient(circle at 12% 10%,rgba(121,149,154,.16),transparent 25%),radial-gradient(circle at 88% 8%,rgba(201,182,237,.22),transparent 28%),#f7f6f2}
        .login-hero{min-height:465px;border-radius:34px;background:#111b33;box-shadow:0 28px 75px rgba(23,33,59,.20)}
        .login-hero-photo{background-image:linear-gradient(90deg,rgba(10,19,35,.96) 0%,rgba(10,19,35,.82) 34%,rgba(10,19,35,.40) 68%,rgba(10,19,35,.16) 100%),url("https://images.unsplash.com/photo-1553413077-190dd305871c?auto=format&fit=crop&w=2200&q=92");background-position:center 48%;}
        .login-hero-overlay{background:radial-gradient(circle at 78% 24%,rgba(201,182,237,.34),transparent 25%),radial-gradient(circle at 66% 90%,rgba(120,149,154,.25),transparent 28%)}
        .login-hero-content{max-width:760px;padding:4.6rem 4.2rem}
        .login-eyebrow{background:rgba(201,182,237,.13);border-color:rgba(201,182,237,.34);color:#e5dbf5}
        .login-hero-content h1{font-size:clamp(2.8rem,5.5vw,4.8rem)!important}
        .login-hero-content h1 span{color:#d6c4ef}
        .login-feature-grid{grid-template-columns:repeat(3,1fr);gap:.85rem}
        .login-feature{background:rgba(255,255,255,.075);border-color:rgba(255,255,255,.16);border-radius:19px}
        .login-feature b{color:#d6c4ef}
        .login-panel,.login-intelligence{border-radius:30px;box-shadow:0 22px 55px rgba(30,38,50,.075)}
        .login-intelligence{background:linear-gradient(145deg,rgba(255,255,255,.92),rgba(248,246,252,.92))}
        .login-ai-orb{background:conic-gradient(from 210deg,#17213b,#78959a,#c9b6ed,#17213b);box-shadow:0 12px 30px rgba(83,82,110,.20)}
        .login-metric strong{font-size:1.42rem}
        .login-how{background:linear-gradient(135deg,#fbfaf7,#f4f1fa)}

        @media(max-width:900px){.match-row{grid-template-columns:1.1fr .9fr .9fr;padding:0 .75rem;gap:.5rem}.match-label,.match-yours,.match-supplier{font-size:.76rem}.login-side{min-height:420px}.login-form-title{margin-top:1.5rem}}
        /* Smart Recommendation visual refresh */
        body:has(.login-page-marker) .stApp{background:radial-gradient(circle at 12% 10%,rgba(121,149,154,.16),transparent 25%),radial-gradient(circle at 88% 8%,rgba(201,182,237,.22),transparent 28%),#f7f6f2}
        .login-hero{min-height:465px;border-radius:34px;background:#111b33;box-shadow:0 28px 75px rgba(23,33,59,.20)}
        .login-hero-photo{background-image:linear-gradient(90deg,rgba(10,19,35,.96) 0%,rgba(10,19,35,.82) 34%,rgba(10,19,35,.40) 68%,rgba(10,19,35,.16) 100%),url("https://images.unsplash.com/photo-1553413077-190dd305871c?auto=format&fit=crop&w=2200&q=92");background-position:center 48%;}
        .login-hero-overlay{background:radial-gradient(circle at 78% 24%,rgba(201,182,237,.34),transparent 25%),radial-gradient(circle at 66% 90%,rgba(120,149,154,.25),transparent 28%)}
        .login-hero-content{max-width:760px;padding:4.6rem 4.2rem}
        .login-eyebrow{background:rgba(201,182,237,.13);border-color:rgba(201,182,237,.34);color:#e5dbf5}
        .login-hero-content h1{font-size:clamp(2.8rem,5.5vw,4.8rem)!important}
        .login-hero-content h1 span{color:#d6c4ef}
        .login-feature-grid{grid-template-columns:repeat(3,1fr);gap:.85rem}
        .login-feature{background:rgba(255,255,255,.075);border-color:rgba(255,255,255,.16);border-radius:19px}
        .login-feature b{color:#d6c4ef}
        .login-panel,.login-intelligence{border-radius:30px;box-shadow:0 22px 55px rgba(30,38,50,.075)}
        .login-intelligence{background:linear-gradient(145deg,rgba(255,255,255,.92),rgba(248,246,252,.92))}
        .login-ai-orb{background:conic-gradient(from 210deg,#17213b,#78959a,#c9b6ed,#17213b);box-shadow:0 12px 30px rgba(83,82,110,.20)}
        .login-metric strong{font-size:1.42rem}
        .login-how{background:linear-gradient(135deg,#fbfaf7,#f4f1fa)}

        @media(max-width:900px){.block-container{padding:1.2rem .8rem 3rem}.page-hero{padding:1.7rem;min-height:240px}.page-hero h1{font-size:2rem!important}.page-hero:after{right:-20px;opacity:.65}.page-hero p{font-size:.9rem}}
                .smart-search-banner{display:flex;justify-content:space-between;align-items:center;gap:2rem;padding:1.55rem 1.7rem;margin:.4rem 0 1.25rem;border:1px solid #d9d6ce;border-radius:26px;background:linear-gradient(110deg,#ffffff 0%,#f4f1fa 56%,#e5eff0 100%);box-shadow:0 18px 45px rgba(31,31,28,.07)}
        .smart-search-banner h2{font-family:'Manrope',sans-serif;font-size:2rem;letter-spacing:-.045em;margin:.25rem 0 .35rem}.smart-search-banner p{color:#68655f;margin:0;max-width:720px}
        .smart-ai-orb{width:94px;height:94px;min-width:94px;border-radius:50%;display:flex;flex-direction:column;align-items:center;justify-content:center;background:radial-gradient(circle at 35% 30%,#c9b6ed,#78959a);box-shadow:0 0 0 14px rgba(126,145,160,.08),0 15px 35px rgba(71,77,95,.16);color:white}.smart-ai-orb span{font-size:1.5rem}.smart-ai-orb small{font-size:.48rem;font-weight:800;letter-spacing:.12em;text-align:center}
        .smart-section-label{font-size:.68rem;font-weight:800;letter-spacing:.17em;color:#806e9f;margin:1.45rem 0 .7rem;text-transform:uppercase}.form-mini-title{font-size:.68rem;letter-spacing:.14em;font-weight:800;color:#77746f;margin:.2rem 0 .5rem}
        .smart-preview{display:flex;align-items:center;gap:1rem;padding:1rem 1.1rem;margin:1rem 0;border-radius:18px;border:1px solid #d9d6ce;background:#fbfaf7}.smart-preview.warning{background:#fffaf0;border-color:#ead9b1}.smart-preview.good{background:#f4faf5;border-color:#cfe0d2}.preview-score{display:flex;align-items:baseline;gap:.4rem;min-width:90px}.preview-score strong{font-family:'Manrope',sans-serif;font-size:2rem}.preview-score span{font-size:.68rem;color:#77746f;line-height:1.05}.smart-preview p{margin:.18rem 0 0;color:#6d6962;font-size:.82rem}.preview-pill{margin-left:auto;border-radius:999px;background:#111;color:#fff;padding:.45rem .7rem;font-size:.68rem;font-weight:700}
        .empty-smart-state{text-align:center;padding:3rem 1rem;border:1px dashed #d2cec4;border-radius:24px;background:rgba(255,255,255,.5);margin-top:1rem}.empty-smart-state span{font-size:2rem}.empty-smart-state h3{font-family:'Manrope',sans-serif;margin:.5rem 0 .2rem}.empty-smart-state p{color:#77746f;margin:0}.supplier-result-head{display:flex;align-items:center;gap:.8rem}.supplier-result-head h3{margin:0;font-family:'Manrope',sans-serif}.supplier-result-head p{margin:.15rem 0 0;color:#77746f;font-size:.8rem}.rank-badge{width:38px;height:38px;border-radius:12px;display:grid;place-items:center;background:#111;color:#fff;font-weight:800}.match-score{text-align:right}.match-score span{display:block;font-size:.55rem;letter-spacing:.12em;color:#8172a0;font-weight:800}.match-score strong{font-family:'Manrope',sans-serif;font-size:2rem}.match-score small{color:#77746f}.reason-card{margin:.7rem 0 0;padding:.75rem .9rem;border-radius:14px;background:#f7f5f0;border:1px solid #e3dfd6;display:flex;gap:.5rem;font-size:.82rem}.reason-card b{white-space:nowrap}.reason-card span{color:#68655f}
\n        /* FINAL CLEAN LOGIN DESIGN - overrides all older login styles */\n        body:has(.login-page-marker) .stApp{min-height:100vh!important;background:linear-gradient(90deg,rgba(8,18,31,.91) 0%,rgba(8,18,31,.80) 35%,rgba(8,18,31,.48) 68%,rgba(8,18,31,.20) 100%),url("https://images.unsplash.com/photo-1586528116311-ad8dd3c8310d?auto=format&fit=crop&w=2200&q=90") center/cover fixed!important}\n        body:has(.login-page-marker) .stApp>header{display:none!important}\n        body:has(.login-page-marker) .block-container{max-width:1400px!important;padding:0 5vw 0!important}\n        body:has(.login-page-marker) .clean-login-left{min-height:100vh;display:flex;flex-direction:column;justify-content:center;padding:55px 0;color:#fff}\n        body:has(.login-page-marker) .clean-login-brand{display:flex;align-items:center;gap:13px;margin-bottom:60px}\n        body:has(.login-page-marker) .clean-brand-logo{width:52px;height:52px;display:grid;place-items:center;border-radius:17px;background:rgba(255,255,255,.13);border:1px solid rgba(255,255,255,.25);box-shadow:0 12px 30px rgba(0,0,0,.16);backdrop-filter:blur(14px);font-size:22px}\n        body:has(.login-page-marker) .clean-brand-name{color:#fff!important;font-family:'Manrope',sans-serif;font-size:1.3rem;line-height:1;font-weight:850;letter-spacing:-.04em}\n        body:has(.login-page-marker) .clean-brand-name span{color:#b9d7d2!important}\n        body:has(.login-page-marker) .clean-brand-subtitle{margin-top:6px;color:rgba(255,255,255,.58)!important;font-size:.55rem;font-weight:800;letter-spacing:.16em}\n        body:has(.login-page-marker) .clean-eyebrow{display:inline-flex;width:max-content;padding:9px 15px;border-radius:999px;background:rgba(255,255,255,.10);border:1px solid rgba(255,255,255,.20);color:#dbe9e7!important;font-size:.62rem;font-weight:850;letter-spacing:.14em;backdrop-filter:blur(12px)}\n        body:has(.login-page-marker) .clean-main-title{margin:25px 0 0!important;color:#fff!important;font-family:'Manrope',sans-serif!important;font-size:clamp(3.2rem,5.7vw,5.8rem)!important;line-height:.95!important;font-weight:850!important;letter-spacing:-.07em!important;text-shadow:0 4px 25px rgba(0,0,0,.20)}\n        body:has(.login-page-marker) .clean-main-title span{color:#b9d7d2!important}\n        body:has(.login-page-marker) .clean-main-description{max-width:620px;margin:28px 0 0;color:rgba(255,255,255,.80)!important;font-size:.98rem;line-height:1.75}\n        body:has(.login-page-marker) .clean-benefits{display:flex;flex-wrap:wrap;gap:10px;margin-top:34px}\n        body:has(.login-page-marker) .clean-benefit{display:flex;align-items:center;gap:9px;padding:10px 14px;border-radius:999px;background:rgba(255,255,255,.09);border:1px solid rgba(255,255,255,.16);backdrop-filter:blur(12px);color:#fff!important;font-size:.68rem;font-weight:800}\n        body:has(.login-page-marker) .benefit-icon{width:21px;height:21px;display:grid;place-items:center;border-radius:50%;background:#b9d7d2;color:#142033!important;font-size:.65rem;font-weight:900}\n        body:has(.login-page-marker) .clean-card-heading,body:has(.login-page-marker) .stTabs{max-width:450px;margin-left:auto;margin-right:auto}\n        body:has(.login-page-marker) .clean-card-heading{margin-top:60px;padding:36px 38px 0;background:rgba(255,255,255,.96);border-radius:34px 34px 0 0;border:1px solid rgba(255,255,255,.8);border-bottom:0;box-shadow:0 -20px 60px rgba(0,0,0,.17)}\n        body:has(.login-page-marker) .clean-card-icon{width:42px;height:42px;display:grid;place-items:center;border-radius:14px;background:#e7f0ef;color:#2c5660!important;font-size:18px;margin-bottom:18px}\n        body:has(.login-page-marker) .clean-card-heading h2{margin:0!important;color:#142033!important;font-family:'Manrope',sans-serif!important;font-size:2rem!important;font-weight:850!important;letter-spacing:-.05em!important}\n        body:has(.login-page-marker) .clean-card-heading p{margin:8px 0 0;color:#747c85!important;font-size:.80rem;line-height:1.55}\n        body:has(.login-page-marker) .stTabs{margin-top:0!important;padding:22px 38px 36px!important;background:rgba(255,255,255,.96)!important;border-radius:0 0 34px 34px!important;border:1px solid rgba(255,255,255,.8)!important;border-top:0!important;box-shadow:0 30px 70px rgba(0,0,0,.22)!important}\n        body:has(.login-page-marker) [data-baseweb="tab-list"]{display:flex!important;gap:5px!important;padding:5px!important;border:0!important;border-radius:999px!important;background:#edf0f1!important}\n        body:has(.login-page-marker) [data-baseweb="tab"]{flex:1!important;min-height:44px!important;border-radius:999px!important;color:#747c85!important;font-size:.73rem!important;font-weight:850!important}\n        body:has(.login-page-marker) [data-baseweb="tab"][aria-selected="true"]{background:#142033!important;color:#fff!important;box-shadow:0 7px 18px rgba(20,32,51,.18)!important}\n        body:has(.login-page-marker) div[data-testid="stForm"]{padding:18px 0 0!important;border:0!important;background:transparent!important;box-shadow:none!important}\n        body:has(.login-page-marker) label{color:#343d48!important;font-size:.70rem!important;font-weight:800!important}\n        body:has(.login-page-marker) div[data-baseweb="input"]{min-height:52px!important;background:#f8f9fa!important;border:1px solid #dfe3e7!important;border-radius:17px!important;box-shadow:none!important}\n        body:has(.login-page-marker) div[data-baseweb="input"] input{color:#142033!important;font-size:.85rem!important;font-weight:600!important}\n        body:has(.login-page-marker) div[data-baseweb="input"] input::placeholder{color:#9aa2aa!important;opacity:1!important}\n        body:has(.login-page-marker) div[data-baseweb="input"]:focus-within{background:#fff!important;border-color:#6c8f8d!important;box-shadow:0 0 0 4px rgba(108,143,141,.12)!important}\n        body:has(.login-page-marker) .stSelectbox div[data-baseweb="select"]>div{min-height:52px!important;background:#f8f9fa!important;border:1px solid #dfe3e7!important;border-radius:17px!important;color:#142033!important}\n        body:has(.login-page-marker) div[data-testid="stFormSubmitButton"] button{width:100%!important;min-height:55px!important;margin-top:9px!important;border:0!important;border-radius:999px!important;background:linear-gradient(135deg,#142033,#2c5660)!important;color:#fff!important;font-size:.82rem!important;font-weight:850!important;box-shadow:0 14px 28px rgba(20,32,51,.22)!important;transition:all .2s ease!important}\n        body:has(.login-page-marker) div[data-testid="stFormSubmitButton"] button:hover{transform:translateY(-2px)!important;background:linear-gradient(135deg,#1c3048,#356a70)!important;box-shadow:0 18px 35px rgba(20,32,51,.28)!important}\n        body:has(.login-page-marker) .stAlert{border-radius:16px!important}\n        @media(max-width:900px){body:has(.login-page-marker) .block-container{padding:0 18px 25px!important}body:has(.login-page-marker) .clean-login-left{min-height:auto;padding:35px 0 15px;text-align:center;align-items:center}body:has(.login-page-marker) .clean-login-brand{margin-bottom:35px}body:has(.login-page-marker) .clean-main-title{font-size:3rem!important}body:has(.login-page-marker) .clean-main-description{font-size:.88rem}body:has(.login-page-marker) .clean-benefits{justify-content:center}body:has(.login-page-marker) .clean-card-heading{margin-top:20px}}\n        @media(max-width:520px){body:has(.login-page-marker) .clean-main-title{font-size:2.55rem!important}body:has(.login-page-marker) .clean-card-heading{padding:28px 23px 0;border-radius:27px 27px 0 0}body:has(.login-page-marker) .stTabs{padding:18px 23px 28px!important;border-radius:0 0 27px 27px!important}body:has(.login-page-marker) .clean-benefit{width:100%;justify-content:center}}\n</style>
        """, unsafe_allow_html=True,
    )


PAGE_DETAILS = {
    "Login": ("🚚", "Supplier Recommendation and Risk Analysis System", "MongoDB, Streamlit, analytics, recommendation, and user feedback."),
    "Home": ("🏠", "Home", "Quick overview of supplier performance, product trends, and rating activity."),
    "Upload Data": ("📤", "Upload Data", "Add supplier orders or product trend CSV data into the system."),
    "Clean Data": ("🧹", "Clean Data", "Prepare supplier and product trend data for analysis."),
    "View Data": ("📋", "View Data", "Inspect supplier metrics and product trend records."),
    "EDA & KPI Analysis": ("📈", "EDA & KPI Analysis", "Use charts to understand supplier performance, risk, ratings, and product trends."),
    "Supplier Category Trend & Prediction": ("📊", "Supplier Category Trend & Prediction", "Category-specific supplier analytics, product trends, prediction, and what-if analysis."),
    "Supplier Dashboard": ("📊", "Supplier Dashboard", "Your supplier performance, category demand, feedback, and summary."),
    "Supplier Trend": ("📈", "Trend", "Category-specific product trend, current demand, and up/down trend products."),
    "Future Prediction": ("🔮", "Future Prediction", "Future demand, supplier risk prediction, and what-if improvement analysis."),
    "User Rating": ("⭐", "User Rating", "Review user feedback submitted after supplier experience."),
    "Manage Accounts": ("👥", "Manage Accounts", "Manage user accounts, supplier requests, and supplier verification codes."),
    "Best Suppliers": ("🏠", "Best Suppliers", "Search suppliers quickly or view top supplier options."),
    "Find Supplier": ("🔎", "Find Supplier", "Enter requirements and receive ranked supplier recommendations."),
    "Rate Supplier": ("⭐", "Rate Supplier", "Submit feedback after selecting and using a supplier."),
    "My History": ("🕘", "My History", "View selected suppliers and ratings you submitted."),
}


def page_header(page_key):
    icon, title, subtitle = PAGE_DETAILS.get(page_key, ("📌", page_key, ""))
    st.markdown(
        f"""
        <div class="page-hero">
            <div class="page-hero-top">
                <div class="page-hero-icon">{icon}</div>
                <div style="flex:1">
                    <div class="ui-kicker">SUPPLYLOGIX • AI WORKSPACE</div>
                    <h1>{title}</h1>
                    <p>{subtitle}</p>
                </div>
                <div class="ui-status"><span class="ui-status-dot"></span>Workspace ready</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def table_cell_style(value):
    text = str(value).strip().lower()
    if text == "high":
        return "background-color: #fee2e2; color: #991b1b; font-weight: 700;"
    if text == "medium":
        return "background-color: #fef3c7; color: #92400e; font-weight: 700;"
    if text == "low":
        return "background-color: #dcfce7; color: #166534; font-weight: 700;"
    if text in {"improving", "uptrend", "delivered", "good service", "good recovery"}:
        return "background-color: #dcfce7; color: #166534; font-weight: 700;"
    if text in {"declining", "downtrend", "cancelled", "poor quality", "supply loss"}:
        return "background-color: #fee2e2; color: #991b1b; font-weight: 700;"
    if text in {"stable", "recommended", "selected"}:
        return "background-color: #e0f2fe; color: #075985; font-weight: 700;"
    return ""


def numeric_cell_style(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if number >= 80:
        return "background-color: #dcfce7; color: #166534; font-weight: 700;"
    if number >= 60:
        return "background-color: #e0f2fe; color: #075985; font-weight: 700;"
    if number >= 40:
        return "background-color: #fef3c7; color: #92400e; font-weight: 700;"
    return "background-color: #fee2e2; color: #991b1b; font-weight: 700;"


def rating_cell_style(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if number >= 4:
        return "background-color: #dcfce7; color: #166534; font-weight: 700;"
    if number >= 3:
        return "background-color: #fef3c7; color: #92400e; font-weight: 700;"
    return "background-color: #fee2e2; color: #991b1b; font-weight: 700;"


def delay_cell_style(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if number <= 1:
        return "background-color: #dcfce7; color: #166534; font-weight: 700;"
    if number <= 3:
        return "background-color: #fef3c7; color: #92400e; font-weight: 700;"
    return "background-color: #fee2e2; color: #991b1b; font-weight: 700;"


def style_table(df):
    if df is None or df.empty:
        return df
    styled = df.style.set_properties(
        **{
            "background-color": "#ffffff",
            "color": "#0f172a",
            "border-color": "#e2e8f0",
        }
    )
    styled = styled.set_table_styles(
        [
            {"selector": "thead th", "props": [("background-color", "#eaf0ff"), ("color", "#0f172a"), ("font-weight", "750")]},
            {"selector": "tbody tr:nth-child(even)", "props": [("background-color", "#f8fafc")]},
        ]
    )
    badge_columns = [
        "risk_level",
        "trend_level",
        "trend_status",
        "status",
        "event_type",
        "Level",
    ]
    score_columns = [
        "supplier_rank_score",
        "supplier_kpi_score",
        "final_score",
        "risk_prediction_score",
        "requirement_match_score",
        "trend_score",
        "quality_score",
        "Raw Data Quality",
        "Value",
    ]
    rating_columns = [
        "final_rating",
        "user_rating",
        "risk_handling_rating",
        "quality_rating",
        "rating",
        "recent_rating",
    ]
    delay_columns = ["avg_delay", "max_delay", "delay_days", "delivery_duration"]
    for col in badge_columns:
        if col in df.columns:
            styled = styled.map(table_cell_style, subset=[col])
    for col in score_columns:
        if col in df.columns:
            styled = styled.map(numeric_cell_style, subset=[col])
    for col in rating_columns:
        if col in df.columns:
            styled = styled.map(rating_cell_style, subset=[col])
    for col in delay_columns:
        if col in df.columns:
            styled = styled.map(delay_cell_style, subset=[col])
    return styled


def ui_dataframe(df, width="stretch"):
    st.dataframe(style_table(df), width=width)


RAW_TO_CLEAN_COLUMNS = {
    "Order_ID": "order_id",
    "Buyer_ID": "buyer_id",
    "Supplier_ID": "supplier",
    "Product_Category": "product_category",
    "Quantity_Ordered": "quantity_ordered",
    "Order_Date": "order_date",
    "Dispatch_Date": "dispatch_date",
    "Delivery_Date": "delivery_date",
    "Shipping_Mode": "shipping_mode",
    "Order_Value_USD": "order_value_usd",
    "Delay_Days": "delay_days",
    "Disruption_Type": "disruption_type",
    "Disruption_Severity": "disruption_severity",
    "Historical_Disruption_Count": "historical_disruption_count",
    "Supplier_Reliability_Score": "reliability_score",
    "Organization_ID": "organization_id",
    "Supply_Risk_Flag": "supply_risk_flag",
}

SEVERITY_MAP = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}
EVENT_TYPES = [
    "late delivery",
    "poor quality",
    "supply loss",
    "high cost",
    "good service",
    "good recovery",
]
DATA_EXPORT_KEYS = [
    "raw_orders",
    "cleaned_orders",
    "supplier_metrics",
    "supplier_ratings",
    "hot_suppliers",
    "recommendation_logs",
    "activity_logs",
    "supplier_verification_codes",
]
HOT_SUPPLIERS_COLLECTION = COLLECTIONS.get("hot_suppliers", "hot_suppliers")
AUTH_TOKEN_PARAM = "login_token"
PRODUCT_TRENDS_PATH = Path(__file__).parent / "data" / "product_trends.csv"
PRODUCT_TRENDS_CLEAN_MARKER = Path(__file__).parent / "data" / ".product_trends_cleaned"

PRIORITY_WEIGHTS = {
    "Balanced": {"kpi": 0.30, "rating": 0.20, "risk": 0.25, "match": 0.25},
    "Low Cost": {"kpi": 0.15, "rating": 0.10, "risk": 0.15, "match": 0.60},
    "High Quality": {"kpi": 0.20, "rating": 0.40, "risk": 0.15, "match": 0.25},
    "Fast Delivery": {"kpi": 0.15, "rating": 0.10, "risk": 0.15, "match": 0.60},
    "Low Risk": {"kpi": 0.15, "rating": 0.10, "risk": 0.60, "match": 0.15},
}


def password_hash(password):
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


@st.cache_resource
def get_database():
    db = connect_to_mongodb()
    if db is not None:
        ensure_indexes(db)
    return db


def ensure_default_users(db):
    users = db[COLLECTIONS["users"]]
    default_users = [
        {"username": "admin", "password": "admin123", "role": "admin", "supplier_id": None},
        {"username": "user", "password": "user123", "role": "user", "supplier_id": None},
        {"username": "supplier_s31", "password": "supplier123", "role": "supplier", "supplier_id": "S31"},
        {"username": "supplier_s12", "password": "supplier123", "role": "supplier", "supplier_id": "S12"},
        {"username": "supplier_s10", "password": "supplier123", "role": "supplier", "supplier_id": "S10"},
    ]
    for account in default_users:
        users.update_one(
            {"username": account["username"]},
            {
                "$setOnInsert": {
                    "password_hash": password_hash(account["password"]),
                    "role": account["role"],
                    "supplier_id": account["supplier_id"],
                    "is_active": True,
                    "account_status": "approved",
                    "created_at": datetime.now(timezone.utc),
                }
            },
            upsert=True,
        )


def session_user_from_doc(user_doc):
    return {
        "username": user_doc["username"],
        "role": user_doc["role"],
        "supplier_id": user_doc.get("supplier_id"),
    }


def create_login_session(db, user_doc):
    token = secrets.token_urlsafe(32)
    db[COLLECTIONS["auth_sessions"]].insert_one(
        {
            "token": token,
            "username": user_doc["username"],
            "created_at": datetime.now(timezone.utc),
        }
    )
    st.session_state[AUTH_TOKEN_PARAM] = token
    st.query_params[AUTH_TOKEN_PARAM] = token


def restore_login_session(db):
    token = st.query_params.get(AUTH_TOKEN_PARAM) or st.session_state.get(AUTH_TOKEN_PARAM)
    if not token:
        return False
    session_doc = db[COLLECTIONS["auth_sessions"]].find_one({"token": token}, {"_id": 0})
    if not session_doc:
        return False
    user_doc = db[COLLECTIONS["users"]].find_one({"username": session_doc["username"]}, {"_id": 0})
    if not user_doc:
        return False
    if user_doc.get("role") == "supplier" and user_doc.get("account_status", "pending") != "approved":
        return False
    if not user_doc.get("is_active", False):
        return False
    st.session_state["user"] = session_user_from_doc(user_doc)
    st.session_state[AUTH_TOKEN_PARAM] = token
    st.query_params[AUTH_TOKEN_PARAM] = token
    return True


def clear_login_session(db):
    token = st.query_params.get(AUTH_TOKEN_PARAM) or st.session_state.get(AUTH_TOKEN_PARAM)
    if token:
        db[COLLECTIONS["auth_sessions"]].delete_one({"token": token})
        st.session_state.pop(AUTH_TOKEN_PARAM, None)
    else:
        st.session_state.pop(AUTH_TOKEN_PARAM, None)
    if AUTH_TOKEN_PARAM in st.query_params:
        del st.query_params[AUTH_TOKEN_PARAM]


def supplier_id_info(db, supplier_id, current_username=None):
    supplier_id = supplier_id.strip().upper() if supplier_id else ""
    clean_df = dataframe_from_collection(db, COLLECTIONS["cleaned_orders"], {"supplier": supplier_id})
    raw_df = dataframe_from_collection(db, COLLECTIONS["raw_orders"], {"Supplier_ID": supplier_id})
    source_df = clean_df if not clean_df.empty else raw_df
    categories = []
    if not clean_df.empty and "product_category" in clean_df.columns:
        categories = sorted(clean_df["product_category"].dropna().astype(str).unique())
    elif not raw_df.empty and "Product_Category" in raw_df.columns:
        categories = sorted(raw_df["Product_Category"].dropna().astype(str).unique())
    supplier_exists = not source_df.empty
    order_count = len(source_df)
    claimed_query = {
        "role": "supplier",
        "supplier_id": supplier_id,
        "account_status": "approved",
    }
    if current_username:
        claimed_query["username"] = {"$ne": current_username}
    claimed_account = db[COLLECTIONS["users"]].find_one(claimed_query, {"_id": 0, "username": 1})
    already_claimed = claimed_account is not None
    return {
        "supplier_id": supplier_id,
        "supplier_id_exists": "Yes" if supplier_exists else "No",
        "order_count": order_count,
        "categories": ", ".join(categories) if categories else "None",
        "category_match": "Yes" if categories else "No",
        "already_claimed": "Yes" if already_claimed else "No",
        "claimed_by": claimed_account.get("username", "None") if claimed_account else "None",
    }


def generate_verification_code(supplier_id):
    alphabet = string.ascii_uppercase + string.digits
    suffix = "".join(secrets.choice(alphabet) for _ in range(6))
    return f"{supplier_id.upper()}-{suffix}"


def verification_collection(db):
    return db[COLLECTIONS["supplier_verification_codes"]]


def get_supplier_verification_code(db, supplier_id):
    supplier_id = supplier_id.strip().upper()
    return verification_collection(db).find_one({"supplier_id": supplier_id}, {"_id": 0})


def save_supplier_verification_code(db, supplier_id, actor):
    supplier_id = supplier_id.strip().upper()
    code = generate_verification_code(supplier_id)
    verification_collection(db).update_one(
        {"supplier_id": supplier_id},
        {
            "$set": {
                "supplier_id": supplier_id,
                "verification_code": code,
                "is_used": False,
                "updated_at": datetime.now(timezone.utc),
                "updated_by": actor,
            },
            "$setOnInsert": {"created_at": datetime.now(timezone.utc)},
        },
        upsert=True,
    )
    log_activity(db, "supplier_verification_code_generated", actor, {"supplier_id": supplier_id})
    return code


def validate_supplier_verification_code(db, supplier_id, verification_code):
    record = get_supplier_verification_code(db, supplier_id)
    if not record:
        return False, "No verification code found for this supplier ID. Ask admin to generate one."
    if record.get("is_used"):
        return False, "This verification code was already used."
    if str(record.get("verification_code", "")).strip() != verification_code.strip():
        return False, "Invalid supplier verification code."
    return True, "Verification code is valid."


def create_account(db, username, password, confirm_password, role, supplier_id=None, verification_code=None):
    username = username.strip()
    supplier_id = supplier_id.strip().upper() if supplier_id else None
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    if password != confirm_password:
        return False, "Passwords do not match."
    if role not in {"user", "supplier"}:
        return False, "Only user and supplier signup are allowed."
    if role == "supplier" and not supplier_id:
        return False, "Supplier ID is required for supplier signup."
    if role == "supplier":
        valid_code, message = validate_supplier_verification_code(db, supplier_id, verification_code or "")
        if not valid_code:
            return False, message
        info = supplier_id_info(db, supplier_id)
        if info["supplier_id_exists"] != "Yes":
            return False, "Supplier ID was not found in uploaded supplier data."
        if info["order_count"] <= 0:
            return False, "Supplier ID has no order history."
        if info["category_match"] != "Yes":
            return False, "Supplier ID has no product category match."
        if info["already_claimed"] == "Yes":
            return False, "This supplier ID is already claimed by another approved supplier account."
    users = db[COLLECTIONS["users"]]
    if users.find_one({"username": username}):
        return False, "Username already exists."
    is_supplier = role == "supplier"
    users.insert_one(
        {
            "username": username,
            "password_hash": password_hash(password),
            "role": role,
            "supplier_id": supplier_id,
            "is_active": not is_supplier,
            "account_status": "pending" if is_supplier else "approved",
            "created_at": datetime.now(timezone.utc),
        }
    )
    if is_supplier:
        verification_collection(db).update_one(
            {"supplier_id": supplier_id},
            {"$set": {"is_used": True, "used_by": username, "used_at": datetime.now(timezone.utc)}},
        )
    log_activity(db, "account_created", username, {"role": role, "supplier_id": supplier_id})
    if is_supplier:
        return True, "Supplier request created. Please wait for admin approval."
    return True, "Account created. You can login now."


def load_collection(db, key):
    return dataframe_from_collection(db, COLLECTIONS[key])


def normalize_reliability(series):
    values = pd.to_numeric(series, errors="coerce").fillna(0)
    return np.where(values > 1, values / 100, values)


def clean_orders(raw_df):
    missing = [col for col in RAW_TO_CLEAN_COLUMNS if col not in raw_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    df = raw_df[list(RAW_TO_CLEAN_COLUMNS)].rename(columns=RAW_TO_CLEAN_COLUMNS).copy()
    for col in ["order_date", "dispatch_date", "delivery_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    numeric_columns = [
        "quantity_ordered",
        "order_value_usd",
        "delay_days",
        "historical_disruption_count",
        "reliability_score",
        "supply_risk_flag",
    ]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["supplier"] = df["supplier"].astype(str).str.strip().str.upper()
    df["product_category"] = df["product_category"].astype(str).str.strip().str.title()
    df["shipping_mode"] = df["shipping_mode"].astype(str).str.strip().str.title()
    df["disruption_type"] = df["disruption_type"].fillna("None").astype(str).str.strip().str.title()
    df["disruption_severity"] = df["disruption_severity"].fillna("None").astype(str).str.strip().str.title()

    df = df.drop_duplicates(subset=["order_id"]).dropna(subset=["order_id", "supplier", "product_category"])
    df["quantity_ordered"] = df["quantity_ordered"].fillna(0).clip(lower=0)
    df["order_value_usd"] = df["order_value_usd"].fillna(0).clip(lower=0)
    df["delay_days"] = df["delay_days"].fillna(0).clip(lower=0)
    df["historical_disruption_count"] = df["historical_disruption_count"].fillna(0).clip(lower=0)
    df["reliability_score"] = normalize_reliability(df["reliability_score"])
    df["supply_risk_flag"] = df["supply_risk_flag"].fillna(0).clip(0, 1).astype(int)
    df["delivery_duration"] = (df["delivery_date"] - df["order_date"]).dt.days.fillna(df["delay_days"]).clip(lower=0)
    df["unit_price"] = np.where(df["quantity_ordered"] > 0, df["order_value_usd"] / df["quantity_ordered"], 0)
    df["severity_score"] = df["disruption_severity"].str.upper().map(SEVERITY_MAP).fillna(0)
    df["has_disruption"] = (df["disruption_type"].str.upper() != "NONE").astype(int)
    df["on_time_flag"] = (df["delay_days"] <= 0).astype(int)
    df["quality_score"] = ((df["reliability_score"] * 100) * 0.75) + ((100 - df["severity_score"] * 25) * 0.25)
    df["quality_rating"] = (df["quality_score"] / 20).clip(1, 5).round(2)

    for col in ["order_date", "dispatch_date", "delivery_date"]:
        df[col] = df[col].dt.strftime("%Y-%m-%d")
    return df


def data_quality_summary(raw_df):
    if raw_df.empty:
        return {
            "score": 0,
            "missing_values": 0,
            "duplicate_orders": 0,
            "invalid_dates": 0,
            "total_rows": 0,
        }
    missing_values = int(raw_df.isna().sum().sum())
    duplicate_orders = int(raw_df.duplicated(subset=["Order_ID"]).sum()) if "Order_ID" in raw_df else 0
    invalid_dates = 0
    for col in ["Order_Date", "Dispatch_Date", "Delivery_Date"]:
        if col in raw_df:
            invalid_dates += int(pd.to_datetime(raw_df[col], errors="coerce").isna().sum())
    total_cells = max(raw_df.shape[0] * raw_df.shape[1], 1)
    penalty = (missing_values / total_cells * 45) + (duplicate_orders / max(len(raw_df), 1) * 35) + (invalid_dates / max(len(raw_df), 1) * 20)
    return {
        "score": round(max(0, 100 - penalty), 1),
        "missing_values": missing_values,
        "duplicate_orders": duplicate_orders,
        "invalid_dates": invalid_dates,
        "total_rows": len(raw_df),
    }


def product_trend_quality_summary(raw_df):
    if raw_df.empty:
        return {
            "score": 0,
            "missing_values": 0,
            "duplicate_rows": 0,
            "invalid_months": 0,
            "total_rows": 0,
        }
    missing_values = int(raw_df.isna().sum().sum())
    duplicate_rows = int(raw_df.duplicated().sum())
    invalid_months = 0
    if "month" in raw_df:
        invalid_months = int(pd.to_datetime(raw_df["month"], errors="coerce").isna().sum())
    total_cells = max(raw_df.shape[0] * raw_df.shape[1], 1)
    penalty = (missing_values / total_cells * 45) + (duplicate_rows / max(len(raw_df), 1) * 35) + (invalid_months / max(len(raw_df), 1) * 20)
    return {
        "score": round(max(0, 100 - penalty), 1),
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "invalid_months": invalid_months,
        "total_rows": len(raw_df),
    }


def clean_product_trends(raw_df):
    missing = [col for col in PRODUCT_TREND_REQUIRED_COLUMNS if col not in raw_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    columns = PRODUCT_TREND_REQUIRED_COLUMNS + [col for col in PRODUCT_TREND_OPTIONAL_COLUMNS if col in raw_df.columns]
    df = raw_df[columns].copy()
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    df["product_category"] = df["product_category"].astype(str).str.strip().str.title()
    df["product_name"] = df["product_name"].astype(str).str.strip().str.title()

    for col in ["search_volume", "sales_count", "growth_rate", "trend_score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if "trend_level" not in df.columns:
        df["trend_level"] = np.select(
            [df["trend_score"] >= 75, df["trend_score"] >= 45],
            ["High", "Medium"],
            default="Low",
        )
    else:
        df["trend_level"] = df["trend_level"].astype(str).str.strip().str.title()
        df.loc[~df["trend_level"].isin(["High", "Medium", "Low"]), "trend_level"] = np.select(
            [df["trend_score"] >= 75, df["trend_score"] >= 45],
            ["High", "Medium"],
            default="Low",
        )
    if "data_source" not in df.columns:
        df["data_source"] = "Unknown"
    else:
        df["data_source"] = df["data_source"].fillna("Unknown").astype(str).str.strip()

    df = df.dropna(subset=["month", "product_category", "product_name"])
    df = df.drop_duplicates(subset=["month", "product_category", "product_name"])
    df["search_volume"] = df["search_volume"].clip(lower=0).round(0).astype(int)
    df["sales_count"] = df["sales_count"].clip(lower=0).round(0).astype(int)
    df["trend_score"] = df["trend_score"].clip(0, 100).round(2)
    df["growth_rate"] = df["growth_rate"].round(2)
    df["month"] = df["month"].dt.strftime("%Y-%m")
    return df.sort_values(["product_category", "product_name", "month"]).reset_index(drop=True)


def product_trend_cleaning_is_saved(db):
    if st.session_state.get("cleaned_product_trend_rows", 0) > 0:
        return True
    if PRODUCT_TRENDS_CLEAN_MARKER.exists():
        return True

    last_clean = db[COLLECTIONS["activity_logs"]].find_one(
        {"action": "product_trend_cleaning_completed"},
        sort=[("created_at", -1)],
    )
    if not last_clean:
        return False

    last_dirty = db[COLLECTIONS["activity_logs"]].find_one(
        {"action": {"$in": ["product_trends_updated", "product_trends_updated_from_path", "product_trend_data_deleted"]}},
        sort=[("created_at", -1)],
    )
    if not last_dirty:
        return True
    return bool(last_clean.get("created_at") and last_dirty.get("created_at") and last_clean["created_at"] > last_dirty["created_at"])


def mark_product_trends_cleaned(row_count):
    PRODUCT_TRENDS_CLEAN_MARKER.parent.mkdir(parents=True, exist_ok=True)
    PRODUCT_TRENDS_CLEAN_MARKER.write_text(str(row_count), encoding="utf-8")
    st.session_state["cleaned_product_trend_rows"] = row_count


def mark_product_trends_dirty():
    if PRODUCT_TRENDS_CLEAN_MARKER.exists():
        PRODUCT_TRENDS_CLEAN_MARKER.unlink()
    st.session_state.pop("cleaned_product_trend_rows", None)


def supplier_cleaning_difference_tables(raw_df, clean_df, limit=10):
    if raw_df.empty or clean_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    raw_compare = raw_df[list(RAW_TO_CLEAN_COLUMNS)].rename(columns=RAW_TO_CLEAN_COLUMNS).copy()
    clean_compare = clean_df.copy()
    display_columns = ["order_id", "supplier", "product_category", "disruption_type", "disruption_severity"]
    raw_compare = raw_compare[[col for col in display_columns if col in raw_compare.columns]]
    clean_compare = clean_compare[[col for col in display_columns if col in clean_compare.columns]]
    merged = raw_compare.merge(clean_compare, on="order_id", suffixes=("_before", "_after"), how="inner")

    changed_mask = pd.Series(False, index=merged.index)
    for col in ["disruption_type", "disruption_severity"]:
        before_col = f"{col}_before"
        after_col = f"{col}_after"
        if before_col in merged.columns and after_col in merged.columns:
            before_missing = merged[before_col].isna() | merged[before_col].astype(str).str.strip().eq("")
            changed_value = merged[before_col].fillna("<missing>").astype(str) != merged[after_col].fillna("<missing>").astype(str)
            changed_mask = changed_mask | before_missing | changed_value

    changed = merged[changed_mask].head(limit)
    if changed.empty:
        return pd.DataFrame(), pd.DataFrame()

    before_table = pd.DataFrame(
        {
            "order_id": changed["order_id"],
            "supplier": changed.get("supplier_before", changed.get("supplier_after")),
            "product_category": changed.get("product_category_before", changed.get("product_category_after")),
            "disruption_type": changed["disruption_type_before"].fillna("<missing>"),
            "disruption_severity": changed["disruption_severity_before"].fillna("<missing>"),
        }
    )
    after_table = pd.DataFrame(
        {
            "order_id": changed["order_id"],
            "supplier": changed.get("supplier_after", changed.get("supplier_before")),
            "product_category": changed.get("product_category_after", changed.get("product_category_before")),
            "disruption_type": changed["disruption_type_after"].fillna("<missing>"),
            "disruption_severity": changed["disruption_severity_after"].fillna("<missing>"),
        }
    )
    return before_table.reset_index(drop=True), after_table.reset_index(drop=True)


def fmt_number(value, decimals=2):
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return "0.00"


def risk_level(score):
    if score < 35:
        return "Low"
    if score < 65:
        return "Medium"
    return "High"


def calculate_rating_aggregates(ratings_df):
    if ratings_df.empty:
        return pd.DataFrame(columns=["supplier", "product_category", "user_rating", "rating_count", "bad_feedback_count", "recent_rating"])
    df = ratings_df.copy()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0)
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    agg = (
        df.groupby(["supplier", "product_category"], as_index=False)
        .agg(
            user_rating=("rating", "mean"),
            rating_count=("rating", "count"),
            bad_feedback_count=("rating", lambda x: int((x <= 2).sum())),
            recent_rating=("rating", lambda x: round(x.tail(5).mean(), 2)),
        )
    )
    agg["user_rating"] = agg["user_rating"].round(2)
    return agg


def calculate_supplier_metrics(clean_df, ratings_df=None):
    if clean_df.empty:
        return clean_df

    df = clean_df.copy()
    numeric_cols = [
        "quantity_ordered",
        "order_value_usd",
        "delay_days",
        "historical_disruption_count",
        "reliability_score",
        "supply_risk_flag",
        "delivery_duration",
        "unit_price",
        "severity_score",
        "has_disruption",
        "on_time_flag",
        "quality_score",
        "quality_rating",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    grouped = (
        df.groupby(["supplier", "product_category"], as_index=False)
        .agg(
            total_orders=("order_id", "count"),
            total_quantity=("quantity_ordered", "sum"),
            total_order_value=("order_value_usd", "sum"),
            avg_unit_price=("unit_price", "mean"),
            avg_delay=("delay_days", "mean"),
            max_delay=("delay_days", "max"),
            reliability=("reliability_score", "mean"),
            on_time_delivery_rate=("on_time_flag", "mean"),
            disruption_count=("has_disruption", "sum"),
            disruption_frequency=("has_disruption", "mean"),
            avg_severity=("severity_score", "mean"),
            historical_disruption_count=("historical_disruption_count", "mean"),
            supply_risk_rate=("supply_risk_flag", "mean"),
            quality_rating=("quality_rating", "mean"),
        )
    )

    rating_agg = calculate_rating_aggregates(ratings_df if ratings_df is not None else pd.DataFrame())
    grouped = grouped.merge(rating_agg, on=["supplier", "product_category"], how="left")
    grouped["user_rating"] = grouped["user_rating"].fillna(grouped["quality_rating"]).clip(1, 5)
    grouped["recent_rating"] = grouped["recent_rating"].fillna(grouped["user_rating"]).clip(1, 5)
    grouped["rating_count"] = grouped["rating_count"].fillna(0).astype(int)
    grouped["bad_feedback_count"] = grouped["bad_feedback_count"].fillna(0).astype(int)

    delay_max = max(grouped["avg_delay"].max(), 1)
    history_max = max(grouped["historical_disruption_count"].max(), 1)
    bad_feedback_max = max(grouped["bad_feedback_count"].max(), 1)
    grouped["risk_score"] = (
        ((1 - grouped["reliability"]) * 25)
        + ((grouped["avg_delay"] / delay_max) * 20)
        + (grouped["disruption_frequency"] * 15)
        + (grouped["supply_risk_rate"] * 15)
        + ((grouped["avg_severity"] / 3) * 10)
        + ((5 - grouped["user_rating"]) / 4 * 10)
        + (grouped["bad_feedback_count"] / bad_feedback_max * 5)
    ).clip(0, 100)
    grouped["risk_level"] = grouped["risk_score"].apply(risk_level)
    grouped["risk_handling_rating"] = (5 - grouped["risk_score"] / 25).clip(1, 5)
    grouped["supplier_kpi_score"] = (
        grouped["reliability"] * 35
        + grouped["on_time_delivery_rate"] * 20
        + (grouped["quality_rating"] / 5 * 20)
        + ((100 - grouped["risk_score"]) / 100 * 25)
    ).clip(0, 100)
    grouped["final_rating"] = ((grouped["risk_handling_rating"] * 0.55) + (grouped["user_rating"] * 0.45)).clip(1, 5)
    grouped["trend_status"] = np.select(
        [
            grouped["recent_rating"] >= grouped["user_rating"] + 0.25,
            grouped["recent_rating"] <= grouped["user_rating"] - 0.25,
        ],
        ["Improving", "Declining"],
        default="Stable",
    )
    grouped["supplier_rank_score"] = (
        grouped["supplier_kpi_score"] * 0.55
        + grouped["user_rating"] / 5 * 20
        + (100 - grouped["risk_score"]) * 0.25
    ).clip(0, 100)

    for col in grouped.select_dtypes(include=[np.number]).columns:
        grouped[col] = grouped[col].round(2)
    return grouped.sort_values("supplier_rank_score", ascending=False)


def refresh_metrics(db):
    clean_df = load_collection(db, "cleaned_orders")
    ratings_df = load_collection(db, "supplier_ratings")
    metrics_df = calculate_supplier_metrics(clean_df, ratings_df)
    replace_collection_from_dataframe(db, COLLECTIONS["supplier_metrics"], metrics_df)
    return metrics_df


REQUIRED_METRIC_COLUMNS = {
    "supplier",
    "product_category",
    "supplier_kpi_score",
    "user_rating",
    "risk_score",
    "risk_level",
    "final_rating",
    "supplier_rank_score",
    "trend_status",
}


def load_supplier_metrics(db):
    metrics_df = load_collection(db, "supplier_metrics")
    if not metrics_df.empty and REQUIRED_METRIC_COLUMNS.issubset(metrics_df.columns):
        return metrics_df

    clean_df = load_collection(db, "cleaned_orders")
    if clean_df.empty:
        return pd.DataFrame()
    return refresh_metrics(db)


def safe_metric_table(metrics_df, columns):
    existing = [column for column in columns if column in metrics_df.columns]
    if not existing:
        return pd.DataFrame()
    return metrics_df[existing]


def format_product_trend_display(df):
    formatted = df.copy()
    for col in ["trend_score", "growth_rate", "predicted_next_trend_score", "trend_score_change"]:
        if col in formatted.columns:
            formatted[col] = pd.to_numeric(formatted[col], errors="coerce").fillna(0).round(2)
    for col in ["search_volume", "sales_count"]:
        if col in formatted.columns:
            formatted[col] = pd.to_numeric(formatted[col], errors="coerce").fillna(0).round(0).astype(int)
    return formatted


PRODUCT_TREND_REQUIRED_COLUMNS = [
    "month",
    "product_category",
    "product_name",
    "search_volume",
    "sales_count",
    "growth_rate",
    "trend_score",
]
PRODUCT_TREND_OPTIONAL_COLUMNS = ["trend_level", "data_source"]


def validate_product_trend_csv(trend_df):
    missing = set(PRODUCT_TREND_REQUIRED_COLUMNS) - set(trend_df.columns)
    if missing:
        return False, f"Missing required columns: {', '.join(sorted(missing))}"
    for col in ["search_volume", "sales_count", "growth_rate", "trend_score"]:
        trend_df[col] = pd.to_numeric(trend_df[col], errors="coerce")
    invalid_numeric = trend_df[["search_volume", "sales_count", "growth_rate", "trend_score"]].isna().sum().sum()
    if invalid_numeric:
        return False, "Trend CSV has invalid numeric values."
    return True, "Trend CSV is valid."


def explain_supplier(row, budget=None, deadline=None):
    reasons = []
    if row.get("reliability", 0) >= 0.8:
        reasons.append("high reliability")
    if row.get("avg_delay", 99) <= 2:
        reasons.append("low average delay")
    if row.get("risk_level") == "Low":
        reasons.append("low predicted supply risk")
    if row.get("user_rating", 0) >= 4:
        reasons.append("strong user rating")
    if budget and row.get("avg_unit_price", 0) <= budget:
        reasons.append("fits the budget")
    if deadline and row.get("avg_delay", 0) <= deadline:
        reasons.append("fits the deadline")
    if row.get("trend_status") == "Declining":
        reasons.append("but recent feedback is declining")
    return "Recommended because it has " + ", ".join(reasons or ["acceptable overall performance"]) + "."


def explain_best_requirement_match(row, brief, result_count):
    """Render the best-match comparison using Streamlit native components.

    This intentionally avoids injecting the table as raw HTML because Streamlit
    can display HTML literally depending on the rendering context/version.
    """
    budget = float(brief.get("budget", 0) or 0)
    quantity = int(brief.get("quantity", 0) or 0)
    min_quality = float(brief.get("quality", 0) or 0)
    deadline = float(brief.get("deadline", 0) or 0)
    requested_count = int(brief.get("requested_supplier_count", result_count) or result_count)

    supplier = str(row.get("supplier", "Selected supplier"))
    category = str(row.get("product_category", brief.get("category", "selected category")))
    supplier_price = float(row.get("avg_unit_price", 0) or 0)
    supplier_quantity = int(float(row.get("total_quantity", 0) or 0))
    supplier_quality = float(row.get("quality_rating", row.get("final_rating", 0)) or 0)
    supplier_delay = float(row.get("avg_delay", 0) or 0)
    final_score = float(row.get("final_score", 0) or 0)
    risk_level_value = str(row.get("risk_level", "N/A"))

    budget_status = "Fits your budget" if budget <= 0 or supplier_price <= budget else "Above your budget"
    quantity_status = "Capacity matched" if supplier_quantity >= quantity else "Lower historical capacity"
    quality_status = "Quality matched" if supplier_quality >= min_quality else "Below quality target"
    deadline_status = "Delivery matched" if supplier_delay <= deadline else "Slower than deadline"

    # Summary card: native Streamlit text, so it can never fall back to raw HTML.
    st.markdown(
        f"### Why **{supplier}**?"
        f"  \nBest match for **{category}** · **{final_score:.0f}/100** match score · **{risk_level_value}** risk"
    )

    comparison = pd.DataFrame(
        [
            ["Budget per unit", f"${budget:,.2f}", f"${supplier_price:,.2f}"],
            ["Quantity", f"{quantity:,}", f"{supplier_quantity:,} handled"],
            ["Quality rating", f"{min_quality:.1f}/5", f"{supplier_quality:.1f}/5"],
            ["Delivery delay", f"{deadline:.0f} days max", f"{supplier_delay:.1f} days avg"],
        ],
        columns=["REQUIREMENT", "YOUR TARGET", "SUPPLIER"],
    )

    # Pandas Styler gives us a real HTML table through st.table, not literal HTML text.
    styled = (
        comparison.style
        .hide(axis="index")
        .set_properties(
            subset=["REQUIREMENT"],
            **{"font-weight": "600", "color": "#26324a", "text-align": "left"},
        )
        .set_properties(
            subset=["YOUR TARGET"],
            **{"font-weight": "700", "color": "#5f6878", "text-align": "right"},
        )
        .set_properties(
            subset=["SUPPLIER"],
            **{"font-weight": "800", "color": "#2f7658", "text-align": "right", "background-color": "#f1faf5"},
        )
        .set_table_styles(
            [
                {"selector": "table", "props": [
                    ("width", "100%"), ("border-collapse", "separate"),
                    ("border-spacing", "0"), ("border", "1px solid #dfe3e8"),
                    ("border-radius", "18px"), ("overflow", "hidden"),
                    ("background", "#ffffff"), ("box-shadow", "0 14px 35px rgba(25,35,55,.07)"),
                ]},
                {"selector": "thead th", "props": [
                    ("background", "#17213b"), ("color", "#ffffff"),
                    ("font-size", "12px"), ("font-weight", "800"),
                    ("letter-spacing", "1.2px"), ("padding", "15px 18px"),
                    ("border", "none"), ("text-align", "right"),
                ]},
                {"selector": "thead th:first-child", "props": [("text-align", "left")]},
                {"selector": "tbody td", "props": [
                    ("padding", "17px 18px"), ("border-top", "1px solid #e8ebef"),
                    ("font-size", "14px"), ("background", "#ffffff"),
                ]},
                {"selector": "tbody tr:hover td", "props": [("background", "#faf9ff")]},
            ]
        )
    )
    st.table(styled)

    # Match status card using native columns.
    c1, c2, c3 = st.columns([0.75, 3.0, 0.85])
    with c1:
        st.markdown("**✓**  \n*BEST MATCH*")
    with c2:
        st.markdown(f"**{budget_status} · {quantity_status}**")
        st.caption(f"{quality_status} · {deadline_status} · Showing {result_count}/{requested_count} supplier(s)")
    with c3:
        st.markdown(f"### {final_score:.0f}/100")

# Columns required by the supplier recommendation engine.
# Keep this definition before validate_recommendation_input() so the
# Find Supplier page can validate the metrics dataframe without a NameError.
RECOMMENDATION_REQUIRED_COLUMNS = {
    "supplier",
    "product_category",
    "supplier_kpi_score",
    "user_rating",
    "risk_score",
    "quality_rating",
    "avg_delay",
    "avg_unit_price",
    "total_quantity",
}

def validate_recommendation_input(metrics_df, priority):
    if metrics_df.empty:
        return False, "No supplier metrics available."
    missing_columns = RECOMMENDATION_REQUIRED_COLUMNS - set(metrics_df.columns)
    if missing_columns:
        return False, f"Missing recommendation columns: {', '.join(sorted(missing_columns))}"
    if priority not in PRIORITY_WEIGHTS:
        return False, "Unknown recommendation priority."
    return True, "Recommendation input is valid."


REQUIREMENT_MATCH_WEIGHTS = {
    "Balanced": {"cost": 0.25, "deadline": 0.25, "quality": 0.25, "quantity": 0.25},
    "Low Cost": {"cost": 0.55, "deadline": 0.15, "quality": 0.15, "quantity": 0.15},
    "High Quality": {"cost": 0.15, "deadline": 0.15, "quality": 0.55, "quantity": 0.15},
    "Fast Delivery": {"cost": 0.15, "deadline": 0.55, "quality": 0.15, "quantity": 0.15},
    "Low Risk": {"cost": 0.15, "deadline": 0.20, "quality": 0.20, "quantity": 0.45},
}


PRIORITY_SORT_COLUMNS = {
    "Balanced": "requirement_match_score",
    "Low Cost": "cost_match",
    "High Quality": "quality_match",
    "Fast Delivery": "deadline_match",
    "Low Risk": "risk_prediction_score",
}


def filter_supplier_options(metrics_df, category, budget, min_quality, deadline, quantity=None):
    options = metrics_df[metrics_df["product_category"].str.lower() == category.lower()].copy()
    if options.empty:
        return options

    for col in ["avg_unit_price", "quality_rating", "avg_delay", "total_quantity"]:
        options[col] = pd.to_numeric(options[col], errors="coerce").fillna(0)
    if budget > 0:
        options = options[options["avg_unit_price"] <= budget]
    options = options[(options["quality_rating"] >= min_quality) & (options["avg_delay"] <= deadline)]
    if quantity and not options.empty:
        capable_options = options[options["total_quantity"] >= quantity]
        if not capable_options.empty:
            return capable_options
    return options


def calculate_requirement_match(options, quantity, budget, deadline, priority):
    scored = options.copy()
    scored["cost_match"] = np.where(budget > 0, ((budget - scored["avg_unit_price"]) / budget * 100).clip(0, 100), 70)
    scored["deadline_match"] = ((deadline - scored["avg_delay"]) / max(deadline, 1) * 100).clip(0, 100)
    scored["quality_match"] = (scored["quality_rating"] / 5 * 100).clip(0, 100)
    scored["quantity_match"] = np.where(scored["total_quantity"] >= quantity, 100, scored["total_quantity"] / max(quantity, 1) * 100)
    match_weights = REQUIREMENT_MATCH_WEIGHTS.get(priority, REQUIREMENT_MATCH_WEIGHTS["Balanced"])
    scored["requirement_match_score"] = (
        scored["cost_match"] * match_weights["cost"]
        + scored["deadline_match"] * match_weights["deadline"]
        + scored["quality_match"] * match_weights["quality"]
        + scored["quantity_match"] * match_weights["quantity"]
    ).round(2)
    return scored


def calculate_recommendation_scores(options, priority):
    scored = options.copy()
    scored["user_rating_score"] = (scored["user_rating"] / 5 * 100).round(2)
    scored["risk_prediction_score"] = (100 - scored["risk_score"]).round(2)

    weights = PRIORITY_WEIGHTS[priority]
    scored["final_score"] = (
        scored["supplier_kpi_score"] * weights["kpi"]
        + scored["user_rating_score"] * weights["rating"]
        + scored["risk_prediction_score"] * weights["risk"]
        + scored["requirement_match_score"] * weights["match"]
    ).round(2)
    return scored


def rank_recommendations(options, top_n, priority):
    sort_columns = ["final_score"]
    priority_column = PRIORITY_SORT_COLUMNS.get(priority)
    if priority_column in options.columns and priority_column not in sort_columns:
        sort_columns.append(priority_column)
    if "requirement_match_score" in options.columns and "requirement_match_score" not in sort_columns:
        sort_columns.append("requirement_match_score")
    return options.sort_values(sort_columns, ascending=[False] * len(sort_columns)).head(top_n)


def recommend_suppliers(metrics_df, category, quantity, budget, min_quality, deadline, priority, top_n):
    is_valid, _ = validate_recommendation_input(metrics_df, priority)
    if not is_valid:
        return pd.DataFrame()

    options = filter_supplier_options(metrics_df, category, budget, min_quality, deadline, quantity)
    if options.empty:
        return options

    options = calculate_requirement_match(options, quantity, budget, deadline, priority)
    options = calculate_recommendation_scores(options, priority)
    options["explanation"] = options.apply(lambda row: explain_supplier(row, budget, deadline), axis=1)
    return rank_recommendations(options, top_n, priority)


def build_backup_zip(db):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for key in DATA_EXPORT_KEYS:
            df = load_collection(db, key)
            archive.writestr(f"{key}.csv", df.to_csv(index=False))
    buffer.seek(0)
    return buffer.getvalue()


def build_product_trend_backup_zip():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        if PRODUCT_TRENDS_PATH.exists():
            archive.write(PRODUCT_TRENDS_PATH, arcname="product_trends.csv")
        else:
            archive.writestr("product_trends.csv", "")
    buffer.seek(0)
    return buffer.getvalue()


def hot_supplier_keys(db, username):
    hot_df = dataframe_from_collection(db, HOT_SUPPLIERS_COLLECTION, {"username": username})
    if hot_df.empty or not {"supplier", "product_category"}.issubset(hot_df.columns):
        return set()
    return set(zip(hot_df["supplier"].astype(str), hot_df["product_category"].astype(str)))


def save_hot_supplier(db, username, row):
    doc = {
        "username": username,
        "supplier": row["supplier"],
        "product_category": row["product_category"],
        "final_score": row.get("final_score", row.get("supplier_rank_score")),
        "final_rating": row.get("final_rating"),
        "risk_level": row.get("risk_level"),
        "risk_score": row.get("risk_score"),
        "avg_delay": row.get("avg_delay"),
        "avg_unit_price": row.get("avg_unit_price"),
        "explanation": row.get("explanation"),
        "created_at": datetime.now(timezone.utc),
    }
    db[HOT_SUPPLIERS_COLLECTION].update_one(
        {"username": username, "supplier": row["supplier"], "product_category": row["product_category"]},
        {"$set": doc},
        upsert=True,
    )


def save_selected_supplier(db, username, row, source):
    selection = {
        "username": username,
        "supplier": row["supplier"],
        "product_category": row["product_category"],
        "final_score": row.get("final_score", row.get("supplier_rank_score")),
        "risk_level": row.get("risk_level"),
        "status": "selected",
        "source": source,
        "created_at": datetime.now(timezone.utc),
    }
    db[COLLECTIONS["recommendation_logs"]].insert_one(selection)
    log_activity(db, f"supplier_selected_from_{source}", username, selection)
    st.session_state["selected_supplier"] = selection
    return selection


def sync_supplier_saved_scores(db, supplier, product_category, metrics_df):
    if metrics_df.empty:
        return
    match = metrics_df[(metrics_df["supplier"] == supplier) & (metrics_df["product_category"] == product_category)]
    if match.empty:
        return
    row = match.iloc[0]
    updated_values = {
        "final_score": row.get("supplier_rank_score"),
        "final_rating": row.get("final_rating"),
        "risk_level": row.get("risk_level"),
        "risk_score": row.get("risk_score"),
        "avg_delay": row.get("avg_delay"),
        "avg_unit_price": row.get("avg_unit_price"),
    }
    db[HOT_SUPPLIERS_COLLECTION].update_many(
        {"supplier": supplier, "product_category": product_category},
        {"$set": updated_values},
    )
    db[COLLECTIONS["recommendation_logs"]].update_many(
        {"supplier": supplier, "product_category": product_category, "status": "selected"},
        {"$set": {"final_score": row.get("supplier_rank_score"), "risk_level": row.get("risk_level")}},
    )


def remove_hot_supplier(db, username, supplier, product_category):
    db[HOT_SUPPLIERS_COLLECTION].delete_one(
        {"username": username, "supplier": supplier, "product_category": product_category}
    )


def page_login(db):
    """Clean modern SupplyLogix login and account creation page."""
    st.markdown('<div class="login-page-marker"></div>', unsafe_allow_html=True)

    left, right = st.columns([1.12, 0.88], gap="large")

    with left:
        st.markdown("""
        <div class="clean-login-left">
            <div class="clean-login-brand">
                <div class="clean-brand-logo">🚚</div>
                <div>
                    <div class="clean-brand-name">Supply<span>Logix</span></div>
                    <div class="clean-brand-subtitle">AI SUPPLIER INTELLIGENCE</div>
                </div>
            </div>
            <div class="clean-eyebrow">✦ SMART RECOMMENDATION PLATFORM</div>
            <h1 class="clean-main-title">Find smarter.<br><span>Buy with confidence.</span></h1>
            <p class="clean-main-description">
                Turn supplier data into better procurement decisions. Compare performance,
                detect risk, and discover the strongest supplier match for every requirement.
            </p>
            <div class="clean-benefits">
                <div class="clean-benefit"><div class="benefit-icon">✓</div><b>Smart Supplier Matching</b></div>
                <div class="clean-benefit"><div class="benefit-icon">✓</div><b>Risk Analysis</b></div>
                <div class="clean-benefit"><div class="benefit-icon">✓</div><b>Performance Insights</b></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown("""
        <div class="clean-card-heading">
            <div class="clean-card-icon">✦</div>
            <h2>Welcome back</h2>
            <p>Sign in to continue to your supplier intelligence workspace.</p>
        </div>
        """, unsafe_allow_html=True)

        login_tab, signup_tab = st.tabs(["Sign in", "Create account"])

        with login_tab:
            with st.form("login_form", clear_on_submit=False):
                username = st.text_input("Username", placeholder="Enter your username", key="login_username")
                password = st.text_input("Password", type="password", placeholder="Enter your password", key="login_password")
                submitted = st.form_submit_button("Sign in  →", use_container_width=True)

            if submitted:
                user = db[COLLECTIONS["users"]].find_one({
                    "username": username.strip(),
                    "password_hash": password_hash(password),
                })
                if not user:
                    st.error("Invalid username or password.")
                elif user.get("role") == "supplier" and user.get("account_status", "pending") != "approved":
                    st.warning("Your supplier account is waiting for admin approval.")
                elif not user.get("is_active", False):
                    st.error("This account is inactive. Please contact admin.")
                else:
                    st.session_state["user"] = session_user_from_doc(user)
                    create_login_session(db, user)
                    st.rerun()

        with signup_tab:
            with st.form("signup_form", clear_on_submit=False):
                new_username = st.text_input("Username", placeholder="Choose a username", key="signup_username")
                new_password = st.text_input("Password", type="password", placeholder="Minimum 6 characters", key="signup_password")
                confirm_password = st.text_input("Confirm password", type="password", placeholder="Re-enter password", key="signup_confirm_password")
                role = st.selectbox("Account type", ["user", "supplier"], key="signup_role")
                supplier_id = ""
                verification_code = ""
                if role == "supplier":
                    supplier_id = st.text_input("Supplier ID", placeholder="Example: S31", key="signup_supplier_id")
                    verification_code = st.text_input("Verification code", placeholder="Example: S10-ABC123", key="signup_verification_code")
                signup_submitted = st.form_submit_button("Create account  →", use_container_width=True)

            if signup_submitted:
                success, message = create_account(
                    db, new_username, new_password, confirm_password,
                    role, supplier_id, verification_code
                )
                if success:
                    st.success(message)
                else:
                    st.error(message)
def alert_rows(metrics_df):
    if metrics_df.empty:
        return []
    alerts = []
    for _, row in metrics_df.iterrows():
        label = f"{row['supplier']} - {row['product_category']}"
        if row["user_rating"] < 3:
            alerts.append({"Alert": f"{label} rating dropped below 3.0", "Level": "High"})
        if row["risk_level"] == "High":
            alerts.append({"Alert": f"{label} predicted risk is High", "Level": "High"})
        if row["trend_status"] == "Declining":
            alerts.append({"Alert": f"{label} feedback trend is declining", "Level": "Medium"})
    return alerts[:10]


def page_admin_dashboard(db):
    page_header("Home")
    raw_df = load_collection(db, "raw_orders")
    clean_df = load_collection(db, "cleaned_orders")
    metrics_df = load_supplier_metrics(db)
    ratings_df = load_collection(db, "supplier_ratings")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Orders", len(raw_df))
    c2.metric("Suppliers", clean_df["supplier"].nunique() if not clean_df.empty else 0)
    c3.metric("Categories", clean_df["product_category"].nunique() if not clean_df.empty else 0)
    c4.metric("High Risk", int((metrics_df["risk_level"] == "High").sum()) if not metrics_df.empty else 0)
    c5.metric("Avg Rating", round(metrics_df["final_rating"].mean(), 2) if not metrics_df.empty else 0)

    quality = data_quality_summary(raw_df)
    st.subheader("Raw Data Quality Preview")
    st.caption("These checks are calculated from uploaded raw data. Cleaned rows appear after running cleaning.")
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Raw Rows", len(raw_df))
    sc2.metric("Cleaned Rows", len(clean_df))
    sc3.metric("Raw Data Quality", f"{quality['score']}/100")
    sc4.metric("Missing Values", quality["missing_values"])
    if len(raw_df) > 0:
        st.caption(f"Duplicates: {quality['duplicate_orders']} | Invalid dates: {quality['invalid_dates']}")

    alerts = alert_rows(metrics_df)
    if alerts:
        st.subheader("Admin Alerts")
        ui_dataframe(pd.DataFrame(alerts), width="stretch")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Supplier Summary")
        if not metrics_df.empty:
            ui_dataframe(
                safe_metric_table(
                    metrics_df,
                    ["supplier", "product_category", "total_orders", "final_rating", "user_rating", "risk_level", "avg_delay", "trend_status"],
                ).head(12),
                width="stretch",
            )
        else:
            st.info("No supplier metrics yet. Run cleaning first.")

    with col2:
        st.subheader("Product Trend Summary")
        trends_df = load_product_trends()
        if not trends_df.empty:
            latest_month = trends_df["month"].max()
            latest_trends = trends_df[trends_df["month"] == latest_month].copy()
            ui_dataframe(
                safe_metric_table(
                    latest_trends.sort_values("trend_score", ascending=False),
                    ["product_category", "product_name", "trend_level", "growth_rate", "trend_score", "data_source"],
                ).head(12),
                width="stretch",
            )
        else:
            st.info("No product trend data yet. Upload product trend data first.")

    if st.button("Open Data Page"):
        st.session_state["admin_nav_target"] = "View Data"
        st.rerun()

    st.subheader("Rating Activity")
    r1, r2 = st.columns(2)
    r1.metric("User Ratings", len(ratings_df))
    bad_feedback = 0
    if not ratings_df.empty and "rating" in ratings_df.columns:
        bad_feedback = int((pd.to_numeric(ratings_df["rating"], errors="coerce") <= 2).sum())
    r2.metric("Bad Feedback", bad_feedback)


def page_upload(db):
    page_header("Upload Data")
    mode_col1, mode_col2 = st.columns(2)
    if mode_col1.button("Supplier", use_container_width=True):
        st.session_state["upload_mode"] = "Supplier"
    if mode_col2.button("Product Trend", use_container_width=True):
        st.session_state["upload_mode"] = "Product Trend"
    upload_mode = st.session_state.get("upload_mode", "Supplier")

    if upload_mode == "Supplier":
        st.subheader("Supplier")
        st.caption("Uploads append new rows. Existing order IDs are skipped.")
        uploaded = st.file_uploader("Upload supplier order CSV", type=["csv"])
        if uploaded:
            try:
                raw_df = pd.read_csv(uploaded, encoding="ISO-8859-1")
                ui_dataframe(raw_df.head(20), width="stretch")
                if st.button("Append CSV to MongoDB"):
                    inserted, skipped = append_dataframe_unique(db, COLLECTIONS["raw_orders"], raw_df, "Order_ID")
                    log_activity(db, "raw_data_appended", st.session_state["user"]["username"], {"inserted": inserted, "skipped_duplicates": skipped})
                    st.success(f"Added {inserted} new rows. Skipped {skipped} duplicate or invalid rows.")
            except Exception as exc:
                st.error(f"Could not read/upload this CSV: {exc}")

        st.divider()
        st.subheader("Load CSV From Local Path")
        st.caption("Use this if the browser file uploader disconnects. It reads the CSV directly from your computer path.")
        default_test_path = r"C:\Users\harle\Documents\Codex\2026-08-05\ana\outputs\supplier_cleaning_upload_test_10_rows.csv"
        local_csv_path = st.text_input("CSV file path", value=default_test_path)
        if st.button("Append"):
            try:
                raw_df = pd.read_csv(local_csv_path, encoding="ISO-8859-1")
                inserted, skipped = append_dataframe_unique(db, COLLECTIONS["raw_orders"], raw_df, "Order_ID")
                log_activity(
                    db,
                    "local_csv_appended",
                    st.session_state["user"]["username"],
                    {"path": local_csv_path, "inserted": inserted, "skipped_duplicates": skipped},
                )
                st.success(f"Added {inserted} new rows. Skipped {skipped} duplicate or invalid rows.")
                ui_dataframe(raw_df.head(20), width="stretch")
            except Exception as exc:
                st.error(f"Could not load this local CSV path: {exc}")
        return

    st.subheader("Product Trend")
    st.caption("Updates the Supplier Dashboard product trend analysis. Source can be Google Trends CSV summarized into this format.")

    template_df = pd.DataFrame(
        [
            {
                "month": "2026-08",
                "product_category": "Machinery",
                "product_name": "Industrial Robot",
                "search_volume": 61000,
                "sales_count": 6300,
                "growth_rate": 18,
                "trend_score": 88,
                "trend_level": "High",
                "data_source": "Google Trends",
            }
        ]
    )
    st.download_button(
        "Download Product Trend CSV Template",
        template_df.to_csv(index=False),
        "product_trends_template.csv",
        "text/csv",
    )

    uploaded_trend = st.file_uploader("Upload product trend CSV", type=["csv"], key="product_trend_uploader")
    if uploaded_trend:
        try:
            trend_df = pd.read_csv(uploaded_trend)
            valid, message = validate_product_trend_csv(trend_df)
            if not valid:
                st.error(message)
                return
            st.success(message)
            st.subheader("Product Trend Preview")
            ui_dataframe(trend_df.head(30), width="stretch")
            if st.button("Save Product Trend Dataset"):
                PRODUCT_TRENDS_PATH.parent.mkdir(parents=True, exist_ok=True)
                trend_df.to_csv(PRODUCT_TRENDS_PATH, index=False)
                mark_product_trends_dirty()
                log_activity(
                    db,
                    "product_trends_updated",
                    st.session_state["user"]["username"],
                    {"rows": len(trend_df), "categories": int(trend_df["product_category"].nunique())},
                )
                st.success(f"Saved {len(trend_df)} product trend rows. Supplier Dashboard now uses this dataset.")
        except Exception as exc:
            st.error(f"Could not read product trend CSV: {exc}")

    st.divider()
    st.subheader("Load Product Trend CSV From Local Path")
    st.caption("Use this if the browser uploader disconnects. This replaces the current product trend dataset.")
    default_trend_path = str(PRODUCT_TRENDS_PATH)
    trend_csv_path = st.text_input("Product trend CSV file path", value=default_trend_path)
    if st.button("Save Product Trend From Local Path"):
        try:
            trend_df = pd.read_csv(trend_csv_path)
            valid, message = validate_product_trend_csv(trend_df)
            if not valid:
                st.error(message)
                return
            PRODUCT_TRENDS_PATH.parent.mkdir(parents=True, exist_ok=True)
            trend_df.to_csv(PRODUCT_TRENDS_PATH, index=False)
            mark_product_trends_dirty()
            log_activity(
                db,
                "product_trends_updated_from_path",
                st.session_state["user"]["username"],
                {"path": trend_csv_path, "rows": len(trend_df), "categories": int(trend_df["product_category"].nunique())},
            )
            st.success(f"Saved {len(trend_df)} product trend rows from local path.")
            ui_dataframe(trend_df.head(30), width="stretch")
        except Exception as exc:
            st.error(f"Could not load product trend local CSV path: {exc}")
    return



def page_clean(db):
    page_header("Clean Data")

    mode_col1, mode_col2 = st.columns(2)
    if mode_col1.button("Supplier", use_container_width=True):
        st.session_state["clean_data_mode"] = "Supplier"
    if mode_col2.button("Product Trend", use_container_width=True):
        st.session_state["clean_data_mode"] = "Product Trend"
    clean_mode = st.session_state.get("clean_data_mode", "Supplier")

    if clean_mode == "Product Trend":
        st.subheader("Product Trend")
        if not PRODUCT_TRENDS_PATH.exists():
            st.warning("Upload product trend data first.")
            return

        raw_trend_df = pd.read_csv(PRODUCT_TRENDS_PATH)
        quality = product_trend_quality_summary(raw_trend_df)
        product_trend_is_cleaned = product_trend_cleaning_is_saved(db)
        cleaned_trend_rows = len(raw_trend_df) if product_trend_is_cleaned else 0
        st.subheader("Raw Product Trend Quality Preview")
        st.caption("These checks are calculated from uploaded raw product trend data. Cleaned rows appear after running cleaning.")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Raw Rows", len(raw_trend_df))
        c2.metric("Cleaned Rows", cleaned_trend_rows)
        c3.metric("Raw Data Quality", f"{quality['score']}/100")
        c4.metric("Missing Values", quality["missing_values"])
        c5.metric("Duplicate Rows", quality["duplicate_rows"])
        st.caption(f"Invalid months: {quality['invalid_months']}")

        st.subheader("Column Quality")
        ui_dataframe(
            pd.DataFrame(
                {
                    "column": raw_trend_df.columns,
                    "missing_values": raw_trend_df.isna().sum().values,
                    "missing_percent": (raw_trend_df.isna().mean().values * 100).round(2),
                }
            ),
            width="stretch",
        )

        if st.button("Run Product Trend Cleaning"):
            try:
                cleaned_trend_df = clean_product_trends(raw_trend_df)
                PRODUCT_TRENDS_PATH.parent.mkdir(parents=True, exist_ok=True)
                cleaned_trend_df.to_csv(PRODUCT_TRENDS_PATH, index=False)
                log_activity(
                    db,
                    "product_trend_cleaning_completed",
                    st.session_state["user"]["username"],
                    {"records": len(cleaned_trend_df), "quality_score": quality["score"]},
                )
                mark_product_trends_cleaned(len(cleaned_trend_df))
                st.success(f"Cleaned {len(cleaned_trend_df)} product trend rows.")
            except Exception as exc:
                st.error(str(exc))

        if product_trend_cleaning_is_saved(db):
            saved_trend_df = pd.read_csv(PRODUCT_TRENDS_PATH)
            if not saved_trend_df.empty:
                st.subheader("Cleaned Product Trend Table")
                st.caption("This cleaned product trend table appears after cleaning and is used by the Supplier Dashboard.")
                ui_dataframe(saved_trend_df, width="stretch")
        return

    st.subheader("Supplier")
    raw_df = load_collection(db, "raw_orders")
    if raw_df.empty:
        st.warning("Upload supplier data first.")
        return

    quality = data_quality_summary(raw_df)
    cleaned_rows = db[COLLECTIONS["cleaned_orders"]].count_documents({})
    st.subheader("Raw Supplier Data Quality Preview")
    st.caption("These checks are calculated from uploaded raw supplier data. Cleaned rows appear after running cleaning.")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Raw Rows", len(raw_df))
    c2.metric("Cleaned Rows", cleaned_rows)
    c3.metric("Raw Data Quality", f"{quality['score']}/100")
    c4.metric("Missing Values", quality["missing_values"])
    c5.metric("Duplicate Orders", quality["duplicate_orders"])
    st.caption(f"Invalid dates: {quality['invalid_dates']}")

    st.subheader("Column Quality")
    ui_dataframe(
        pd.DataFrame(
            {
                "column": raw_df.columns,
                "missing_values": raw_df.isna().sum().values,
                "missing_percent": (raw_df.isna().mean().values * 100).round(2),
            }
        ),
        width="stretch",
    )
    if st.button("Run Supplier Cleaning and Refresh Metrics"):
        try:
            clean_df = clean_orders(raw_df)
            count = replace_collection_from_dataframe(db, COLLECTIONS["cleaned_orders"], clean_df)
            metrics_df = refresh_metrics(db)
            log_activity(db, "cleaning_completed", st.session_state["user"]["username"], {"records": count, "quality_score": quality["score"]})
            st.success(f"Cleaned {count} supplier order rows and refreshed {len(metrics_df)} supplier metric rows.")
        except Exception as exc:
            st.error(str(exc))

    saved_clean_df = load_collection(db, COLLECTIONS["cleaned_orders"])
    if not saved_clean_df.empty:
        st.subheader("Cleaned Supplier Data Table")
        st.caption("This cleaned supplier order table stays visible after rerun because it is saved in MongoDB.")
        ui_dataframe(saved_clean_df, width="stretch")

        before_diff, after_diff = supplier_cleaning_difference_tables(raw_df, saved_clean_df)
        if not before_diff.empty and not after_diff.empty:
            st.subheader("Before Cleaning vs After Cleaning Summary")
            st.caption("Only rows with visible cleaning changes are shown. Missing disruption values are filled as None after cleaning.")
            before_col, after_col = st.columns(2)
            with before_col:
                st.markdown("**Before Cleaning**")
                ui_dataframe(before_diff, width="stretch")
            with after_col:
                st.markdown("**After Cleaning**")
                ui_dataframe(after_diff, width="stretch")
    return


def page_view_data(db):
    page_header("View Data")
    mode_col1, mode_col2 = st.columns(2)
    if mode_col1.button("Supplier", use_container_width=True):
        st.session_state["view_data_mode"] = "Supplier"
    if mode_col2.button("Product Trend", use_container_width=True):
        st.session_state["view_data_mode"] = "Product Trend"
    view_mode = st.session_state.get("view_data_mode", "Supplier")

    if view_mode == "Product Trend":
        trends_df = load_product_trends()
        st.subheader("Product Trend")
        if trends_df.empty:
            st.info("No product trend dataset found. Use Upload Data > Product Trend first.")
            return
        c1, c2, c3 = st.columns(3)
        category = c1.selectbox("Category", ["All"] + sorted(trends_df["product_category"].dropna().unique()), key="view_product_category")
        product = c2.selectbox("Product", ["All"] + sorted(trends_df["product_name"].dropna().unique()), key="view_product_name")
        trend_level_options = ["All"] + sorted(trends_df["trend_level"].dropna().unique()) if "trend_level" in trends_df.columns else ["All"]
        trend_level = c3.selectbox("Trend Level", trend_level_options, key="view_product_trend_level")

        filtered = trends_df.copy()
        if category != "All":
            filtered = filtered[filtered["product_category"] == category]
        if product != "All":
            filtered = filtered[filtered["product_name"] == product]
        if trend_level != "All" and "trend_level" in filtered.columns:
            filtered = filtered[filtered["trend_level"] == trend_level]

        ui_dataframe(filtered, width="stretch")
        st.download_button("Export Product Trend CSV", filtered.to_csv(index=False), "product_trends_report.csv", "text/csv")

        st.divider()
        st.subheader("DELETE")
        st.warning("Use this only when you want to reset product trend data and upload a new product trend dataset.")
        st.download_button(
            "Download Backup ZIP",
            build_product_trend_backup_zip(),
            file_name=f"product_trend_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            mime="application/zip",
        )
        product_backup_saved = st.checkbox("I have saved the backup", key="product_trend_backup_saved")
        product_confirm_delete = st.text_input("Type DELETE to confirm reset", key="product_trend_confirm_delete")
        if st.button(
            "Delete All Product Trend Data",
            type="primary",
            disabled=not (product_backup_saved and product_confirm_delete == "DELETE"),
        ):
            if PRODUCT_TRENDS_PATH.exists():
                PRODUCT_TRENDS_PATH.unlink()
            mark_product_trends_dirty()
            log_activity(db, "product_trend_data_deleted", st.session_state["user"]["username"], {"backup_confirmed": True})
            st.success("Product trend data was deleted.")
            st.rerun()
        return

    metrics_df = load_supplier_metrics(db)

    if not metrics_df.empty:
        c1, c2, c3 = st.columns(3)
        category = c1.selectbox("Category", ["All"] + sorted(metrics_df["product_category"].dropna().unique()))
        supplier = c2.selectbox("Supplier", ["All"] + sorted(metrics_df["supplier"].dropna().unique()))
        risk = c3.selectbox("Risk", ["All", "Low", "Medium", "High"])
        filtered = metrics_df.copy()
        if category != "All":
            filtered = filtered[filtered["product_category"] == category]
        if supplier != "All":
            filtered = filtered[filtered["supplier"] == supplier]
        if risk != "All":
            filtered = filtered[filtered["risk_level"] == risk]
        st.subheader("All Supplier Metrics")
        ui_dataframe(filtered, width="stretch")
        st.download_button("Export Supplier Metrics CSV", filtered.to_csv(index=False), "supplier_metrics_report.csv", "text/csv")
    else:
        st.info("No supplier metrics found. Upload and clean data first.")

    st.divider()
    st.subheader("DELETE")
    st.warning("Use this only when you want to reset all supplier data and upload a new dataset.")
    st.download_button(
        "Download Backup ZIP",
        build_backup_zip(db),
        file_name=f"supplier_data_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
        mime="application/zip",
    )
    backup_saved = st.checkbox("I have saved the backup")
    confirm_delete = st.text_input("Type DELETE to confirm reset")
    if st.button("Delete All Supplier Data", type="primary", disabled=not (backup_saved and confirm_delete == "DELETE")):
        clear_supplier_data(db)
        st.session_state.pop("cleaned_supplier_rows", None)
        log_activity(db, "supplier_data_deleted", st.session_state["user"]["username"], {"backup_confirmed": True})
        st.success("Supplier data, cleaned data, metrics, user ratings, and recommendation history were deleted.")
        st.rerun()


def page_eda(db):
    page_header("EDA & KPI Analysis")
    mode_col1, mode_col2 = st.columns(2)
    if mode_col1.button("Supplier", use_container_width=True):
        st.session_state["eda_mode"] = "Supplier"
    if mode_col2.button("Product Trend", use_container_width=True):
        st.session_state["eda_mode"] = "Product Trend"
    eda_mode = st.session_state.get("eda_mode", "Supplier")

    if eda_mode == "Product Trend":
        trends_df = load_product_trends()
        if trends_df.empty:
            st.info("No product trend dataset found. Use Upload Data > Product Trend first.")
            return

        latest_month = trends_df["month"].max()
        latest_df = trends_df[trends_df["month"] == latest_month].copy()

        chart1, chart2 = st.columns(2)
        with chart1:
            st.plotly_chart(
                px.bar(
                    latest_df.sort_values("trend_score", ascending=False),
                    x="product_name",
                    y="trend_score",
                    color="product_category",
                    title="Latest Product Trend Score",
                ),
                width="stretch",
            )
        with chart2:
            if "trend_level" in trends_df.columns:
                trend_counts = latest_df.groupby(["product_category", "trend_level"], as_index=False).size()
                st.plotly_chart(
                    px.bar(
                        trend_counts,
                        x="product_category",
                        y="size",
                        color="trend_level",
                        title="High / Medium / Low Trend Count by Category",
                    ),
                    width="stretch",
                )

        chart3, chart4 = st.columns(2)
        with chart3:
            st.plotly_chart(
                px.line(
                    trends_df.sort_values("month"),
                    x="month",
                    y="trend_score",
                    color="product_name",
                    facet_col="product_category",
                    facet_col_wrap=2,
                    title="Product Trend Score Over Time",
                ),
                width="stretch",
            )
        with chart4:
            avg_growth = latest_df.groupby("product_category", as_index=False)["growth_rate"].mean()
            st.plotly_chart(
                px.bar(avg_growth, x="product_category", y="growth_rate", title="Average Latest Growth Rate by Category"),
                width="stretch",
            )

        return

    clean_df = load_collection(db, "cleaned_orders")
    metrics_df = load_supplier_metrics(db)
    ratings_df = load_collection(db, "supplier_ratings")
    if clean_df.empty or metrics_df.empty:
        st.warning("Clean data first.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(px.bar(metrics_df.nlargest(10, "supplier_rank_score"), x="supplier", y="supplier_rank_score", color="product_category", title="Top 10 Suppliers by Score"), width="stretch")
    with col2:
        high_risk_chart = metrics_df.nlargest(10, "risk_score").copy()
        high_risk_chart["supplier_category"] = high_risk_chart["supplier"] + " | " + high_risk_chart["product_category"]
        st.plotly_chart(
            px.bar(
                high_risk_chart,
                x="supplier_category",
                y="risk_score",
                color="risk_level",
                hover_data=["supplier", "product_category", "risk_score", "risk_level"],
                title="Highest Risk Suppliers by Category",
                labels={"supplier_category": "supplier | category"},
            ),
            width="stretch",
        )

    col3, col4 = st.columns(2)
    with col3:
        delay = clean_df.groupby("product_category", as_index=False)["delay_days"].mean()
        st.plotly_chart(px.bar(delay, x="product_category", y="delay_days", title="Average Delay by Category"), width="stretch")
    with col4:
        risk_counts = metrics_df.groupby(["product_category", "risk_level"], as_index=False).size()
        st.plotly_chart(px.bar(risk_counts, x="product_category", y="size", color="risk_level", title="Risk Count by Category"), width="stretch")

    if not ratings_df.empty:
        col5, col6 = st.columns(2)
        with col5:
            rating_avg = ratings_df.groupby("supplier", as_index=False)["rating"].mean().sort_values("rating", ascending=False)
            st.plotly_chart(px.bar(rating_avg.head(10), x="supplier", y="rating", title="Average User Rating by Supplier"), width="stretch")
        with col6:
            events = ratings_df.groupby("event_type", as_index=False).size().sort_values("size", ascending=False)
            st.plotly_chart(px.bar(events, x="event_type", y="size", title="Feedback Event Summary"), width="stretch")


def monthly_category_trend(clean_df, category, supplier_id=None):
    df = clean_df[clean_df["product_category"] == category].copy()
    if supplier_id is not None and "supplier" in df.columns:
        df = df[df["supplier"] == supplier_id].copy()
    if df.empty:
        return pd.DataFrame()
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df = df.dropna(subset=["order_date"])
    for col in ["quantity_ordered", "delay_days", "supply_risk_flag", "has_disruption"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["month"] = df["order_date"].dt.to_period("M").astype(str)
    return (
        df.groupby("month", as_index=False)
        .agg(
            total_orders=("order_id", "count"),
            total_quantity=("quantity_ordered", "sum"),
            avg_delay=("delay_days", "mean"),
            risk_count=("supply_risk_flag", "sum"),
            disruption_count=("has_disruption", "sum"),
        )
        .round(2)
    )


def monthly_supplier_trend(clean_df, supplier_id):
    df = clean_df[clean_df["supplier"] == supplier_id].copy()
    if df.empty:
        return pd.DataFrame()
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df = df.dropna(subset=["order_date"])
    for col in ["quantity_ordered", "delay_days", "supply_risk_flag", "has_disruption"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["month"] = df["order_date"].dt.to_period("M").astype(str)
    return (
        df.groupby("month", as_index=False)
        .agg(
            total_orders=("order_id", "count"),
            total_quantity=("quantity_ordered", "sum"),
            avg_delay=("delay_days", "mean"),
            risk_count=("supply_risk_flag", "sum"),
            disruption_count=("has_disruption", "sum"),
        )
        .round(2)
    )


def predict_next_demand(trend_df):
    if trend_df.empty:
        return 0, "Stable", "Not enough category order data for prediction."
    latest = float(trend_df["total_quantity"].iloc[-1])
    if len(trend_df) == 1:
        return round(latest, 2), "Stable", "Only one month of data is available, so the prediction uses the latest demand."
    previous = float(trend_df["total_quantity"].iloc[-2])
    avg_change = trend_df["total_quantity"].diff().dropna().mean()
    predicted = float(max(0, latest + avg_change))
    growth = ((latest - previous) / previous * 100) if previous else 0
    direction = "Increasing" if growth > 5 else "Decreasing" if growth < -5 else "Stable"
    reason = f"Latest month demand changed by {round(growth, 2)}% compared with the previous month."
    return round(predicted, 2), direction, reason


def predict_supplier_future_risk(metric_row, trend_df):
    current_risk = float(metric_row.get("risk_score", 0))
    delay_change = 0
    disruption_change = 0
    if len(trend_df) >= 2:
        delay_change = float(trend_df["avg_delay"].iloc[-1] - trend_df["avg_delay"].iloc[-2])
        disruption_change = float(trend_df["disruption_count"].iloc[-1] - trend_df["disruption_count"].iloc[-2])
    predicted = current_risk + max(delay_change, 0) * 4 + max(disruption_change, 0) * 2
    if metric_row.get("trend_status") == "Declining":
        predicted += 8
    elif metric_row.get("trend_status") == "Improving":
        predicted -= 5
    predicted = round(float(np.clip(predicted, 0, 100)), 2)
    reasons = []
    if delay_change > 0:
        reasons.append("category average delay increased")
    if disruption_change > 0:
        reasons.append("category disruptions increased")
    if metric_row.get("trend_status") == "Declining":
        reasons.append("recent user rating trend is declining")
    if not reasons:
        reasons.append("recent category trend is stable")
    return predicted, risk_level(predicted), ", ".join(reasons)


def apply_what_if(metric_row, simulated_delay, simulated_rating, simulated_disruption, simulated_reliability):
    current_score = float(metric_row.get("supplier_rank_score", 0))
    current_delay = float(metric_row.get("avg_delay", 0))
    current_disruption = float(metric_row.get("disruption_frequency", 0))
    current_reliability = float(metric_row.get("reliability", 0))
    current_rating = float(metric_row.get("user_rating", 0))

    simulated_risk = (
        ((1 - simulated_reliability) * 35)
        + (min(simulated_delay, 10) / 10 * 25)
        + (simulated_disruption * 25)
        + ((5 - simulated_rating) / 4 * 15)
    )
    current_comparable_risk = (
        ((1 - current_reliability) * 35)
        + (min(current_delay, 10) / 10 * 25)
        + (current_disruption * 25)
        + ((5 - current_rating) / 4 * 15)
    )
    risk_after = round(float(np.clip(simulated_risk, 0, 100)), 2)
    current_comparable_risk = round(float(np.clip(current_comparable_risk, 0, 100)), 2)
    score_after = current_score + (current_comparable_risk - risk_after) * 0.55 + (simulated_rating - current_rating) * 2
    score_after = round(float(np.clip(score_after, 0, 100)), 2)
    scenario_details = {
        "score_change": round(score_after - current_score, 2),
        "risk_change": round(risk_after - current_comparable_risk, 2),
        "current_comparable_risk": current_comparable_risk,
        "simulated_delay": round(simulated_delay, 2),
        "simulated_disruption_frequency": round(simulated_disruption, 2),
        "simulated_reliability": round(simulated_reliability, 2),
        "simulated_rating": round(simulated_rating, 2),
    }
    return score_after, risk_after, risk_level(risk_after), scenario_details


def load_product_trends():
    if not PRODUCT_TRENDS_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(PRODUCT_TRENDS_PATH)
    for col in ["search_volume", "sales_count", "growth_rate", "trend_score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["month"] = df["month"].astype(str)
    df["product_category"] = df["product_category"].astype(str).str.title()
    df["product_name"] = df["product_name"].astype(str)
    return df


def category_product_trends(category):
    trends_df = load_product_trends()
    if trends_df.empty:
        return pd.DataFrame()
    return trends_df[trends_df["product_category"] == category].copy()


def current_trending_products(product_df):
    if product_df.empty:
        return pd.DataFrame()
    latest_month = product_df["month"].max()
    latest = product_df[product_df["month"] == latest_month].copy()
    return latest.sort_values(["trend_score", "growth_rate", "sales_count"], ascending=False)


def future_trending_products(product_df):
    if product_df.empty:
        return pd.DataFrame()
    rows = []
    for product_name, group in product_df.sort_values("month").groupby("product_name"):
        latest = group.iloc[-1]
        avg_change = group["trend_score"].diff().dropna().tail(3).mean()
        if pd.isna(avg_change):
            avg_change = 0
        predicted_score = float(np.clip(latest["trend_score"] + avg_change, 0, 100))
        trend_direction = "Uptrend" if avg_change > 1 else "Downtrend" if avg_change < -1 else "Stable"
        rows.append(
            {
                "product_name": product_name,
                "product_category": latest["product_category"],
                "latest_month": latest["month"],
                "trend_score": round(float(latest["trend_score"]), 2),
                "growth_rate": round(float(latest["growth_rate"]), 2),
                "sales_count": int(latest["sales_count"]),
                "predicted_next_trend_score": round(predicted_score, 2),
                "trend_direction": trend_direction,
                "trend_score_change": round(float(avg_change), 2),
                "prediction_reason": "Trend score is rising recently" if trend_direction == "Uptrend" else "Trend score is falling recently" if trend_direction == "Downtrend" else "Trend score is stable",
            }
        )
    return pd.DataFrame(rows).sort_values("predicted_next_trend_score", ascending=False)


def get_supplier_page_context(db):
    user = st.session_state["user"]
    supplier_id = user.get("supplier_id") or user["username"].upper()
    clean_df = load_collection(db, "cleaned_orders")
    metrics_df = load_supplier_metrics(db)
    ratings_df = load_collection(db, "supplier_ratings")
    if clean_df.empty or metrics_df.empty:
        st.warning("Admin must upload and clean data first.")
        return None

    supplier_metrics = metrics_df[metrics_df["supplier"] == supplier_id].copy()
    if supplier_metrics.empty:
        st.warning(f"No supplier metrics found for {supplier_id}.")
        return None

    category_options = sorted(supplier_metrics["product_category"].dropna().unique())
    saved_category = st.session_state.get("supplier_selected_category")
    if saved_category not in category_options:
        saved_category = category_options[0]
        st.session_state["supplier_selected_category"] = saved_category
    widget_key = f"supplier_category_picker_{st.session_state.get('supplier_page', 'dashboard')}"
    category = st.selectbox(
        "Your Product Category",
        category_options,
        index=category_options.index(saved_category),
        key=widget_key,
    )
    st.session_state["supplier_selected_category"] = category
    metric_row = supplier_metrics[supplier_metrics["product_category"] == category].iloc[0]
    category_df = clean_df[clean_df["product_category"] == category].copy()
    category_metrics = metrics_df[metrics_df["product_category"] == category].copy()
    trend_df = monthly_category_trend(clean_df, category, supplier_id)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Supplier", supplier_id)
    c2.metric("Category", category)
    c3.metric("Current Risk", metric_row["risk_level"])
    c4.metric("Score", f"{metric_row['supplier_rank_score']}/100")
    c5.metric("Rating", f"{metric_row['final_rating']}/5")

    return {
        "supplier_id": supplier_id,
        "category": category,
        "clean_df": clean_df,
        "metrics_df": metrics_df,
        "ratings_df": ratings_df,
        "metric_row": metric_row,
        "category_df": category_df,
        "category_metrics": category_metrics,
        "trend_df": trend_df,
    }


def render_category_demand_charts(category, trend_df):
    st.subheader("Product Demand Chart")
    if trend_df.empty:
        st.info("No monthly trend data is available for this category.")
        return
    t1, t2 = st.columns(2)
    with t1:
        st.plotly_chart(px.line(trend_df, x="month", y="total_quantity", markers=True, title=f"{category} Monthly Quantity Demand"), width="stretch")
    with t2:
        st.plotly_chart(px.bar(trend_df, x="month", y="total_orders", title=f"{category} Monthly Order Count"), width="stretch")


def render_supplier_benchmark(supplier_id, category, metric_row, category_metrics):
    st.subheader(f"{supplier_id} vs {category} Average")
    benchmark = pd.DataFrame(
        {
            "Metric": ["Average Delay", "Final Rating", "Risk Score", "Reliability"],
            supplier_id: [
                metric_row["avg_delay"],
                metric_row["final_rating"],
                metric_row["risk_score"],
                metric_row["reliability"],
            ],
            f"{category} Average": [
                category_metrics["avg_delay"].mean(),
                category_metrics["final_rating"].mean(),
                category_metrics["risk_score"].mean(),
                category_metrics["reliability"].mean(),
            ],
        }
    ).round(2)
    ui_dataframe(benchmark, width="stretch")
    st.plotly_chart(px.bar(benchmark, x="Metric", y=[supplier_id, f"{category} Average"], barmode="group", title=f"{supplier_id} vs {category} Average"), width="stretch")


def render_supplier_feedback(supplier_id, category, ratings_df):
    st.subheader("Supplier Feedback")
    supplier_ratings = ratings_df[(ratings_df["supplier"] == supplier_id) & (ratings_df["product_category"] == category)] if not ratings_df.empty else pd.DataFrame()
    if supplier_ratings.empty:
        st.info("No user feedback for this supplier/category yet.")
        return supplier_ratings, 0, 0
    ui_dataframe(
        safe_metric_table(
            supplier_ratings.sort_values("created_at", ascending=False),
            ["created_at", "username", "rating", "event_type", "comment"],
        ),
        width="stretch",
    )
    avg_feedback = round(pd.to_numeric(supplier_ratings["rating"], errors="coerce").mean(), 2)
    return supplier_ratings, len(supplier_ratings), avg_feedback


def page_supplier_dashboard(db):
    page_header("Supplier Dashboard")
    context = get_supplier_page_context(db)
    if context is None:
        return

    supplier_id = context["supplier_id"]
    category = context["category"]
    supplier_id = context["supplier_id"]
    metric_row = context["metric_row"]
    trend_df = context["trend_df"]
    category_metrics = context["category_metrics"]
    ratings_df = context["ratings_df"]

    render_category_demand_charts(category, trend_df)
    render_supplier_benchmark(supplier_id, category, metric_row, category_metrics)
    _, feedback_count, avg_feedback = render_supplier_feedback(supplier_id, category, ratings_df)

    st.subheader("Supplier Dashboard Summary")
    supplier_trend_df = monthly_supplier_trend(context["clean_df"], supplier_id)
    predicted_demand, demand_direction, _ = predict_next_demand(supplier_trend_df)
    predicted_risk, predicted_risk_level, _ = predict_supplier_future_risk(metric_row, trend_df)
    current_demand = float(supplier_trend_df["total_quantity"].iloc[-1]) if not supplier_trend_df.empty else 0
    demand_delta_color = "normal" if demand_direction in {"Increasing", "Decreasing"} else "off"
    demand_delta = f"+ {demand_direction}" if demand_direction == "Increasing" else f"- {demand_direction}" if demand_direction == "Decreasing" else demand_direction
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Predicted Next Month Demand", predicted_demand, demand_delta, delta_color=demand_delta_color)
    s2.metric("Future Risk", predicted_risk_level, f"{predicted_risk}/100")
    s3.metric("Avg Feedback", f"{avg_feedback}/5", f"{feedback_count} ratings")
    s4.metric("Current Score", f"{metric_row['supplier_rank_score']}/100")
    s5.metric("Current Month Demand", f"{current_demand:,.0f}")


def page_supplier_trend(db):
    page_header("Supplier Trend")
    context = get_supplier_page_context(db)
    if context is None:
        return

    category = context["category"]
    product_df = category_product_trends(category)
    current_products = current_trending_products(product_df)
    future_products = future_trending_products(product_df)

    st.subheader("Specific Product Trend Analysis")
    if product_df.empty:
        st.info("No product trend dataset found for this category.")
        return

    latest_month = product_df["month"].max()
    current_top = current_products.iloc[0]
    future_top = future_products.sort_values("predicted_next_trend_score", ascending=False).iloc[0]
    downtrend_products = future_products.sort_values("trend_score_change", ascending=True)
    future_down = downtrend_products.iloc[0]
    pt1, pt2, pt3, pt4 = st.columns(4)
    pt1.metric("Latest Trend Month", latest_month)
    pt2.metric("Trending Now", current_top["product_name"], f"{current_top['trend_score']}/100")
    pt3.metric("Predicted Future Trend", future_top["product_name"], f"{future_top['predicted_next_trend_score']}/100")
    pt4.metric("Predicted Downtrend", future_down["product_name"], f"{future_down['trend_score_change']} score change")

    product_chart = product_df.sort_values("month")
    pc1, pc2 = st.columns(2)
    with pc1:
        st.plotly_chart(
            px.line(product_chart, x="month", y="trend_score", color="product_name", markers=True, title=f"{category} Product Trend Score Over Time"),
            width="stretch",
        )
    with pc2:
        st.plotly_chart(
            px.bar(current_products, x="product_name", y="sales_count", color="growth_rate", title=f"{category} Current Product Demand"),
            width="stretch",
        )

    st.subheader("Current Trending Products")
    ui_dataframe(
        format_product_trend_display(
            safe_metric_table(current_products, ["product_name", "search_volume", "sales_count", "growth_rate", "trend_score"])
        ),
        width="stretch",
    )
    st.subheader("Future Product Trend Prediction")
    ui_dataframe(
        format_product_trend_display(
            safe_metric_table(
                future_products,
                ["product_name", "trend_score", "growth_rate", "sales_count", "predicted_next_trend_score", "trend_direction", "trend_score_change", "prediction_reason"],
            )
        ),
        width="stretch",
    )
    st.subheader("Predicted Downtrend Products")
    ui_dataframe(
        format_product_trend_display(
            safe_metric_table(
                downtrend_products,
                ["product_name", "trend_score", "growth_rate", "predicted_next_trend_score", "trend_direction", "trend_score_change", "prediction_reason"],
            )
        ),
        width="stretch",
    )


def page_supplier_future_prediction(db):
    page_header("Future Prediction")
    context = get_supplier_page_context(db)
    if context is None:
        return

    category = context["category"]
    supplier_id = context["supplier_id"]
    metric_row = context["metric_row"]
    trend_df = context["trend_df"]

    st.subheader("Category Risk Trend")
    if not trend_df.empty:
        r1, r2 = st.columns(2)
        with r1:
            st.plotly_chart(px.line(trend_df, x="month", y="avg_delay", markers=True, title=f"{category} Average Delay Trend"), width="stretch")
        with r2:
            st.plotly_chart(px.bar(trend_df, x="month", y="risk_count", title=f"{category} Risk Flag Count"), width="stretch")
    else:
        st.info("No monthly trend data is available for this category.")

    st.subheader("Future Prediction")
    supplier_trend_df = monthly_supplier_trend(context["clean_df"], supplier_id)
    predicted_demand, demand_direction, demand_reason = predict_next_demand(supplier_trend_df)
    predicted_risk, predicted_risk_level, risk_reason = predict_supplier_future_risk(metric_row, trend_df)
    p1, p2, p3 = st.columns(3)
    p1.metric("Predicted Next Demand", predicted_demand)
    p2.metric("Demand Trend", demand_direction)
    p3.metric("Predicted Future Risk", predicted_risk_level, f"{predicted_risk}/100")
    st.write(f"Demand reason: {demand_reason}")
    st.write(f"Risk reason: {risk_reason}.")
    st.info(
        "Risk level is based on the supplier risk score. Low means 0-34, Medium means 35-64, and High means 65-100. "
        "The score uses reliability, average delay, disruption frequency, supply risk flags, disruption severity, user rating, and bad feedback."
    )

    st.subheader("What-If Supplier Improvement")
    w1, w2 = st.columns(2)
    current_delay = float(metric_row.get("avg_delay", 0))
    current_rating = float(metric_row.get("user_rating", 0))
    current_disruption = float(metric_row.get("disruption_frequency", 0))
    current_reliability = float(metric_row.get("reliability", 0))
    simulated_delay = w1.slider("What if average delay becomes days", 0.0, 10.0, min(10.0, current_delay), 0.5)
    simulated_disruption = w2.slider("What if disruption frequency becomes", 0.0, 1.0, min(1.0, current_disruption), 0.05)
    simulated_rating = w1.slider("What if user rating becomes", 1.0, 5.0, min(5.0, max(1.0, current_rating)), 0.1)
    simulated_reliability = w2.slider("What if reliability becomes", 0.0, 1.0, min(1.0, max(0.0, current_reliability)), 0.05)
    score_after, risk_after, risk_after_level, scenario_details = apply_what_if(
        metric_row,
        simulated_delay,
        simulated_rating,
        simulated_disruption,
        simulated_reliability,
    )
    score_delta = scenario_details["score_change"]
    risk_delta = scenario_details["risk_change"]
    result_score_label = "What-If Score"
    result_risk_label = "What-If Risk"
    st.caption(
        f"Current values: delay {round(current_delay, 2)} days, disruption {round(current_disruption, 2)}, "
        f"reliability {round(current_reliability, 2)}, rating {round(current_rating, 2)}. "
        f"What-if values: delay {scenario_details['simulated_delay']} days, disruption {scenario_details['simulated_disruption_frequency']}, "
        f"reliability {scenario_details['simulated_reliability']}, rating {scenario_details['simulated_rating']}."
    )
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Current Score", f"{metric_row['supplier_rank_score']}/100")
    a2.metric(result_score_label, f"{score_after}/100", score_delta)
    a3.metric("Current Risk", f"{risk_level(scenario_details['current_comparable_risk'])} ({scenario_details['current_comparable_risk']}/100)")
    a4.metric(result_risk_label, f"{risk_after_level} ({risk_after}/100)", risk_delta)
    st.info(
        "What-if risk is recalculated from the slider values. More delay, more disruption, lower reliability, or lower rating increases risk. "
        "Less delay, less disruption, higher reliability, or higher rating reduces risk."
    )
    st.plotly_chart(
        px.bar(
            pd.DataFrame(
                {
                    "Metric": ["Current Score", result_score_label, "Current Risk", result_risk_label],
                    "Value": [metric_row["supplier_rank_score"], score_after, scenario_details["current_comparable_risk"], risk_after],
                }
            ),
            x="Metric",
            y="Value",
            title="Current vs What-If Result",
        ),
        width="stretch",
    )

    st.plotly_chart(
        px.bar(
            pd.DataFrame(
                {
                    "Metric": ["Score", "Score", "Risk", "Risk"],
                    "Scenario": ["Current", "What-If", "Current", "What-If"],
                    "Value": [metric_row["supplier_rank_score"], score_after, scenario_details["current_comparable_risk"], risk_after],
                }
            ),
            x="Metric",
            y="Value",
            color="Scenario",
            barmode="group",
            title="Summary: Current vs What-If",
        ),
        width="stretch",
    )


def page_ratings_feedback(db):
    page_header("User Rating")
    ratings_df = load_collection(db, "supplier_ratings")
    st.subheader("User Rating")
    if ratings_df.empty:
        st.info("No user ratings submitted yet.")
    else:
        activity = ratings_df.sort_values("created_at", ascending=False).copy()
        ui_dataframe(
            safe_metric_table(
                activity,
                ["created_at", "username", "supplier", "product_category", "rating", "event_type", "comment"],
            ),
            width="stretch",
        )
        st.download_button("Export Ratings CSV", ratings_df.to_csv(index=False), "supplier_ratings_report.csv", "text/csv")


def user_account_table(db):
    users = dataframe_from_collection(db, COLLECTIONS["users"], {"role": "user"})
    if users.empty:
        return users
    logs = load_collection(db, "recommendation_logs")
    ratings = load_collection(db, "supplier_ratings")
    selected_counts = pd.DataFrame(columns=["username", "selected_supplier_count"])
    rating_counts = pd.DataFrame(columns=["username", "ratings_given_count"])
    if not logs.empty and {"username", "status"}.issubset(logs.columns):
        selected_counts = (
            logs[logs["status"] == "selected"]
            .groupby("username", as_index=False)
            .size()
            .rename(columns={"size": "selected_supplier_count"})
        )
    if not ratings.empty and "username" in ratings.columns:
        rating_counts = ratings.groupby("username", as_index=False).size().rename(columns={"size": "ratings_given_count"})
    users = users.merge(selected_counts, on="username", how="left").merge(rating_counts, on="username", how="left")
    users["selected_supplier_count"] = users["selected_supplier_count"].fillna(0).astype(int)
    users["ratings_given_count"] = users["ratings_given_count"].fillna(0).astype(int)
    return safe_metric_table(users, ["username", "is_active", "created_at", "selected_supplier_count", "ratings_given_count"])


def supplier_account_table(db):
    suppliers = dataframe_from_collection(db, COLLECTIONS["users"], {"role": "supplier"})
    if suppliers.empty:
        return suppliers
    ratings = load_collection(db, "supplier_ratings")
    rows = []
    for _, account in suppliers.iterrows():
        supplier_id = str(account.get("supplier_id", "")).upper()
        info = supplier_id_info(db, supplier_id, current_username=account.get("username"))
        supplier_ratings = ratings[ratings["supplier"] == supplier_id] if not ratings.empty and "supplier" in ratings.columns else pd.DataFrame()
        avg_rating = 0
        if not supplier_ratings.empty and "rating" in supplier_ratings.columns:
            avg_rating = round(pd.to_numeric(supplier_ratings["rating"], errors="coerce").mean(), 2)
        rows.append(
            {
                "username": account.get("username"),
                "supplier_id": supplier_id,
                "is_active": account.get("is_active", False),
                "account_status": account.get("account_status", "pending"),
                "supplier_id_exists": info["supplier_id_exists"],
                "order_count": info["order_count"],
                "categories": info["categories"],
                "already_claimed": info["already_claimed"],
                "created_at": account.get("created_at"),
                "feedback_count": len(supplier_ratings),
                "avg_rating": avg_rating,
            }
        )
    return pd.DataFrame(rows)


def update_account_username(db, old_username, new_username, actor):
    new_username = new_username.strip()
    if len(new_username) < 3:
        return False, "Username must be at least 3 characters."
    if db[COLLECTIONS["users"]].find_one({"username": new_username}):
        return False, "Username already exists."
    db[COLLECTIONS["users"]].update_one({"username": old_username}, {"$set": {"username": new_username}})
    log_activity(db, "account_username_updated", actor, {"old_username": old_username, "new_username": new_username})
    return True, "Username updated."


def update_account_password(db, username, new_password, actor):
    if len(new_password) < 6:
        return False, "Password must be at least 6 characters."
    db[COLLECTIONS["users"]].update_one({"username": username}, {"$set": {"password_hash": password_hash(new_password)}})
    log_activity(db, "account_password_reset", actor, {"username": username})
    return True, "Password reset."


def page_manage_users(db):
    page_header("Manage Accounts")
    actor = st.session_state["user"]["username"]
    mode_col1, mode_col2 = st.columns(2)
    if mode_col1.button("User Accounts", use_container_width=True):
        st.session_state["manage_account_mode"] = "User Accounts"
    if mode_col2.button("Supplier Accounts", use_container_width=True):
        st.session_state["manage_account_mode"] = "Supplier Accounts"
    mode = st.session_state.get("manage_account_mode", "User Accounts")

    if mode == "Supplier Accounts":
        supplier_df = supplier_account_table(db)
        pending_df = supplier_df[supplier_df["account_status"] == "pending"].copy() if not supplier_df.empty else pd.DataFrame()
        managed_supplier_df = supplier_df[supplier_df["account_status"] != "pending"].copy() if not supplier_df.empty else pd.DataFrame()
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Supplier Accounts", len(supplier_df))
        s2.metric("Pending Requests", len(pending_df))
        s3.metric("Approved", int((supplier_df["account_status"] == "approved").sum()) if not supplier_df.empty else 0)
        s4.metric("Active", int((supplier_df["is_active"] == True).sum()) if not supplier_df.empty else 0)

        st.subheader("Supplier Accounts")
        if supplier_df.empty:
            st.info("No supplier accounts yet.")
        else:
            ui_dataframe(supplier_df, width="stretch")

        if not managed_supplier_df.empty:
            st.subheader("Manage Supplier Account")
            selected_supplier_user = st.selectbox("Select supplier account", managed_supplier_df["username"].tolist(), key="selected_supplier_account")
            selected_doc = db[COLLECTIONS["users"]].find_one({"username": selected_supplier_user}, {"_id": 0})
            current_supplier_id = selected_doc.get("supplier_id", "") if selected_doc else ""
            info = supplier_id_info(db, current_supplier_id, current_username=selected_supplier_user)

            u1, u2 = st.columns(2)
            new_supplier_username = u1.text_input("Update username", value=selected_supplier_user, key="supplier_new_username")
            if u1.button("Update Supplier Username"):
                success, message = update_account_username(db, selected_supplier_user, new_supplier_username, actor)
                if success:
                    st.success(message)
                else:
                    st.error(message)
                if success:
                    st.rerun()
            new_supplier_password = u2.text_input("New password", type="password", key="supplier_new_password")
            if u2.button("Reset Supplier Password"):
                success, message = update_account_password(db, selected_supplier_user, new_supplier_password, actor)
                if success:
                    st.success(message)
                else:
                    st.error(message)

            a1, a2 = st.columns(2)
            if a1.button("Activate"):
                db[COLLECTIONS["users"]].update_one({"username": selected_supplier_user}, {"$set": {"is_active": True}})
                log_activity(db, "supplier_account_activated", actor, {"username": selected_supplier_user})
                st.success("Supplier account activated.")
                st.rerun()
            if a2.button("Deactivate"):
                db[COLLECTIONS["users"]].update_one({"username": selected_supplier_user}, {"$set": {"is_active": False}})
                log_activity(db, "supplier_account_deactivated", actor, {"username": selected_supplier_user})
                st.success("Supplier account deactivated.")
                st.rerun()

            st.subheader("Supplier Activity")
            ratings = load_collection(db, "supplier_ratings")
            supplier_ratings = ratings[ratings["supplier"] == current_supplier_id] if not ratings.empty and "supplier" in ratings.columns else pd.DataFrame()
            if supplier_ratings.empty:
                st.info("No supplier feedback activity.")
            else:
                ui_dataframe(safe_metric_table(supplier_ratings.sort_values("created_at", ascending=False), ["created_at", "username", "supplier", "product_category", "rating", "event_type", "comment"]), width="stretch")
        else:
            st.info("No approved or rejected supplier accounts to manage yet.")

        st.subheader("Create Supplier Account")
        with st.form("admin_create_supplier_account"):
            supplier_username = st.text_input("Supplier username")
            supplier_password = st.text_input("Supplier password", type="password")
            supplier_id = st.text_input("Supplier ID", placeholder="Example: S10").strip().upper()
            create_supplier_submitted = st.form_submit_button("Create Approved Supplier")
        if create_supplier_submitted:
            info = supplier_id_info(db, supplier_id)
            if db[COLLECTIONS["users"]].find_one({"username": supplier_username.strip()}):
                st.error("Username already exists.")
            elif len(supplier_password) < 6:
                st.error("Password must be at least 6 characters.")
            elif info["supplier_id_exists"] != "Yes" or info["order_count"] <= 0 or info["category_match"] != "Yes":
                st.error("Supplier ID verification failed.")
            elif info["already_claimed"] == "Yes":
                st.error("This supplier ID is already claimed by another approved supplier account.")
            else:
                db[COLLECTIONS["users"]].insert_one(
                    {
                        "username": supplier_username.strip(),
                        "password_hash": password_hash(supplier_password),
                        "role": "supplier",
                        "supplier_id": supplier_id,
                        "is_active": True,
                        "account_status": "approved",
                        "created_at": datetime.now(timezone.utc),
                    }
                )
                log_activity(db, "supplier_account_created_by_admin", actor, {"username": supplier_username.strip(), "supplier_id": supplier_id})
                st.success("Supplier account created and approved.")
                st.rerun()

        if not pending_df.empty:
            st.subheader("Approval")
            pending_supplier_user = st.selectbox("Select pending supplier request", pending_df["username"].tolist(), key="pending_supplier_account")
            pending_doc = db[COLLECTIONS["users"]].find_one({"username": pending_supplier_user}, {"_id": 0})
            pending_supplier_id = pending_doc.get("supplier_id", "") if pending_doc else ""
            pending_info = supplier_id_info(db, pending_supplier_id, current_username=pending_supplier_user)
            selected_status = pending_doc.get("account_status", "pending") if pending_doc else "pending"
            selected_active = "Active" if pending_doc and pending_doc.get("is_active", False) else "Inactive"
            st.info(
                f"Pending account: {pending_supplier_user} | Supplier ID: {pending_supplier_id} | "
                f"Status: {selected_status} | Account: {selected_active}"
            )
            ap1, ap2, ap3, ap4, ap5 = st.columns(5)
            ap1.metric("Supplier ID Exists", pending_info["supplier_id_exists"])
            ap2.metric("Order History", pending_info["order_count"])
            ap3.metric("Category Match", pending_info["category_match"])
            ap4.metric("Already Claimed", pending_info["already_claimed"])
            ap5.metric("Claimed By", pending_info["claimed_by"])
            st.caption(f"Categories: {pending_info['categories']}")
            a1, a2 = st.columns(2)
            if a1.button("Approve"):
                if pending_info["supplier_id_exists"] == "Yes" and pending_info["order_count"] > 0 and pending_info["category_match"] == "Yes" and pending_info["already_claimed"] == "No":
                    db[COLLECTIONS["users"]].update_one({"username": pending_supplier_user}, {"$set": {"account_status": "approved", "is_active": True}})
                    log_activity(db, "supplier_account_approved", actor, {"username": pending_supplier_user, "supplier_id": pending_supplier_id})
                    st.success("Supplier account approved.")
                    st.rerun()
                else:
                    st.error(f"Cannot approve. This supplier ID is already claimed by {pending_info['claimed_by']}.")
            if a2.button("Reject"):
                db[COLLECTIONS["users"]].update_one({"username": pending_supplier_user}, {"$set": {"account_status": "rejected", "is_active": False}})
                log_activity(db, "supplier_account_rejected", actor, {"username": pending_supplier_user, "supplier_id": pending_supplier_id})
                st.success("Supplier account rejected.")
                st.rerun()
            if pending_info["already_claimed"] == "Yes":
                st.warning(f"{pending_supplier_id} is already approved for {pending_info['claimed_by']}. Use replace only if this pending account is the real supplier.")
                if st.button("Approve and Replace Existing Claim"):
                    db[COLLECTIONS["users"]].update_many(
                        {
                            "role": "supplier",
                            "supplier_id": pending_supplier_id,
                            "account_status": "approved",
                            "username": {"$ne": pending_supplier_user},
                        },
                        {"$set": {"account_status": "replaced", "is_active": False}},
                    )
                    db[COLLECTIONS["users"]].update_one(
                        {"username": pending_supplier_user},
                        {"$set": {"account_status": "approved", "is_active": True}},
                    )
                    log_activity(
                        db,
                        "supplier_account_approved_replacing_claim",
                        actor,
                        {"username": pending_supplier_user, "supplier_id": pending_supplier_id, "previous_claim": pending_info["claimed_by"]},
                    )
                    st.success("Supplier account approved and previous claim was deactivated.")
                    st.rerun()
        else:
            st.subheader("Approval")
            st.info("No pending supplier requests to approve.")

        st.subheader("Supplier Verification Codes")
        vc1, vc2 = st.columns([2, 1])
        supplier_id_for_code = vc1.text_input("Supplier ID for code", placeholder="Example: S10").strip().upper()
        if vc2.button("Generate Code", use_container_width=True):
            info = supplier_id_info(db, supplier_id_for_code)
            if not supplier_id_for_code:
                st.error("Enter supplier ID first.")
            elif info["supplier_id_exists"] != "Yes":
                st.error("Supplier ID not found in uploaded supplier data.")
            else:
                code = save_supplier_verification_code(db, supplier_id_for_code, actor)
                st.success(f"Verification code for {supplier_id_for_code}: {code}")

        codes_df = dataframe_from_collection(db, COLLECTIONS["supplier_verification_codes"])
        if not codes_df.empty:
            ui_dataframe(safe_metric_table(codes_df, ["supplier_id", "verification_code", "is_used", "used_by", "created_at", "updated_at"]), width="stretch")

        st.subheader("Pending Supplier Requests")
        if pending_df.empty:
            st.info("No pending supplier requests.")
        else:
            ui_dataframe(
                safe_metric_table(
                    pending_df,
                    ["username", "supplier_id", "supplier_id_exists", "order_count", "categories", "already_claimed", "account_status", "created_at"],
                ),
                width="stretch",
            )
        return

    users_df = user_account_table(db)
    u1, u2, u3 = st.columns(3)
    u1.metric("User Accounts", len(users_df))
    u2.metric("Active Users", int((users_df["is_active"] == True).sum()) if not users_df.empty else 0)
    u3.metric("Ratings Given", int(users_df["ratings_given_count"].sum()) if not users_df.empty else 0)

    st.subheader("User Accounts")
    if users_df.empty:
        st.info("No user accounts yet.")
    else:
        ui_dataframe(users_df, width="stretch")

    if not users_df.empty:
        st.subheader("Manage User Accounts")
        selected = st.selectbox("Select user account", users_df["username"].tolist(), key="selected_user_account")
        c1, c2 = st.columns(2)
        new_username = c1.text_input("Update username", value=selected, key="new_user_username")
        if c1.button("Update Username"):
            success, message = update_account_username(db, selected, new_username, actor)
            if success:
                st.success(message)
            else:
                st.error(message)
            if success:
                st.rerun()
        new_password = c2.text_input("New password", type="password", key="new_user_password")
        if c2.button("Reset Password"):
            success, message = update_account_password(db, selected, new_password, actor)
            if success:
                st.success(message)
            else:
                st.error(message)

        a1, a2 = st.columns(2)
        if a1.button("Activate User"):
            db[COLLECTIONS["users"]].update_one({"username": selected}, {"$set": {"is_active": True}})
            log_activity(db, "user_account_activated", actor, {"username": selected})
            st.success("User account activated.")
            st.rerun()
        if a2.button("Deactivate User"):
            db[COLLECTIONS["users"]].update_one({"username": selected}, {"$set": {"is_active": False}})
            log_activity(db, "user_account_deactivated", actor, {"username": selected})
            st.success("User account deactivated.")
            st.rerun()

        st.subheader("User Activity")
        logs = dataframe_from_collection(db, COLLECTIONS["recommendation_logs"], {"username": selected})
        ratings = dataframe_from_collection(db, COLLECTIONS["supplier_ratings"], {"username": selected})
        act1, act2 = st.columns(2)
        with act1:
            st.caption("Selected Suppliers")
            selected_logs = logs[logs["status"] == "selected"] if not logs.empty and "status" in logs.columns else pd.DataFrame()
            ui_dataframe(safe_metric_table(selected_logs.sort_values("created_at", ascending=False) if not selected_logs.empty else selected_logs, ["supplier", "product_category", "final_score", "risk_level", "created_at"]), width="stretch")
        with act2:
            st.caption("Ratings Given")
            ui_dataframe(safe_metric_table(ratings.sort_values("created_at", ascending=False) if not ratings.empty else ratings, ["supplier", "product_category", "rating", "event_type", "comment", "created_at"]), width="stretch")

    st.subheader("Create New Account")
    with st.form("admin_create_user_account"):
        username = st.text_input("New username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Create User")
    if submitted:
        success, message = create_account(db, username, password, password, "user")
        if success:
            st.success(message)
        else:
            st.error(message)
        if success:
            st.rerun()


def page_user_home(db):
    metrics_df = load_supplier_metrics(db)
    if metrics_df.empty:
        st.warning("No supplier data available yet.")
        return
    if "supplier_rank_score" not in metrics_df.columns:
        st.warning("Supplier metrics need to be refreshed by admin from Clean Data.")
        return
    st.markdown(
        f"""
        <div class="page-hero">
            <div class="page-hero-top">
                <div class="ui-kicker">SUPPLIER DISCOVERY</div>
                <h1>Find the right supplier<br>with confidence.</h1>
                <p>Explore high-performing suppliers, compare risk and keep your favourites close.</p>
            </div>
        </div>
        """, unsafe_allow_html=True,
    )
    title_col, search_col = st.columns([1.55, 1])
    search_text = search_col.text_input("Search", placeholder="Example: S10, Machinery, Food").strip()
    if not search_text:
        supplier_count = int(metrics_df["supplier"].nunique()) if "supplier" in metrics_df.columns else 0
        category_count = int(metrics_df["product_category"].nunique()) if "product_category" in metrics_df.columns else 0
        score_series = pd.to_numeric(metrics_df["supplier_rank_score"], errors="coerce") if "supplier_rank_score" in metrics_df.columns else pd.Series(dtype=float)
        rating_series = pd.to_numeric(metrics_df["final_rating"], errors="coerce") if "final_rating" in metrics_df.columns else pd.Series(dtype=float)
        top_score = float(score_series.max()) if not score_series.empty else 0
        top_rating = float(rating_series.max()) if not rating_series.empty else 0
        top_score_row = metrics_df.loc[score_series.idxmax()] if not score_series.dropna().empty else {}
        top_rating_row = metrics_df.loc[rating_series.idxmax()] if not rating_series.dropna().empty else {}
        top_score_supplier = f"{top_score_row.get('supplier', 'N/A')} - {top_score_row.get('product_category', 'N/A')}" if hasattr(top_score_row, "get") else "N/A"
        top_rating_supplier = f"{top_rating_row.get('supplier', 'N/A')} - {top_rating_row.get('product_category', 'N/A')}" if hasattr(top_rating_row, "get") else "N/A"
        st.markdown('<div class="discover-kicker">DISCOVER</div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(f'<div class="metric-card"><div class="label">Suppliers</div><div class="value">{supplier_count}</div><div class="subvalue">Available suppliers</div></div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="metric-card"><div class="label">Categories</div><div class="value">{category_count}</div><div class="subvalue">Product groups</div></div>', unsafe_allow_html=True)
        m3.markdown(f'<div class="metric-card"><div class="label">Top score</div><div class="value">{top_score:.0f}/100</div><div class="subvalue">{top_score_supplier}</div></div>', unsafe_allow_html=True)
        m4.markdown(f'<div class="metric-card"><div class="label">Top rating</div><div class="value">{top_rating:.1f}/5</div><div class="subvalue">{top_rating_supplier}</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="discover-kicker">BEST SUPPLIERS BY CATEGORY</div>', unsafe_allow_html=True)
        title_col.markdown(
            """
            <div class="inline-page-title">
                <h1>🏠 Best Suppliers</h1>
                <p>Quickly view top suppliers or search by supplier ID and category.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        title_col.markdown(
            """
            <div class="inline-page-title">
                <h1>🔎 Search Results</h1>
                <p>Select or favourite matching suppliers from your search.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    result_columns = ["supplier", "product_category", "final_rating", "risk_level", "supplier_rank_score", "avg_delay", "avg_unit_price", "trend_status"]
    username = st.session_state["user"]["username"]

    if search_text:
        query = search_text.lower()
        supplier_match = metrics_df["supplier"].astype(str).str.lower() == query
        category_match = metrics_df["product_category"].astype(str).str.lower() == query
        if supplier_match.any():
            search_results = metrics_df[supplier_match].copy().sort_values("product_category")
        elif category_match.any():
            search_results = metrics_df[category_match].copy().sort_values("supplier_rank_score", ascending=False)
        else:
            search_results = metrics_df[
                metrics_df["supplier"].astype(str).str.lower().str.contains(query, na=False)
                | metrics_df["product_category"].astype(str).str.lower().str.contains(query, na=False)
            ].copy().sort_values("supplier_rank_score", ascending=False)

        if search_results.empty:
            st.info("No matching supplier or category found.")
        else:
            st.subheader("Select Supplier")
            st.caption("Choose one supplier from the search results below.")
            current_hot = hot_supplier_keys(db, username)
            h1, h2, h3, h4, h5, h6, h7, h8, h9 = st.columns([1, 1.4, 1, 1, 1, 1, 1, 0.8, 0.5])
            h1.caption("Supplier")
            h2.caption("Category")
            h3.caption("Rating")
            h4.caption("Risk")
            h5.caption("Score")
            h6.caption("Delay")
            h7.caption("Price")
            h8.caption("Select")
            h9.caption("Fav")
            for _, row in search_results.iterrows():
                c1, c2, c3, c4, c5, c6, c7, c8, c9 = st.columns([1, 1.4, 1, 1, 1, 1, 1, 0.8, 0.5])
                c1.write(str(row["supplier"]))
                c2.write(str(row["product_category"]))
                c3.write(f"{fmt_number(row.get('final_rating', 0))}/5")
                c4.write(str(row.get("risk_level", "")))
                c5.write(f"{fmt_number(row.get('supplier_rank_score', 0))}/100")
                c6.write(f"{fmt_number(row.get('avg_delay', 0))} days")
                c7.write(fmt_number(row.get("avg_unit_price", 0)))
                if c8.button("Select", key=f"home_select_{row['supplier']}_{row['product_category']}"):
                    save_selected_supplier(db, username, row, "home_search")
                    st.success(f"Selected {row['supplier']} for {row['product_category']}.")
                is_favourite = (str(row["supplier"]), str(row["product_category"])) in current_hot
                star_label = "★" if is_favourite else "☆"
                star_help = "Remove from favourite supplier" if is_favourite else "Add to favourite supplier"
                if c9.button(star_label, key=f"home_fav_{row['supplier']}_{row['product_category']}", help=star_help):
                    if is_favourite:
                        remove_hot_supplier(db, username, row["supplier"], row["product_category"])
                        log_activity(db, "favourite_supplier_removed", username, {"supplier": row["supplier"], "category": row["product_category"], "source": "home_search"})
                        st.success(f"{row['supplier']} removed from favourite supplier.")
                    else:
                        save_hot_supplier(db, username, row)
                        log_activity(db, "favourite_supplier_saved", username, {"supplier": row["supplier"], "category": row["product_category"], "source": "home_search"})
                        st.success(f"{row['supplier']} saved as a favourite supplier.")
                    st.rerun()
        return
    else:
        best = metrics_df.sort_values("supplier_rank_score", ascending=False).groupby("product_category", as_index=False).first()
        ui_dataframe(
            safe_metric_table(best, ["product_category", "supplier", "final_rating", "risk_level", "supplier_rank_score", "trend_status"]),
            width="stretch",
        )

    st.subheader("Favourite Supplier")
    hot_df = dataframe_from_collection(db, HOT_SUPPLIERS_COLLECTION, {"username": st.session_state["user"]["username"]})
    if hot_df.empty:
        st.info("No favourite supplier yet. Use the star button in Supplier Recommendation to save one.")
    else:
        if "created_at" in hot_df.columns:
            hot_df = hot_df.sort_values("created_at", ascending=False)
        fh1, fh2, fh3, fh4, fh5, fh6, fh7, fh8 = st.columns([1, 1.4, 1, 1, 1, 1, 0.9, 0.5])
        fh1.caption("Supplier")
        fh2.caption("Category")
        fh3.caption("Score")
        fh4.caption("Rating")
        fh5.caption("Risk")
        fh6.caption("Price")
        fh7.caption("Select")
        fh8.caption("Fav")
        for _, row in hot_df.iterrows():
            fc1, fc2, fc3, fc4, fc5, fc6, fc7, fc8 = st.columns([1, 1.4, 1, 1, 1, 1, 0.9, 0.5])
            fc1.write(str(row.get("supplier")))
            fc2.write(str(row.get("product_category")))
            fc3.write(f"{fmt_number(row.get('final_score', 0))}/100")
            fc4.write(f"{fmt_number(row.get('final_rating', 0))}/5")
            fc5.write(str(row.get("risk_level", "")))
            fc6.write(fmt_number(row.get("avg_unit_price", 0)))
            if fc7.button("Select", key=f"home_reselect_{row.get('supplier')}_{row.get('product_category')}"):
                save_selected_supplier(db, username, row, "favourite_reselect")
                st.success(f"Selected {row.get('supplier')} for {row.get('product_category')}.")
            if fc8.button("★", key=f"home_remove_fav_{row.get('supplier')}_{row.get('product_category')}", help="Remove from favourite supplier"):
                remove_hot_supplier(db, username, row.get("supplier"), row.get("product_category"))
                log_activity(db, "favourite_supplier_removed", username, {"supplier": row.get("supplier"), "category": row.get("product_category"), "source": "home"})
                st.success(f"{row.get('supplier')} removed from favourite supplier.")
                st.rerun()


def page_find_supplier(db):
    page_header("Supplier Recommendation")
    metrics_df = load_supplier_metrics(db)
    if metrics_df.empty:
        st.warning("Admin must clean data first.")
        return
    categories = sorted(metrics_df["product_category"].dropna().astype(str).unique())
    username = st.session_state["user"]["username"]
    defaults = {"smart_category": categories[0] if categories else "", "smart_quantity": 1000, "smart_budget": 100.0, "smart_quality": 4.0, "smart_deadline": 14, "smart_priority": "Balanced", "smart_top_n": 5}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    st.markdown("""
    <div class="smart-search-banner"><div><div class="ui-kicker">SMART RECOMMENDATION</div><h2>Build your supplier brief</h2><p>Set your business requirements once. SupplyLogix will filter, score, compare and explain the strongest matches.</p></div><div class="smart-ai-orb"><span>✦</span><small>SMART<br>MATCH</small></div></div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="smart-section-label">01 · CHOOSE YOUR BUYING STRATEGY</div>', unsafe_allow_html=True)
    cols=st.columns(5); presets=[("Balanced","⚖️","Best overall"),("Low Cost","◈","Protect budget"),("High Quality","✦","Max quality"),("Fast Delivery","↗","Meet deadline"),("Low Risk","◉","Reduce risk")]
    for col,(name,icon,desc) in zip(cols,presets):
        active=st.session_state["smart_priority"]==name
        if col.button(f"{icon}  {name} · {desc}",key=f"preset_{name}",use_container_width=True,type="primary" if active else "secondary"):
            st.session_state["smart_priority"]=name
    st.markdown('<div class="smart-section-label">02 · YOUR REQUIREMENTS</div>',unsafe_allow_html=True)
    left,right=st.columns([1.08,1],gap="large")
    with left:
        st.markdown('<div class="form-mini-title">PRODUCT & VOLUME</div>',unsafe_allow_html=True)
        category=st.selectbox("Product category",categories,key="smart_category")
        quantity=st.number_input("Required quantity",min_value=1,step=100,key="smart_quantity",help="Minimum supplier capacity considered in the matching score.")
        deadline=st.number_input("Maximum acceptable delivery delay (days)",min_value=1,step=1,key="smart_deadline")
    with right:
        st.markdown('<div class="form-mini-title">COMMERCIAL & QUALITY</div>',unsafe_allow_html=True)
        budget=st.number_input("Maximum budget per unit (USD)",min_value=0.0,step=5.0,key="smart_budget")
        min_quality=st.slider("Minimum supplier quality",1.0,5.0,step=0.1,key="smart_quality")
        top_n=st.slider("Number of Suppliers",3,10,key="smart_top_n")
    preview=filter_supplier_options(metrics_df,category,budget,min_quality,deadline,quantity)
    if preview.empty:
        preview_count=0; preview_text="No exact matches yet — loosen one requirement to unlock candidates."; tone="warning"
    else:
        preview_count=int(preview["supplier"].nunique()); median_price=float(pd.to_numeric(preview["avg_unit_price"],errors="coerce").median()); median_delay=float(pd.to_numeric(preview["avg_delay"],errors="coerce").median()); preview_text=f"{preview_count} supplier(s) pass your hard filters · typical unit price ${median_price:,.2f} · typical delay {median_delay:.1f} days"; tone="good"
    st.markdown(f"""<div class="smart-preview {tone}"><div class="preview-score"><strong>{preview_count}</strong><span>eligible<br>suppliers</span></div><div><b>Live match preview</b><p>{preview_text}</p></div><div class="preview-pill">{st.session_state["smart_priority"]}</div></div>""",unsafe_allow_html=True)
    submitted=st.button("✦  Find my best supplier matches",use_container_width=True,type="primary")
    if submitted:
        priority=st.session_state["smart_priority"]; results=recommend_suppliers(metrics_df,category,quantity,budget,min_quality,deadline,priority,top_n)
        log_doc={"username":username,"category":category,"quantity":quantity,"budget":budget,"min_quality":min_quality,"deadline":deadline,"priority":priority,"requested_supplier_count":top_n,"result_count":len(results),"selected_supplier":None,"status":"recommended","created_at":datetime.now(timezone.utc)}
        inserted=db[COLLECTIONS["recommendation_logs"]].insert_one(log_doc); st.session_state["last_recommendation_id"]=str(inserted.inserted_id); st.session_state["last_results"]=results.to_dict("records"); st.session_state["last_search_brief"]={"category":category,"quantity":quantity,"budget":budget,"quality":min_quality,"deadline":deadline,"priority":priority,"requested_supplier_count":top_n}
        if results.empty:
            st.error("No supplier satisfies all requirements."); category_options=metrics_df[metrics_df["product_category"].str.lower()==category.lower()].copy()
            if not category_options.empty:
                st.info("Try increasing the budget or deadline, or lowering the minimum quality requirement."); ui_dataframe(safe_metric_table(category_options.sort_values("supplier_rank_score",ascending=False),["supplier","product_category","avg_unit_price","avg_delay","quality_rating","final_rating","risk_level","supplier_rank_score"]),width="stretch")
            return
        if len(results) < top_n:
            st.warning(f"Only {len(results)} supplier(s) match your requirements, so the system cannot show all {top_n} requested suppliers.")
        else:
            st.success(f"Smart match complete — showing {len(results)} of {top_n} requested suppliers.")
    results=pd.DataFrame(st.session_state.get("last_results",[]))
    if results.empty:
        st.markdown("""<div class="empty-smart-state"><span>✦</span><h3>Your shortlist will appear here</h3><p>Choose your strategy and requirements above, then let SupplyLogix rank the best-fit suppliers.</p></div>""",unsafe_allow_html=True); return
    brief=st.session_state.get("last_search_brief",{}); st.markdown('<div class="smart-section-label">03 · MOST MATCHED REQUIREMENT</div>',unsafe_allow_html=True)
    explain_best_requirement_match(results.iloc[0], brief, len(results))
    current_hot=hot_supplier_keys(db,username)
    for rank,(_,row) in enumerate(results.iterrows(),start=1):
        score=float(row["final_score"]); risk=str(row["risk_level"])
        with st.container(border=True):
            a,b=st.columns([3.6,1])
            with a: st.markdown(f"""<div class="supplier-result-head"><div class="rank-badge">#{rank}</div><div><h3>{row["supplier"]}</h3><p>{row["product_category"]} · {row["trend_status"]} trend</p></div></div>""",unsafe_allow_html=True)
            with b: st.markdown(f"""<div class="match-score"><span>SMART MATCH</span><strong>{score:.0f}</strong><small>/100</small></div>""",unsafe_allow_html=True)
            q1,q2,q3,q4=st.columns(4); q1.metric("Rating",f"{float(row['final_rating']):.1f}/5"); q2.metric("Risk",f"{risk} · {float(row['risk_score']):.0f}"); q3.metric("Avg delay",f"{float(row['avg_delay']):.1f} d"); q4.metric("Unit price",f"${float(row['avg_unit_price']):,.2f}")
            st.markdown(f"""<div class="reason-card"><b>Why this supplier?</b><span>{row["explanation"]}</span></div>""",unsafe_allow_html=True)
            select_col,hot_col=st.columns([5,1])
            if select_col.button(f"Select {row['supplier']}",key=f"select_{row['supplier']}_{row['product_category']}",use_container_width=True):
                save_selected_supplier(db,username,row,"smart_recommendation"); st.success(f"Selected {row['supplier']}. You can rate it after the experience.")
            fav=(str(row["supplier"]),str(row["product_category"])) in current_hot
            if hot_col.button("★ Saved" if fav else "☆ Save",key=f"hot_{row['supplier']}_{row['product_category']}",use_container_width=True):
                if fav: remove_hot_supplier(db,username,row["supplier"],row["product_category"]); log_activity(db,"favourite_supplier_removed",username,{"supplier":row["supplier"],"category":row["product_category"]})
                else: save_hot_supplier(db,username,row); log_activity(db,"favourite_supplier_saved",username,{"supplier":row["supplier"],"category":row["product_category"]})
                st.rerun()
    st.markdown('<div class="smart-section-label">04 · DECISION COMPARISON</div>',unsafe_allow_html=True); ui_dataframe(safe_metric_table(results,["supplier","product_category","final_score","final_rating","user_rating","risk_level","risk_score","avg_delay","avg_unit_price","trend_status"]),width="stretch")


def page_risk_analysis(db):
    """Supplier Risk Analysis System based on calculated supplier metrics."""
    page_header("Risk Analysis")

    metrics_df = load_supplier_metrics(db)
    if metrics_df.empty:
        st.warning("No supplier metrics are available yet.")
        st.info("Ask an admin to upload and clean supplier data first.")
        return

    df = metrics_df.copy()
    numeric_columns = [
        "risk_score", "supplier_rank_score", "supplier_kpi_score",
        "final_rating", "user_rating", "avg_delay", "reliability",
        "disruption_frequency", "supply_risk_rate", "avg_severity",
        "bad_feedback_count", "total_orders", "avg_unit_price",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    st.markdown(
        """
        <div class="page-hero">
            <div class="page-hero-top">
                <div class="ui-kicker">RISK INTELLIGENCE</div>
                <h1>Understand supplier risk<br>before you buy.</h1>
                <p>Review predicted risk, operational drivers, feedback signals and supplier exposure in one decision-ready view.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    categories = sorted(df["product_category"].dropna().astype(str).unique()) if "product_category" in df.columns else []
    filter_left, filter_mid, filter_right = st.columns([1.2, 1.2, 1])
    with filter_left:
        selected_category = st.selectbox(
            "Product category",
            ["All categories"] + categories,
            key="risk_category_filter",
        )
    with filter_mid:
        selected_level = st.multiselect(
            "Risk level",
            ["Low", "Medium", "High"],
            default=["Low", "Medium", "High"],
            key="risk_level_filter",
        )
    with filter_right:
        min_score = st.slider(
            "Minimum risk score",
            0, 100, 0,
            key="risk_min_score",
        )

    filtered = df.copy()
    if selected_category != "All categories":
        filtered = filtered[filtered["product_category"].astype(str) == selected_category]
    if selected_level:
        filtered = filtered[filtered["risk_level"].astype(str).isin(selected_level)]
    else:
        filtered = filtered.iloc[0:0]
    filtered = filtered[filtered["risk_score"] >= min_score].copy()

    total_suppliers = int(filtered["supplier"].nunique()) if "supplier" in filtered.columns else len(filtered)
    high_count = int((filtered["risk_level"] == "High").sum()) if "risk_level" in filtered.columns else 0
    medium_count = int((filtered["risk_level"] == "Medium").sum()) if "risk_level" in filtered.columns else 0
    avg_risk = float(filtered["risk_score"].mean()) if not filtered.empty else 0
    avg_rating = float(filtered["final_rating"].mean()) if not filtered.empty and "final_rating" in filtered.columns else 0

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Suppliers analyzed", total_suppliers)
    k2.metric("High risk", high_count)
    k3.metric("Medium risk", medium_count)
    k4.metric("Average risk", f"{avg_risk:.0f}/100")
    k5.metric("Average rating", f"{avg_rating:.1f}/5")

    if filtered.empty:
        st.info("No suppliers match the current risk filters.")
        return

    left, right = st.columns(2)

    with left:
        st.subheader("Risk Distribution")
        level_counts = (
            filtered["risk_level"]
            .value_counts()
            .reindex(["Low", "Medium", "High"], fill_value=0)
            .rename_axis("Risk level")
            .reset_index(name="Suppliers")
        )
        fig = px.bar(
            level_counts,
            x="Risk level",
            y="Suppliers",
            text="Suppliers",
            title="Supplier risk levels",
        )
        fig.update_layout(margin=dict(l=10, r=10, t=55, b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Risk Score by Supplier")
        plot_df = filtered.sort_values("risk_score", ascending=False).head(15)
        fig = px.bar(
            plot_df,
            x="risk_score",
            y="supplier",
            color="risk_level",
            orientation="h",
            hover_data=[
                c for c in ["product_category", "avg_delay", "reliability", "final_rating"]
                if c in plot_df.columns
            ],
            title="Highest-risk supplier profiles",
        )
        fig.update_layout(
            margin=dict(l=10, r=10, t=55, b=10),
            yaxis={"categoryorder": "total ascending"},
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Risk Driver Analysis")

    supplier_options = filtered.apply(
        lambda row: f"{row['supplier']} · {row['product_category']}",
        axis=1,
    ).tolist()
    selected_supplier = st.selectbox(
        "Supplier to analyze",
        supplier_options,
        key="risk_supplier_select",
    )

    selected_mask = filtered.apply(
        lambda row: f"{row['supplier']} · {row['product_category']}",
        axis=1,
    ) == selected_supplier
    row = filtered[selected_mask].iloc[0]

    risk_score_value = float(row.get("risk_score", 0))
    risk_level_value = str(row.get("risk_level", "Unknown"))

    if risk_level_value == "High":
        st.error(f"{selected_supplier}: HIGH RISK — {risk_score_value:.0f}/100")
    elif risk_level_value == "Medium":
        st.warning(f"{selected_supplier}: MEDIUM RISK — {risk_score_value:.0f}/100")
    else:
        st.success(f"{selected_supplier}: LOW RISK — {risk_score_value:.0f}/100")

    reliability = float(row.get("reliability", 0))
    avg_delay = float(row.get("avg_delay", 0))
    disruption_frequency = float(row.get("disruption_frequency", 0))
    supply_risk_rate = float(row.get("supply_risk_rate", 0))
    avg_severity = float(row.get("avg_severity", 0))
    user_rating = float(row.get("user_rating", 0))
    bad_feedback = int(row.get("bad_feedback_count", 0))

    driver_rows = [
        {
            "Risk driver": "Reliability",
            "Value": f"{reliability * 100:.1f}%",
            "Interpretation": "Lower reliability increases predicted risk.",
        },
        {
            "Risk driver": "Average delay",
            "Value": f"{avg_delay:.1f} days",
            "Interpretation": "Longer delivery delays increase operational exposure.",
        },
        {
            "Risk driver": "Disruption frequency",
            "Value": f"{disruption_frequency * 100:.1f}%",
            "Interpretation": "Frequent disruptions increase supply risk.",
        },
        {
            "Risk driver": "Supply-risk flag rate",
            "Value": f"{supply_risk_rate * 100:.1f}%",
            "Interpretation": "Historical supply-risk flags increase the risk score.",
        },
        {
            "Risk driver": "Average severity",
            "Value": f"{avg_severity:.2f}",
            "Interpretation": "More severe incidents increase exposure.",
        },
        {
            "Risk driver": "User rating",
            "Value": f"{user_rating:.1f}/5",
            "Interpretation": "Lower feedback ratings increase predicted risk.",
        },
        {
            "Risk driver": "Bad feedback",
            "Value": str(bad_feedback),
            "Interpretation": "Ratings of 2 or below are treated as negative feedback.",
        },
    ]
    ui_dataframe(pd.DataFrame(driver_rows), width="stretch")

    st.subheader("Risk Monitoring Table")
    risk_columns = [
        "supplier", "product_category", "risk_score", "risk_level",
        "supplier_rank_score", "final_rating", "avg_delay", "reliability",
        "disruption_frequency", "supply_risk_rate", "trend_status",
    ]
    ui_dataframe(
        safe_metric_table(
            filtered.sort_values("risk_score", ascending=False),
            risk_columns,
        ),
        width="stretch",
    )

    st.subheader("Recommended Risk Actions")
    actions = []
    if avg_delay > 5:
        actions.append("Review delivery commitments and consider a backup supplier.")
    elif avg_delay > 2:
        actions.append("Monitor delivery performance closely against the agreed deadline.")
    if reliability < 0.8:
        actions.append("Request a supplier reliability improvement plan.")
    if disruption_frequency > 0.10:
        actions.append("Review recent disruption causes and contingency capacity.")
    if supply_risk_rate > 0.10:
        actions.append("Increase monitoring for supply interruptions.")
    if user_rating < 3:
        actions.append("Review recent user feedback before increasing purchase volume.")
    if bad_feedback > 0:
        actions.append("Investigate negative feedback events and corrective actions.")
    if not actions:
        actions.append("No major risk-driver warning was detected. Continue routine monitoring.")

    for action in actions:
        st.markdown(f"- {action}")


def page_rate_supplier(db):
    page_header("Rate Supplier")
    selected = st.session_state.get("selected_supplier")
    history = load_collection(db, "recommendation_logs")
    user = st.session_state["user"]["username"]
    required_history_cols = {"username", "status", "supplier", "product_category"}
    if not history.empty and required_history_cols.issubset(history.columns):
        selected_rows = history[(history["username"] == user) & (history["status"] == "selected")]
    else:
        selected_rows = pd.DataFrame()
    options = []
    if selected:
        options.append(f"{selected['supplier']} | {selected['product_category']}")
    if not selected_rows.empty:
        options.extend((selected_rows["supplier"] + " | " + selected_rows["product_category"]).dropna().unique().tolist())
    options = sorted(set(options))
    if not options:
        st.warning("Select a supplier from recommendations before rating.")
        return

    with st.form("rating_form"):
        choice = st.selectbox("Selected supplier", options)
        supplier, category = [part.strip() for part in choice.split("|", 1)]
        rating = st.slider("Rating", 1, 5, 4)
        event_type = st.selectbox("Event Type", EVENT_TYPES)
        comment = st.text_area("Comment")
        submitted = st.form_submit_button("Submit Rating")
    if submitted:
        doc = {
            "username": user,
            "supplier": supplier,
            "product_category": category,
            "rating": rating,
            "event_type": event_type,
            "comment": comment,
            "created_at": datetime.now(timezone.utc),
        }
        db[COLLECTIONS["supplier_ratings"]].insert_one(doc)
        log_activity(db, "supplier_rated", user, {"supplier": supplier, "category": category, "rating": rating, "event_type": event_type})
        metrics_df = refresh_metrics(db)
        sync_supplier_saved_scores(db, supplier, category, metrics_df)
        st.success("Rating saved. Supplier score and future recommendations now use this feedback.")


def page_user_history(db):
    page_header("My History")
    user = st.session_state["user"]["username"]
    logs = dataframe_from_collection(db, COLLECTIONS["recommendation_logs"], {"username": user})
    ratings = dataframe_from_collection(db, COLLECTIONS["supplier_ratings"], {"username": user})
    favourites = dataframe_from_collection(db, HOT_SUPPLIERS_COLLECTION, {"username": user})
    metrics_df = load_supplier_metrics(db)

    def add_supplier_details(history_df, use_rank_as_final_score=False):
        if history_df.empty or metrics_df.empty:
            return history_df
        metric_cols = [
            "supplier",
            "product_category",
            "supplier_rank_score",
            "final_rating",
            "risk_level",
            "risk_score",
            "avg_delay",
            "avg_unit_price",
        ]
        available_metric_cols = [col for col in metric_cols if col in metrics_df.columns]
        details = metrics_df[available_metric_cols].copy()
        enriched = history_df.merge(details, on=["supplier", "product_category"], how="left", suffixes=("", "_metric"))
        for col in ["final_rating", "risk_level", "risk_score", "avg_delay", "avg_unit_price"]:
            metric_col = f"{col}_metric"
            if metric_col in enriched.columns:
                if col in enriched.columns:
                    enriched[col] = enriched[col].combine_first(enriched[metric_col])
                else:
                    enriched[col] = enriched[metric_col]
                enriched = enriched.drop(columns=[metric_col])
        if use_rank_as_final_score and "supplier_rank_score" in enriched.columns:
            enriched["final_score"] = enriched["supplier_rank_score"]
        return enriched.drop(columns=["supplier_rank_score"], errors="ignore")


    st.subheader("Selected Suppliers")
    if logs.empty or "status" not in logs.columns:
        st.info("No selected suppliers yet.")
    else:
        selected_logs = logs[logs["status"] == "selected"].copy()
        if selected_logs.empty:
            st.info("No selected suppliers yet.")
        else:
            selected_logs = add_supplier_details(selected_logs, use_rank_as_final_score=True)
            if "created_at" in selected_logs.columns:
                selected_logs = selected_logs.sort_values("created_at", ascending=False)
            ui_dataframe(
                safe_metric_table(
                    selected_logs,
                    ["supplier", "product_category", "final_score", "final_rating", "risk_level", "risk_score", "avg_delay", "avg_unit_price", "created_at"],
                ),
                width="stretch",
            )
    st.subheader("Ratings Given")
    if ratings.empty:
        st.info("No ratings given yet.")
    else:
        ratings = add_supplier_details(ratings, use_rank_as_final_score=True)
        if "created_at" in ratings.columns:
            ratings = ratings.sort_values("created_at", ascending=False)
        ui_dataframe(
            safe_metric_table(
                ratings,
                ["supplier", "product_category", "rating", "event_type", "comment", "created_at"],
            ),
            width="stretch",
        )


def smart_sidebar_nav(title, pages, icons, descriptions, current_page, state_key):
    """Premium button-based sidebar navigation with no radio controls."""
    st.markdown('<div class="smart-nav">', unsafe_allow_html=True)
    st.markdown(
        f'<div class="smart-nav-section"><span>{title}</span><span>{len(pages)} sections</span></div>',
        unsafe_allow_html=True,
    )
    for idx, item in enumerate(pages):
        icon = icons.get(item, "•")
        desc = descriptions.get(item, "Open section")
        if item == current_page:
            st.markdown(
                f"""<div class="smart-nav-item smart-nav-active">
                    <div class="nav-icon">{icon}</div>
                    <div class="nav-copy"><div class="nav-title">{item}</div><div class="nav-desc">{desc}</div></div>
                    <div class="nav-arrow">◆</div>
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="smart-nav-button">', unsafe_allow_html=True)
            if st.button(f"{icon}   {item}", key=f"smart_nav_{state_key}_{idx}", use_container_width=True, help=desc):
                st.session_state[state_key] = item
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def render_app():
    inject_custom_css()
    db = get_database()
    if db is None:
        st.error("MongoDB is not connected. Start MongoDB or check MONGODB_URI in .env.")
        st.stop()
    ensure_default_users(db)

    if "user" not in st.session_state:
        if not restore_login_session(db):
            page_login(db)
            return

    user = st.session_state["user"]
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="brand-icon">🚚</div>
                <div>
                    <div class="brand-name">Supply<span>Logix</span></div>
                    <div class="brand-subtitle">AI Supplier Intelligence</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(f"Signed in as **{user['username']}** · {user['role'].title()}")
        if st.button("↪  Sign out", use_container_width=True):
            clear_login_session(db)
            st.session_state.clear()
            st.rerun()
        if user["role"] == "admin":
            st.markdown('<div class="sidebar-role-badge">🛠️ Admin Workspace<small>Data control, analytics and accounts</small></div>', unsafe_allow_html=True)
            pages = ["Dashboard", "Upload Data", "Clean Data", "View Data", "EDA & KPI", "User Rating", "Manage Accounts"]
            page_icons = {
                "Dashboard": "⌂", "Upload Data": "↑", "Clean Data": "✦",
                "View Data": "▦", "EDA & KPI": "◒", "User Rating": "★", "Manage Accounts": "♙",
            }
            page_desc = {
                "Dashboard": "Overview & live alerts", "Upload Data": "Import supplier data",
                "Clean Data": "Prepare analytics-ready data", "View Data": "Browse operational records",
                "EDA & KPI": "Explore performance metrics", "User Rating": "Review feedback signals",
                "Manage Accounts": "Users & approvals",
            }
            if st.session_state.get("admin_nav_target") in pages:
                st.session_state["admin_page"] = st.session_state.pop("admin_nav_target")
            if st.session_state.get("admin_page") not in pages:
                st.session_state["admin_page"] = "Dashboard"
            smart_sidebar_nav("Admin workspace", pages, page_icons, page_desc, st.session_state["admin_page"], "admin_page")
            page = st.session_state["admin_page"]
        elif user["role"] == "supplier":
            st.markdown('<div class="sidebar-role-badge">🚚 Supplier Workspace<small>Performance, trends and forecasting</small></div>', unsafe_allow_html=True)
            supplier_pages = ["Supplier Dashboard", "Supplier Trend", "Future Prediction"]
            supplier_icons = {"Supplier Dashboard": "◉", "Supplier Trend": "↗", "Future Prediction": "✦"}
            supplier_desc = {
                "Supplier Dashboard": "Your performance snapshot",
                "Supplier Trend": "Ratings & delivery trends",
                "Future Prediction": "Forecast future performance",
            }
            if st.session_state.get("supplier_page") not in supplier_pages:
                st.session_state["supplier_page"] = supplier_pages[0]
            smart_sidebar_nav("Supplier workspace", supplier_pages, supplier_icons, supplier_desc, st.session_state["supplier_page"], "supplier_page")
            page = st.session_state["supplier_page"]
        else:
            st.markdown('<div class="sidebar-role-badge">👤 Procurement Workspace<small>Discover, rate and manage suppliers</small></div>', unsafe_allow_html=True)
            user_pages = ["Home", "Supplier Recommendation", "Rate Supplier", "My History"]
            user_icons = {"Home": "⌂", "Supplier Recommendation": "✦", "Risk Analysis": "◉", "Rate Supplier": "★", "My History": "◷"}
            user_desc = {
                "Home": "Your supplier intelligence hub",
                "Supplier Recommendation": "AI-ranked supplier shortlist",
                "Rate Supplier": "Share performance feedback",
                "My History": "Saved suppliers & activity",
            }
            if st.session_state.get("user_page") not in user_pages:
                st.session_state["user_page"] = user_pages[0]
            smart_sidebar_nav("Procurement workspace", user_pages, user_icons, user_desc, st.session_state["user_page"], "user_page")
            page = st.session_state["user_page"]
        st.markdown(
            """
            <div class="sidebar-footer">
                <b>SUPPLYLOGIX</b>
                <span>Supplier intelligence · risk · recommendations</span>
            </div>
            """, unsafe_allow_html=True,
        )

    if page == "Dashboard":
        page_admin_dashboard(db)
    elif page == "Upload Data":
        page_upload(db)
    elif page == "Clean Data":
        page_clean(db)
    elif page == "View Data":
        page_view_data(db)
    elif page == "EDA & KPI":
        page_eda(db)
    elif page == "User Rating":
        page_ratings_feedback(db)
    elif page == "Manage Accounts":
        page_manage_users(db)
    elif page == "Supplier Dashboard":
        page_supplier_dashboard(db)
    elif page == "Supplier Trend":
        page_supplier_trend(db)
    elif page == "Future Prediction":
        page_supplier_future_prediction(db)
    elif page == "Home":
        page_user_home(db)
    elif page == "Supplier Recommendation":
        page_find_supplier(db)
    elif page == "Rate Supplier":
        page_rate_supplier(db)
    elif page == "My History":
        page_user_history(db)


if __name__ == "__main__":
    render_app()
