#!/usr/bin/env python3
"""
Crypto Market & Liquidity Risk Dashboard
Université de Genève · Quantitative Risk Management (Spring 26) · Group 3
Render.com deployment (standard Dash, no JupyterDash)
"""

import os, time, traceback
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, html, dcc, Input, Output, State, ALL, dash_table
import dash_bootstrap_components as dbc

TRADING_DAYS = 365
BRAND        = "#0B3D91"
DEFAULT_TICKERS   = ["BTC-USD","ETH-USD","SOL-USD","BNB-USD","XRP-USD"]
DEFAULT_POSITIONS = {"BTC":5_000_000,"ETH":3_000_000,"SOL":1_000_000,"BNB":800_000,"XRP":200_000}
LIQ_PARTICIPATION = 0.25
DEFAULT_ALPHA     = 0.95
DEFAULT_LAMBDA    = 0.5

SCENARIOS = {
    "Crypto Winter (2022-style)":{"shocks":{"BTC":-0.50,"ETH":-0.60,"SOL":-0.75,"BNB":-0.55,"XRP":-0.55},"adv_mult":0.40,"lambda_mult":1.5,"vol_mult":1.5,"description":"Prolonged bear market: large drops, volumes contract ~60%."},
    "Exchange Collapse (FTX-style)":{"shocks":{"BTC":-0.25,"ETH":-0.30,"SOL":-0.50,"BNB":-0.55,"XRP":-0.30},"adv_mult":0.20,"lambda_mult":3.0,"vol_mult":2.5,"description":"Sudden venue failure, liquidity evaporates."},
    "Stablecoin Panic":{"shocks":{"BTC":-0.15,"ETH":-0.20,"SOL":-0.35,"BNB":-0.25,"XRP":-0.20},"adv_mult":0.50,"lambda_mult":2.0,"vol_mult":1.8,"description":"Loss of confidence in USD stablecoin."},
    "Regulatory Shock":{"shocks":{"BTC":-0.20,"ETH":-0.25,"SOL":-0.40,"BNB":-0.45,"XRP":-0.50},"adv_mult":0.60,"lambda_mult":1.8,"vol_mult":2.2,"description":"Major-jurisdiction enforcement action."},
}

METRIC_EXPLANATIONS = {
    "Volatility":"How strongly prices fluctuate.","VaR":"Loss not exceeded with chosen confidence level over 1 day.",
    "ES":"Expected Shortfall — average loss in the tail beyond VaR.","Max Drawdown":"Largest historical peak-to-trough decline.",
    "L-VaR":"Liquidity-Adjusted VaR = Std VaR + Liquidity Cost (BDSS 1999).","Cornish-Fisher":"VaR adjusted for skewness and kurtosis.",
    "Component VaR":"Each asset's contribution to portfolio VaR.","Bootstrap CI":"95% CI around historical VaR (1 000 draws).","Kupiec POF":"Backtest: p > 0.05 → model adequate.",
}

def _fetch_one(symbol,start,end,attempts=3,sleep=1.5):
    for k in range(attempts):
        try:
            df=yf.download(symbol,start=start,end=end,progress=False,auto_adjust=True)
            if df is not None and not df.empty: return df
        except Exception: pass
        time.sleep(sleep*(k+1))
    return None

def load_features(tickers,start,end):
    frames,missing=[],[]
    for s in tickers:
        df=_fetch_one(s,start,end)
        if df is None or df.empty: missing.append(s); continue
        sub=df.reset_index()
        sub.columns=[c if not isinstance(c,tuple) else c[0] for c in sub.columns]
        sub=sub.rename(columns={"Date":"date","Close":"close","Volume":"volume"})
        sub["asset"]=s.split("-")[0]; sub["volume_usd"]=sub["volume"]
        frames.append(sub[["date","asset","close","volume","volume_usd"]])
    if not frames: return None,missing
    raw=pd.concat(frames,ignore_index=True)
    raw["date"]=pd.to_datetime(raw["date"])
    raw=raw.sort_values(["asset","date"]).reset_index(drop=True)
    g=raw.groupby("asset",group_keys=False)
    raw["return"]=g["close"].pct_change()
    raw["norm_price"]=g["close"].transform(lambda s:s/s.iloc[0])
    raw["rolling_vol_30"]=(g["return"].rolling(30).std().reset_index(level=0,drop=True)*np.sqrt(TRADING_DAYS))
    raw["cum_max"]=g["close"].cummax(); raw["drawdown"]=raw["close"]/raw["cum_max"]-1.0
    raw["adv_30_usd"]=g["volume_usd"].rolling(30).mean().reset_index(level=0,drop=True)
    return raw,missing

def _arr(x): r=np.asarray(pd.Series(x),dtype=float); return r[~np.isnan(r)]
def hist_var(x,c=0.95): r=_arr(x); return float(-np.quantile(r,1-c)) if r.size else np.nan
def parametric_var(x,c=0.95):
    r=_arr(x); return np.nan if r.size<2 else float(-(r.mean()+r.std(ddof=1)*stats.norm.ppf(1-c)))
def cornish_fisher_var(x,c=0.95):
    r=_arr(x)
    if r.size<4: return np.nan
    s=float(stats.skew(r,bias=False)); k=float(stats.kurtosis(r,fisher=True,bias=False)); z=stats.norm.ppf(1-c)
    z_cf=z+(z**2-1)*s/6+(z**3-3*z)*k/24-(2*z**3-5*z)*(s**2)/36
    return float(-(r.mean()+r.std(ddof=1)*z_cf))
def expected_shortfall(x,c=0.95):
    r=_arr(x); q=np.quantile(r,1-c); tail=r[r<=q]; return float(-tail.mean()) if tail.size else np.nan
def ann_vol(x): r=_arr(x); return float(r.std(ddof=1)*np.sqrt(TRADING_DAYS)) if r.size>1 else np.nan
def max_drawdown(prices): p=pd.Series(prices).dropna(); return float((p/p.cummax()-1).min()) if not p.empty else np.nan
def kupiec_pof(returns,c=0.95,window=250):
    r=_arr(returns)
    if r.size<=window+5: return np.nan,np.nan,np.nan
    breaches=n=0
    for i in range(window,len(r)):
        v=hist_var(r[i-window:i],c)
        if not np.isnan(v) and r[i]<-v: breaches+=1
        n+=1
    if n==0 or breaches in (0,n): return (breaches/n if n else np.nan),np.nan,np.nan
    p_hat,p=breaches/n,1-c
    lr=-2*(np.log(((1-p)**(n-breaches))*(p**breaches))-np.log(((1-p_hat)**(n-breaches))*(p_hat**breaches)))
    return p_hat,float(lr),float(1-stats.chi2.cdf(lr,df=1))
def component_var_normal(rets_df,weights,c=0.95):
    rets=rets_df.dropna()
    if rets.empty: return None
    cov=rets.cov().values; w=np.asarray(weights,dtype=float); pv=float(w@cov@w)
    if pv<=0: return None
    cvar=w*((cov@w)/np.sqrt(pv))*(-stats.norm.ppf(1-c)); denom=np.abs(cvar).sum()
    pct=cvar/denom*100 if denom>1e-12 else np.zeros_like(cvar)
    return pd.DataFrame({"Asset":rets.columns,"Component VaR ($)":cvar,"% of Portfolio VaR":pct})
def bootstrap_var_ci(returns,c=0.95,n_boot=1000,seed=42):
    r=_arr(returns)
    if r.size<30: return np.nan,np.nan
    rng=np.random.default_rng(seed); boot=rng.choice(r,size=(n_boot,r.size),replace=True)
    vars_=-np.quantile(boot,1-c,axis=1)
    return float(np.quantile(vars_,0.025)),float(np.quantile(vars_,0.975))
def liquidity_cost(position_usd,adv_usd,sigma,lam=0.5):
    if adv_usd is None or adv_usd<=0 or np.isnan(adv_usd) or np.isnan(sigma): return 0.0
    return float(lam*(position_usd/adv_usd)*sigma*position_usd)
def portfolio_returns_from(df,holdings):
    total=sum(holdings.values()) or 1.0; weights={a:v/total for a,v in holdings.items()}
    pivot=df.pivot_table(index="date",columns="asset",values="return")
    common=[a for a in weights if a in pivot.columns]
    if not common: return pd.Series(dtype=float)
    w=np.array([weights[a] for a in common])
    return (pivot[common]*w).sum(axis=1).dropna()
def rolling_metric(returns,window=250,alpha=0.95,kind="var"):
    fn=hist_var if kind=="var" else expected_shortfall
    return returns.rolling(window).apply(lambda x:fn(x,alpha),raw=True)
def money(x):
    if x is None or (isinstance(x,float) and np.isnan(x)): return "—"
    a=abs(x)
    if a>=1e9: return f"${x/1e9:.2f}B"
    if a>=1e6: return f"${x/1e6:.2f}M"
    if a>=1e3: return f"${x/1e3:.1f}K"
    return f"${x:,.0f}"
def pctf(x,d=2):
    if x is None or (isinstance(x,float) and np.isnan(x)): return "—"
    return f"{x*100:.{d}f}%"

END=pd.Timestamp.today().normalize(); START=END-pd.Timedelta(days=365*3)
print(f"Loading {START.date()} → {END.date()} …")
FEATURES,MISSING=load_features(DEFAULT_TICKERS,START,END)
if FEATURES is None: raise RuntimeError("All downloads failed.")
ASSETS=sorted(FEATURES["asset"].unique().tolist())
DEFAULT_HOLDINGS={a:DEFAULT_POSITIONS.get(a,0) for a in ASSETS}
print(f"✓ {ASSETS}")

def kpi_card(label,value,sub=None,color=BRAND):
    return dbc.Card(dbc.CardBody([
        html.Div(label,style={"fontSize":"0.72rem","color":"#666","textTransform":"uppercase","letterSpacing":"0.5px"}),
        html.Div(value,style={"fontSize":"1.5rem","fontWeight":700,"color":color,"marginTop":"4px"}),
        html.Div(sub or "",style={"fontSize":"0.72rem","color":"#888"}),
    ]),className="shadow-sm h-100",style={"borderLeft":f"4px solid {color}","borderRadius":"6px"})
def section_card(title,children,icon=""):
    return dbc.Card([dbc.CardHeader(html.H5(f"{icon} {title}",className="mb-0",style={"color":BRAND})),dbc.CardBody(children)],className="mb-3 shadow-sm")
def info_alert(items):
    return dbc.Alert([html.Div([html.Strong(f"{k}: "),v]) for k,v in items],color="light",className="mb-3",style={"fontSize":"0.85rem"})

app=Dash(__name__,external_stylesheets=[dbc.themes.FLATLY],title="Crypto Risk Dashboard — UNIGE",suppress_callback_exceptions=True)
server=app.server  # for gunicorn

header=dbc.Card(dbc.CardBody([
    html.Div("Université de Genève · Geneva School of Economics and Management",style={"fontSize":"0.8rem","color":"#bbd"}),
    html.H2("Crypto Market & Liquidity Risk Dashboard",style={"color":"white","marginTop":"4px","marginBottom":"4px"}),
    html.Div(["Quantitative Risk Management (Spring 26) · ",html.B("Group 3")],style={"fontSize":"0.9rem","color":"#dde6f5"}),
]),style={"background":BRAND,"marginBottom":"16px","borderRadius":"6px"})

def portfolio_inputs():
    return [dbc.Row([
        dbc.Col(html.Label(a,style={"fontWeight":600,"fontSize":"0.85rem"}),width=4),
        dbc.Col(dbc.Input(id={"type":"hold","index":a},type="number",min=0,step=10000,value=float(DEFAULT_HOLDINGS[a]),size="sm"),width=8),
    ],className="mb-1") for a in ASSETS]

sidebar=dbc.Card(dbc.CardBody([
    html.H6("📦 Portfolio holdings (USD)",style={"color":BRAND}),
    html.Small("Edit any value — all charts re-compute live.",className="text-muted"),
    html.Div(portfolio_inputs(),style={"marginTop":"8px"}),html.Hr(),
    html.Label("VaR / ES confidence (α)",style={"fontWeight":600,"fontSize":"0.85rem"}),
    dcc.Slider(id="alpha-slider",min=0.90,max=0.99,value=DEFAULT_ALPHA,step=0.01,marks={0.90:"90%",0.95:"95%",0.99:"99%"}),html.Br(),
    html.Label("Liquidity coefficient (λ, BDSS)",style={"fontWeight":600,"fontSize":"0.85rem"}),
    dcc.Slider(id="lambda-slider",min=0.0,max=2.0,value=DEFAULT_LAMBDA,step=0.05,marks={0:"0",0.5:"0.5",1:"1.0",2:"2.0"}),html.Br(),
    html.Label("Stress scenario",style={"fontWeight":600,"fontSize":"0.85rem"}),
    dcc.Dropdown(id="scenario-dd",options=[{"label":k,"value":k} for k in SCENARIOS.keys()],value=list(SCENARIOS.keys())[0],clearable=False,style={"fontSize":"0.85rem"}),
    html.Hr(),html.Div(id="sidebar-summary",style={"fontSize":"0.78rem","color":"#666"}),
]),className="shadow-sm",style={"borderRadius":"6px"})

tabs=dcc.Tabs(id="tabs",value="tab-market",children=[
    dcc.Tab(label="📊 Market Data",value="tab-market"),dcc.Tab(label="📉 VaR & ES",value="tab-var"),
    dcc.Tab(label="💧 Liquidity",value="tab-liq"),dcc.Tab(label="🌪 Crash Scenarios",value="tab-scen"),
    dcc.Tab(label="📚 Methodology",value="tab-meth"),
])

app.layout=dbc.Container([
    header,html.Div(id="kpi-row",className="mb-3"),
    dbc.Row([dbc.Col(sidebar,md=3),dbc.Col([tabs,html.Div(id="tab-content",className="mt-3")],md=9)]),
    html.Div("© 2026 Group 3 · UNIGE · Quantitative Risk Management (Spring 26)",
             style={"textAlign":"center","color":"#999","padding":"20px 0","fontSize":"0.78rem","marginTop":"24px"}),
],fluid=True,style={"maxWidth":"1500px","padding":"20px"})

# Tab renderers (same logic as Nuvolos version)
def render_market(holdings,alpha):
    df=FEATURES.copy(); port_ret=portfolio_returns_from(df,holdings)
    vol_30=port_ret.tail(30).std(ddof=1)*np.sqrt(TRADING_DAYS) if len(port_ret)>=30 else np.nan
    ret_7d=(1+port_ret.tail(7)).prod()-1 if len(port_ret)>=7 else np.nan
    avg_p=df.pivot_table(index="date",columns="asset",values="close").mean(axis=1).dropna()
    mdd=max_drawdown(avg_p); last_r=port_ret.iloc[-1] if not port_ret.empty else np.nan
    risk=("High" if not pd.isna(last_r) and last_r<-0.03 else "Medium" if not pd.isna(last_r) and last_r<-0.015 else "Moderate" if not pd.isna(last_r) else "N/A")
    senti=("Risk-off" if not pd.isna(ret_7d) and ret_7d<-0.10 else "Risk-on" if not pd.isna(ret_7d) and ret_7d>0.10 else "Neutral" if not pd.isna(ret_7d) else "N/A")
    fig_norm=px.line(df,x="date",y="norm_price",color="asset",title="Normalised price",height=320); fig_norm.update_layout(template="plotly_white",legend_title="")
    fig_vol=px.line(df,x="date",y="rolling_vol_30",color="asset",title="30d rolling vol (ann. √365)",height=320); fig_vol.update_layout(template="plotly_white",yaxis_tickformat=".0%",legend_title="")
    fig_dd=px.line(df,x="date",y="drawdown",color="asset",title="Drawdown from peak",height=320); fig_dd.update_layout(template="plotly_white",yaxis_tickformat=".0%",legend_title="")
    corr=df.pivot_table(index="date",columns="asset",values="return").corr().round(2)
    fig_corr=go.Figure(go.Heatmap(z=corr.values,x=corr.columns.tolist(),y=corr.index.tolist(),colorscale="RdBu_r",zmin=-1,zmax=1,text=corr.values,texttemplate="%{text:.2f}"))
    fig_corr.update_layout(title="Correlation matrix",template="plotly_white",height=320)
    rc={"High":"#c0392b","Medium":"#e67e22","Moderate":"#27ae60"}.get(risk,"#999"); sc={"Risk-on":"#27ae60","Risk-off":"#c0392b"}.get(senti,"#999")
    return html.Div([
        dbc.Row([dbc.Col(kpi_card("Risk regime",risk,"1-day signal",rc)),dbc.Col(kpi_card("Vol (30d)",pctf(vol_30,1),"Annualised")),
                 dbc.Col(kpi_card("7-day return",pctf(ret_7d,2),"Trailing")),dbc.Col(kpi_card("Max drawdown",pctf(mdd,1),"Equal-weight")),
                 dbc.Col(kpi_card("Sentiment",senti,"7d return",sc))],className="g-2 mb-3"),
        dbc.Row([dbc.Col(dcc.Graph(figure=fig_norm),md=6),dbc.Col(dcc.Graph(figure=fig_vol),md=6)]),
        dbc.Row([dbc.Col(dcc.Graph(figure=fig_dd),md=6),dbc.Col(dcc.Graph(figure=fig_corr),md=6)]),
    ])

def render_var(holdings,alpha):
    df=FEATURES.copy(); port_ret=portfolio_returns_from(df,holdings).dropna()
    if port_ret.empty: return dbc.Alert("No return data.",color="warning")
    AUM=sum(holdings.values()) or 1.0
    methods={"Historical Simulation":hist_var(port_ret,alpha),"Parametric (Normal)":parametric_var(port_ret,alpha),"Cornish-Fisher":cornish_fisher_var(port_ret,alpha)}
    es=expected_shortfall(port_ret,alpha); ci_lo,ci_hi=bootstrap_var_ci(port_ret.values,alpha); phat,lr_,pval=kupiec_pof(port_ret.values,alpha)
    rows=[]
    for name,v in methods.items():
        rows.append({"Method":name,f"VaR {int(alpha*100)}% (%)":f"{v*100:.2f}" if not np.isnan(v) else "—",f"VaR {int(alpha*100)}% ($)":money(v*AUM) if not np.isnan(v) else "—","Bootstrap 95% CI":f"[{ci_lo*100:.2f}%, {ci_hi*100:.2f}%]" if name=="Historical Simulation" and not np.isnan(ci_lo) else ""})
    rows.append({"Method":f"ES {int(alpha*100)}%",f"VaR {int(alpha*100)}% (%)":f"{es*100:.2f}" if not np.isnan(es) else "—",f"VaR {int(alpha*100)}% ($)":money(es*AUM) if not np.isnan(es) else "—","Bootstrap 95% CI":""})
    fig_h=go.Figure(); fig_h.add_trace(go.Histogram(x=port_ret*100,nbinsx=60,marker_color="#9aa9c2"))
    v0=methods["Historical Simulation"]
    if not np.isnan(v0): fig_h.add_vline(x=-v0*100,line_color="red",line_dash="dash",annotation_text=f"VaR {int(alpha*100)}%")
    if not np.isnan(es): fig_h.add_vline(x=-es*100,line_color="darkred",line_dash="dot",annotation_text=f"ES {int(alpha*100)}%")
    fig_h.update_layout(title="Return distribution",template="plotly_white",height=320,xaxis_title="Daily return (%)",showlegend=False)
    rv=rolling_metric(port_ret,250,alpha,"var")*100; re_=rolling_metric(port_ret,250,alpha,"es")*100
    roll_df=pd.DataFrame({"date":rv.index,"Rolling VaR":rv.values,"Rolling ES":re_.values}).dropna()
    fig_roll=px.line(roll_df,x="date",y=["Rolling VaR","Rolling ES"],title=f"Rolling VaR & ES (250d, {int(alpha*100)}%)")
    fig_roll.update_layout(template="plotly_white",height=320,yaxis_title="Loss (%)",legend_title="")
    per=[]
    for a in ASSETS:
        sub=df[df["asset"]==a].sort_values("date"); r=sub["return"].dropna()
        if r.empty: continue
        per.append({"Asset":a,"Last price":f"${sub['close'].iloc[-1]:,.2f}",f"VaR {int(alpha*100)}% (%)":f"{hist_var(r,alpha)*100:.2f}",f"ES {int(alpha*100)}% (%)":f"{expected_shortfall(r,alpha)*100:.2f}","Max DD":f"{max_drawdown(sub['close'])*100:.1f}%","Ann. vol":f"{ann_vol(r)*100:.0f}%"})
    pivot=df.pivot_table(index="date",columns="asset",values="return").dropna()
    w=np.array([holdings.get(a,0)/AUM for a in pivot.columns]); cvar_df=component_var_normal(pivot,w,alpha)
    fig_cvar=(px.bar(cvar_df,x="Asset",y="% of Portfolio VaR",color="Asset",title="Component VaR") if cvar_df is not None else go.Figure().update_layout(title="Component VaR (insufficient data)"))
    fig_cvar.update_layout(template="plotly_white",height=320,showlegend=False)
    kp=[]
    if not (phat is None or (isinstance(phat,float) and np.isnan(phat))): kp.append(f"Breach rate: {phat*100:.2f}% (target {(1-alpha)*100:.0f}%)")
    if pval is not None and not np.isnan(pval): kp.append(f"p-value: {pval:.3f} → {'adequate ✓' if pval>0.05 else 'under-estimates ✗'}")
    if not kp: kp=["Need > 255 days for Kupiec backtest"]
    return html.Div([
        dbc.Row([dbc.Col(section_card(f"VaR comparison @ {int(alpha*100)}%",[dash_table.DataTable(columns=[{"name":c,"id":c} for c in rows[0].keys()],data=rows,style_cell={"fontFamily":"system-ui","padding":"6px","fontSize":"0.82rem"},style_header={"backgroundColor":"#f4f6fa","fontWeight":600})],icon="📊"),md=6),
                 dbc.Col(section_card("Distribution",[dcc.Graph(figure=fig_h)],icon="📉"),md=6)]),
        section_card("Kupiec POF backtest",[html.Ul([html.Li(x) for x in kp])],icon="🧪"),
        section_card(f"Per-asset @ α={int(alpha*100)}%",[dash_table.DataTable(columns=[{"name":c,"id":c} for c in per[0].keys()] if per else [],data=per,style_cell={"fontFamily":"system-ui","padding":"6px","fontSize":"0.82rem"},style_header={"backgroundColor":"#f4f6fa","fontWeight":600})],icon="🪙"),
        dbc.Row([dbc.Col(section_card("Component VaR",[dcc.Graph(figure=fig_cvar)],icon="🧮"),md=6),dbc.Col(section_card("Rolling VaR & ES",[dcc.Graph(figure=fig_roll)],icon="⏱"),md=6)]),
    ])

def render_liq(holdings,alpha,lam):
    df=FEATURES.copy(); latest=df.sort_values("date").groupby("asset").tail(1).set_index("asset")
    AUM=sum(holdings.values()) or 1.0; rows=[]; total_lc=0.0
    for a in ASSETS:
        if a not in latest.index: continue
        h=float(holdings.get(a,0)); adv=float(latest.loc[a,"adv_30_usd"]); sd=df[df["asset"]==a]["return"].std(ddof=1)
        var_a=hist_var(df[df["asset"]==a]["return"],alpha); lc=liquidity_cost(h,adv,sd,lam); total_lc+=lc
        lavar=(var_a*h+lc) if not np.isnan(var_a) else np.nan
        liq_days=(h/(adv*LIQ_PARTICIPATION)) if adv>0 else float("inf")
        ld_str="∞" if (np.isnan(liq_days) or not np.isfinite(liq_days)) else (f"{liq_days*24:.1f}h" if liq_days<1 else f"{liq_days:.3f}d")
        rows.append({"Asset":a,"Position":money(h),"30-day ADV":money(adv),"Position/ADV":f"{(h/adv*100 if adv>0 else 0):.4f}%","Liq. days":ld_str,"σ (daily)":f"{sd*100:.2f}%","Liq. cost":money(lc),"Std VaR ($)":money(var_a*h) if not np.isnan(var_a) else "—","L-VaR ($)":money(lavar) if not np.isnan(lavar) else "—"})
    port_ret=portfolio_returns_from(df,holdings).dropna(); std_var_pct=hist_var(port_ret,alpha) if not port_ret.empty else np.nan; std_var_usd=std_var_pct*AUM if not np.isnan(std_var_pct) else np.nan
    lam_grid=np.linspace(0.0,2.0,21); sens=[]
    for L in lam_grid:
        tot=sum(liquidity_cost(float(holdings.get(a,0)),float(latest.loc[a,"adv_30_usd"]) if a in latest.index else 0,df[df["asset"]==a]["return"].std(ddof=1),L) for a in ASSETS if a in latest.index)
        sens.append({"lambda":L,"Liquidity cost":tot,"L-VaR":(std_var_usd or 0)+tot})
    sens_df=pd.DataFrame(sens)
    fig_lam=go.Figure()
    fig_lam.add_trace(go.Scatter(x=sens_df["lambda"],y=sens_df["Liquidity cost"],name="Liquidity cost",mode="lines+markers"))
    fig_lam.add_trace(go.Scatter(x=sens_df["lambda"],y=sens_df["L-VaR"],name="L-VaR",mode="lines+markers"))
    if std_var_usd and not np.isnan(std_var_usd): fig_lam.add_hline(y=std_var_usd,line_dash="dash",line_color="gray",annotation_text=f"Std VaR={money(std_var_usd)}")
    fig_lam.add_vline(x=lam,line_color="black",line_dash="dot",annotation_text=f"λ={lam:.2f}")
    fig_lam.update_layout(title="L-VaR vs λ (BDSS)",template="plotly_white",height=380,xaxis_title="λ",yaxis_title="USD")
    return html.Div([
        info_alert([("BDSS","LC=λ·(Position/ADV)·σ·Position; L-VaR=Std VaR+LC"),("ADV note","Yahoo Finance crypto Volume already in USD")]),
        section_card(f"Per-asset (λ={lam:.2f}, α={int(alpha*100)}%)",[dash_table.DataTable(columns=[{"name":c,"id":c} for c in rows[0].keys()] if rows else [],data=rows,style_cell={"fontFamily":"system-ui","padding":"6px","fontSize":"0.78rem"},style_header={"backgroundColor":"#f4f6fa","fontWeight":600}),html.Div([html.B("Total LC: "),money(total_lc),"  ",html.B("Std VaR: "),money(std_var_usd),"  ",html.B("L-VaR: "),money((std_var_usd or 0)+total_lc)],style={"marginTop":"10px","color":BRAND,"fontSize":"0.9rem"})],icon="💧"),
        section_card("λ sensitivity",[dcc.Graph(figure=fig_lam)],icon="🎚"),
    ])

def render_scen(holdings,alpha,lam,scenario):
    sc=SCENARIOS[scenario]; df=FEATURES.copy(); latest=df.sort_values("date").groupby("asset").tail(1).set_index("asset")
    AUM=sum(holdings.values()) or 1.0; rows=[]; total_mtm=0.0; total_stress_lc=0.0
    for a in ASSETS:
        if a not in latest.index: continue
        h=float(holdings.get(a,0)); shock=sc["shocks"].get(a,-0.30); adv=float(latest.loc[a,"adv_30_usd"]); sd=df[df["asset"]==a]["return"].std(ddof=1)
        mtm=h*shock; slc=liquidity_cost(h,adv*sc["adv_mult"],sd*sc["vol_mult"],lam*sc["lambda_mult"])
        total_mtm+=mtm; total_stress_lc+=slc
        rows.append({"Asset":a,"Position":money(h),"Shock":f"{shock*100:+.0f}%","MTM P&L":money(mtm),"Stressed ADV":money(adv*sc["adv_mult"]),"Stressed σ":f"{sd*sc['vol_mult']*100:.2f}%","Stressed λ":f"{lam*sc['lambda_mult']:.2f}","Stressed LC":money(slc),"Total loss":money(mtm-slc)})
    total_loss=abs(total_mtm)+total_stress_lc; cumulative=total_mtm-total_stress_lc
    fig_w=go.Figure(go.Waterfall(orientation="v",measure=["relative","relative","total"],x=["MTM P&L","Stressed LC","Total stressed loss"],text=[money(total_mtm),money(-total_stress_lc),money(cumulative)],y=[total_mtm,-total_stress_lc,cumulative],connector={"line":{"color":"#888"}}))
    fig_w.update_layout(title=f"{scenario} — decomposition",template="plotly_white",height=380,yaxis_title="USD")
    port_ret=portfolio_returns_from(df,holdings).dropna(); base_var_usd=hist_var(port_ret,alpha)*AUM if not port_ret.empty else 0
    base_lc=sum(liquidity_cost(holdings.get(a,0),float(latest.loc[a,"adv_30_usd"]) if a in latest.index else 0,df[df["asset"]==a]["return"].std(ddof=1),lam) for a in ASSETS if a in latest.index)
    fig_cmp=go.Figure([go.Bar(name="Std VaR/shock",x=["Base","Stressed"],y=[base_var_usd,abs(total_mtm)],marker_color="#c0392b"),go.Bar(name="Liq. cost",x=["Base","Stressed"],y=[base_lc,total_stress_lc],marker_color="#e67e22")])
    fig_cmp.update_layout(barmode="stack",title="Base vs stressed",template="plotly_white",height=320,yaxis_title="USD")
    return html.Div([
        dbc.Alert([html.H5(scenario,style={"color":BRAND}),html.Div(sc["description"])],color="warning"),
        dbc.Row([dbc.Col(kpi_card("MTM P&L",money(total_mtm),"Shock leg","#c0392b")),dbc.Col(kpi_card("Stressed LC",money(total_stress_lc),"Fire-sale","#e67e22")),dbc.Col(kpi_card("Total loss",money(-total_loss),f"{total_loss/AUM*100:.1f}% AUM","#8e44ad"))],className="g-2 mb-3"),
        section_card("Waterfall",[dcc.Graph(figure=fig_w)],icon="🌊"),
        section_card("Per-asset",[dash_table.DataTable(columns=[{"name":c,"id":c} for c in rows[0].keys()] if rows else [],data=rows,style_cell={"fontFamily":"system-ui","padding":"6px","fontSize":"0.78rem"},style_header={"backgroundColor":"#f4f6fa","fontWeight":600})],icon="📋"),
        section_card("Base vs stressed",[dcc.Graph(figure=fig_cmp)],icon="⚖"),
    ])

def render_methodology():
    return html.Div([
        section_card("Methodology",[
            html.P("Three VaR estimators (Historical, Parametric, Cornish-Fisher), ES, Bootstrap CI (n=1000), Kupiec POF backtest, Component VaR (Euler), BDSS L-VaR, 4 stress scenarios."),
            html.P([html.B("Cornish-Fisher: "),"z*=z+(z²−1)s/6+(z³−3z)k/24−(2z³−5z)s²/36"]),
            html.P([html.B("BDSS: "),"LC=λ·(Position/ADV)·σ·Position; L-VaR=Std VaR+LC"]),
        ],icon="🔬"),
        section_card("References",[html.Ol([
            html.Li("Bangia et al. (1999). Modeling liquidity risk."),html.Li("Almgren & Chriss (2000). Optimal execution."),
            html.Li("Kupiec (1995). Verifying VaR models."),html.Li("Artzner et al. (1999). Coherent measures of risk."),
            html.Li("Cornish & Fisher (1937). Moments and cumulants."),html.Li("Kyle (1985). Continuous auctions."),
            html.Li("Brunnermeier & Pedersen (2009). Market liquidity."),html.Li("BCBS (2019). Market risk capital (d457)."),
        ])],icon="📚"),
    ])

@app.callback(
    Output("kpi-row","children"),Output("sidebar-summary","children"),Output("tab-content","children"),
    Input("tabs","value"),Input("alpha-slider","value"),Input("lambda-slider","value"),Input("scenario-dd","value"),
    Input({"type":"hold","index":ALL},"value"),State({"type":"hold","index":ALL},"id"),
    prevent_initial_call=True,
)
def update_all(tab,alpha,lam,scenario,vals,ids):
    try:
        holdings={ids[i]["index"]:float(vals[i] or 0) for i in range(len(ids))}; AUM=sum(holdings.values())
        if AUM<=0: return dbc.Alert("⚠ Set holdings > 0.",color="warning"),[html.Div("AUM: $0")],html.Div("Set positive holdings.")
        port_ret=portfolio_returns_from(FEATURES,holdings).dropna()
        std_var_pct=hist_var(port_ret,alpha) if not port_ret.empty else np.nan; std_var_usd=std_var_pct*AUM if not np.isnan(std_var_pct) else np.nan
        latest=FEATURES.sort_values("date").groupby("asset").tail(1).set_index("asset")
        lc_total=sum(liquidity_cost(holdings.get(a,0),float(latest.loc[a,"adv_30_usd"]) if a in latest.index else 0,FEATURES[FEATURES["asset"]==a]["return"].std(ddof=1),lam) for a in ASSETS if a in latest.index)
        lvar=(std_var_usd+lc_total) if not np.isnan(std_var_usd) else np.nan; a_vol=ann_vol(port_ret) if not port_ret.empty else np.nan
        kpi=dbc.Row([
            dbc.Col(kpi_card("Portfolio AUM",money(AUM),f"{len(ASSETS)} assets")),
            dbc.Col(kpi_card(f"Std VaR {int(alpha*100)}%",money(std_var_usd),pctf(std_var_pct),"#c0392b")),
            dbc.Col(kpi_card(f"Liq. cost (λ={lam:.2f})",money(lc_total),f"{lc_total/AUM*100:.4f}% AUM","#e67e22")),
            dbc.Col(kpi_card(f"L-VaR {int(alpha*100)}%",money(lvar),pctf(lvar/AUM) if not np.isnan(lvar) else "—","#8e44ad")),
            dbc.Col(kpi_card("Ann. volatility",pctf(a_vol,1),"√365 basis")),
        ],className="g-2")
        side=[html.Div([html.B("AUM: "),money(AUM)]),html.Div([html.B("α: "),f"{int(alpha*100)}%"]),html.Div([html.B("λ: "),f"{lam:.2f}"]),html.Div([html.B("Scenario: "),scenario])]
        if tab=="tab-market": content=render_market(holdings,alpha)
        elif tab=="tab-var": content=render_var(holdings,alpha)
        elif tab=="tab-liq": content=render_liq(holdings,alpha,lam)
        elif tab=="tab-scen": content=render_scen(holdings,alpha,lam,scenario)
        elif tab=="tab-meth": content=render_methodology()
        else: content=html.Div("Unknown tab")
        return kpi,side,content
    except Exception:
        return dbc.Alert(html.Pre(traceback.format_exc(),style={"whiteSpace":"pre-wrap"}),color="danger"),"",html.Div()

if __name__=="__main__":
    app.run_server(debug=False,host="0.0.0.0",port=int(os.environ.get("PORT",8050)))
