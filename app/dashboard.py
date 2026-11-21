# app/dashboard.py
"""Dashboard for the US accidents severity model.   Run from the project root:
    python app/dashboard.py   then open http://localhost:8080
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nicegui import app as nicegui_app
from nicegui import ui

from src.models.predictor import Predictor

CLASS_NAMES = ["Severity 1", "Severity 2", "Severity 3", "Severity 4"]
MEANING = ["Shortest traffic delay", "Short traffic delay", "Longer traffic delay", "Longest traffic delay"]
COLORS = ["#7BC043", "#F2B705", "#F07B2C", "#D7263D"]
QCOLORS = ["sev1", "sev2", "sev3", "sev4"]
PAPER, INK, FOREST, TERRA, OCHRE, LINE, MUTED = "#F4F7FB", "#0B1530", "#0B1530", "#4A5B8C", "#C5F82A", "#DCE3EF", "#5B6784"
TEAM = [("Syed Usman Habib", "CT-24128"), ("Samiullah Saeed", "CT-24134"), ("Adnan Khaliq", "CT-24143")]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December"]
CITIES = {"Los Angeles": (34.05, -118.24), "New York": (40.71, -74.01), "Houston": (29.76, -95.37),
          "Chicago": (41.88, -87.63), "Miami": (25.76, -80.19), "Seattle": (47.61, -122.33),
          "Atlanta": (33.75, -84.39), "Denver": (39.74, -104.99)}

print("Loading model and category levels (a few seconds)...")
P = Predictor()
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
nicegui_app.add_static_files("/figures", str(FIG_DIR))


def find_weather(word, fallback):
    if word in P.weather_options:
        return word
    return next((w for w in P.weather_options if word.lower() in w.lower()), fallback)


DEFAULT_WEATHER = find_weather("Clear", P.weather_options[0])
state = {"temperature": 65, "humidity": 55, "visibility": 10, "wind": 7, "precipitation": None,
         "weather": DEFAULT_WEATHER, "sunrise_sunset": "Day" if "Day" in P.sun_options else P.sun_options[0],
         "hour": 17, "day_of_week": 4, "month": 11, "junction": False, "traffic_signal": False,
         "crossing": False, "lat": 34.05, "lng": -118.24}
PRESETS = {
    "Clear afternoon": dict(temperature=72, humidity=40, visibility=10, wind=6, precipitation=None,
                            weather=DEFAULT_WEATHER, sunrise_sunset="Day", hour=15, junction=False, traffic_signal=False),
    "Rainy rush hour": dict(temperature=52, humidity=90, visibility=3, wind=14, precipitation=0.2,
                            weather=find_weather("Rain", DEFAULT_WEATHER), sunrise_sunset="Day", hour=17,
                            junction=True, traffic_signal=True),
    "Foggy night": dict(temperature=40, humidity=95, visibility=0.5, wind=3, precipitation=None,
                        weather=find_weather("Fog", DEFAULT_WEATHER), sunrise_sunset="Night", hour=3,
                        junction=False, traffic_signal=False),
    "Snowy morning": dict(temperature=24, humidity=85, visibility=2, wind=12, precipitation=0.1,
                          weather=find_weather("Snow", DEFAULT_WEATHER), sunrise_sunset="Day", hour=8,
                          junction=True, traffic_signal=False),
}

controls, bars, prob_labels = {}, [], []
ui_out = {}

ui.add_head_html("""
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=Instrument+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{--paper:#F4F7FB;--ink:#0B1530;--forest:#0B1530;--terra:#4A5B8C;--ochre:#C5F82A;--lime:#C5F82A;--line:#DCE3EF;--muted:#5B6784}
html,body,.q-layout,.q-page-container,.q-page{background:var(--paper) !important;color:var(--ink);font-family:'Instrument Sans',sans-serif !important}
.serif{font-family:'Bricolage Grotesque',sans-serif !important}
.mono{font-family:'JetBrains Mono',monospace !important}
.ink{color:var(--ink)} .muted{color:var(--muted)}
.panel{background:#FFFFFF;border:1px solid var(--line);border-radius:10px;box-shadow:none}
.sect{font-family:'JetBrains Mono',monospace;font-size:11.5px;letter-spacing:.09em;text-transform:uppercase;color:var(--terra);font-weight:500}
.h2{font-family:'Bricolage Grotesque',sans-serif;font-weight:600;font-size:22px;color:var(--forest);line-height:1.2}
.kpi-n{font-family:'Bricolage Grotesque',sans-serif;font-size:34px;font-weight:700;color:var(--forest);line-height:1.1}
.kpi-n.hot{background:#C5F82A;display:inline-block;padding:0 8px;border-radius:6px}
.kpi-l{font-size:12.5px;color:var(--muted);line-height:1.3}
.q-tab-panel{padding:0} .q-tab-panels{background:transparent}
.q-tab{font-weight:600;text-transform:none;letter-spacing:0;color:var(--muted)} .q-tab--active{color:var(--forest)}.q-tab__indicator{height:4px !important}
.q-field--outlined .q-field__control{background:#fff;border-radius:6px}
.q-table{background:transparent;color:var(--ink)} .q-table th{font-family:'JetBrains Mono',monospace;font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}
.q-table td{font-family:'JetBrains Mono',monospace;font-size:13px}
.q-table td:first-child{font-family:'Instrument Sans',sans-serif;font-weight:600;color:var(--forest)}
.q-expansion-item__container,.q-item{color:var(--ink)}
.text-sev1{color:#7BC043 !important}.text-sev2{color:#F2B705 !important}.text-sev3{color:#F07B2C !important}.text-sev4{color:#D7263D !important}
.text-forest{color:var(--forest) !important}.text-terra{color:var(--terra) !important}.text-lime{color:#C5F82A !important}.bg-forest{background:#0B1530 !important}.q-btn-group .q-btn{border:1px solid #DCE3EF}
.bg-track{background:#E3E9F4 !important}
.paper-btn{border:1px solid var(--forest);color:var(--forest);background:transparent;border-radius:999px;font-weight:600;text-transform:none;letter-spacing:0}
.paper-btn:hover{background:var(--forest);color:#C5F82A}
.note-box{background:#EAF9B8;border-radius:8px;padding:10px 14px;font-size:13px;color:var(--ink)}
.q-header{border-bottom:4px solid #C5F82A}
a{color:var(--terra)}
.seg{height:8px;border-radius:4px;flex:1;background:#DCE3EF;transition:background .4s,box-shadow .4s}
.q-linear-progress__model{background:currentColor !important}
</style>""")
ui.colors(primary="#0B1530", positive="#0B1530", negative="#4A5B8C")


def update(*_):
    if not ui_out:
        return
    r = P.predict(state)
    d = r["decision"]
    ui_out["badge"].set_text(str(d + 1))
    for i, seg in enumerate(ui_out["segs"]):
        seg.style("background:#C5F82A;box-shadow:0 0 0 1px #0B1530 inset" if i <= d else "background:#DCE3EF;box-shadow:none")
    ui_out["name"].set_text(CLASS_NAMES[d])
    ui_out["meaning"].set_text(MEANING[d])
    ui_out["share"].set_text(f"Decision weight {r['share'] * 100:.0f}% after tuning")
    for i, p in enumerate(r["proba"]):
        bars[i].set_value(p)
        prob_labels[i].set_text(f"{p * 100:.1f}%")
        prob_labels[i].style(f"font-weight:700;color:{INK}" if i == d else f"color:{MUTED}")


def slider_row(label, key, mn, mx, step=1, unit=""):
    with ui.column().classes("w-full gap-0"):
        with ui.row().classes("w-full justify-between items-center no-wrap"):
            ui.label(label).classes("text-sm ink")
            val = ui.label(f"{state[key]:g}{unit}").classes("text-sm font-medium ink")
        s = ui.slider(min=mn, max=mx, step=step, value=state[key]).props("dense color=forest")

        def on(e, k=key, v=val):
            state[k] = e.value
            v.set_text(f"{e.value:g}{unit}")
            update()
        s.on_value_change(on)
    controls[key] = s


def simple(el, key):
    def on(e, k=key):
        state[k] = e.value
        update()
    el.on_value_change(on)
    controls[key] = el
    return el


def apply_preset(name):
    for k, v in PRESETS[name].items():
        controls[k].set_value(v)


def pick_city(e):
    if e.value in CITIES:
        controls["lat"].set_value(CITIES[e.value][0])
        controls["lng"].set_value(CITIES[e.value][1])


def kpi(label, value, sub="", good=False):
    with ui.column().classes("panel p-4 gap-1 flex-1").style("min-width:150px"):
        ui.label(label).classes("kpi-l")
        ui.label(value).classes("kpi-n" + (" hot" if good else ""))
        if sub:
            ui.label(sub).classes("kpi-l")


def confusion_html(cm):
    head = "".join(f'<th style="padding:6px 10px;color:#5B6784;font-weight:500;font-family:JetBrains Mono,monospace;font-size:11px">Pred S{j + 1}</th>' for j in range(4))
    rows = ""
    for i in range(4):
        tot = max(sum(cm[i]), 1)
        cells = ""
        for j in range(4):
            share = cm[i][j] / tot
            cells += (f'<td style="padding:8px 10px;text-align:center;border-radius:8px;'
                      f'font-family:JetBrains Mono,monospace;background:rgba(11,21,48,{0.06 + 0.86 * share:.2f});color:{"#fff" if share > .5 else "#0B1530"}">{cm[i][j]:,}'
                      f'<div style="font-size:11px;opacity:.7">{share * 100:.0f}% of true</div></td>')
        rows += (f'<tr><th style="text-align:left;padding:6px 10px;color:#5B6784;font-weight:500;font-family:JetBrains Mono,monospace;font-size:11px">'
                 f'True S{i + 1}</th>{cells}</tr>')
    return (f'<table style="border-collapse:separate;border-spacing:4px;width:100%;color:#0B1530;font-size:13px">'
            f'<tr><th></th>{head}</tr>{rows}</table>')


def report_table(rep):
    rows = [{"cls": n, "precision": f"{rep[n]['precision']:.2f}", "recall": f"{rep[n]['recall']:.2f}",
             "f1": f"{rep[n]['f1-score']:.2f}", "support": f"{int(rep[n]['support']):,}"} for n in CLASS_NAMES]
    cols = [{"name": c, "label": "Class" if c == "cls" else c.capitalize(), "field": c, "align": "left"}
            for c in ("cls", "precision", "recall", "f1", "support")]
    ui.table(columns=cols, rows=rows, row_key="cls").props("dense flat").classes("w-full")


res = P.meta.get("results", {})
tuned_res = res.get("test_tuned", {})
plain_res = res.get("test", {})

with ui.header().classes("items-center justify-between").style(f"background:{FOREST}"):
    with ui.column().classes("gap-0"):
        ui.label("AI & EXPERT SYSTEMS  ·  COMPLEX COMPUTING PROBLEM").classes("mono").style("font-size:10.5px;letter-spacing:.1em;color:#C5F82A")
        ui.label("Predicting how badly a US traffic accident will disrupt the road").classes("serif") \
            .style("font-size:22px;font-weight:700;color:#F4F7FB;line-height:1.15")
    if tuned_res:
        with ui.row().classes("items-center gap-5 no-wrap"):
            for lab, val in (("test accuracy", f"{tuned_res['accuracy'] * 100:.1f}%"), ("macro-F1", f"{tuned_res['macro_f1']:.2f}")):
                with ui.column().classes("gap-0 items-end"):
                    ui.label(val).classes("serif").style("font-size:22px;font-weight:700;color:#F4F7FB;line-height:1")
                    ui.label(lab).classes("mono").style("font-size:10px;letter-spacing:.08em;text-transform:uppercase;color:#AEB9D6")

with ui.column().classes("w-full max-w-6xl mx-auto p-4 gap-4"):
    with ui.tabs().props("dense indicator-color=lime align=left") as tabs:
        t_predict = ui.tab("Predict")
        t_perf = ui.tab("Model performance")
        t_about = ui.tab("About")

    with ui.tab_panels(tabs, value=t_predict).classes("w-full"):
        with ui.tab_panel(t_predict):
            with ui.row().classes("w-full items-center gap-2 mb-3"):
                ui.label("Try a scenario").classes("sect")
                for name in PRESETS:
                    ui.button(name, on_click=lambda n=name: apply_preset(n)).props("flat dense no-caps").classes("px-4 paper-btn")

            with ui.row().classes("w-full gap-5 items-start"):
                with ui.column().classes("panel p-5 gap-3 flex-1").style("min-width:320px"):
                    ui.label("01 · Weather").classes("sect")
                    slider_row("Temperature", "temperature", -20, 110, 1, " F")
                    slider_row("Humidity", "humidity", 0, 100, 1, " %")
                    slider_row("Visibility", "visibility", 0, 10, 0.5, " mi")
                    slider_row("Wind speed", "wind", 0, 60, 1, " mph")
                    with ui.row().classes("w-full gap-3 no-wrap"):
                        simple(ui.select(P.weather_options, label="Condition", value=state["weather"], with_input=True)
                               .classes("flex-1").props("dense outlined"), "weather")
                        simple(ui.number("Precipitation (in)", value=None, min=0, max=5, step=0.05, format="%.2f")
                               .classes("flex-1").props("dense outlined clearable"), "precipitation")
                    ui.label("02 · Time").classes("sect mt-2")
                    slider_row("Hour of day", "hour", 0, 23, 1, ":00")
                    with ui.row().classes("w-full gap-3 no-wrap"):
                        simple(ui.select({i: n for i, n in enumerate(DAYS)}, label="Day", value=state["day_of_week"])
                               .classes("flex-1").props("dense outlined"), "day_of_week")
                        simple(ui.select({i + 1: n for i, n in enumerate(MONTHS)}, label="Month", value=state["month"])
                               .classes("flex-1").props("dense outlined"), "month")
                    simple(ui.toggle(P.sun_options, value=state["sunrise_sunset"]).props("dense no-caps no-caps color=white text-color=forest toggle-color=forest toggle-text-color=lime"),
                           "sunrise_sunset")
                    ui.label("03 · Road").classes("sect mt-2")
                    with ui.row().classes("gap-4"):
                        simple(ui.switch("Junction"), "junction")
                        simple(ui.switch("Traffic signal"), "traffic_signal")
                        simple(ui.switch("Crossing"), "crossing")
                    if P.uses_geo:
                        ui.label("04 · Location").classes("sect mt-2")
                        ui.select(list(CITIES), label="City preset", value="Los Angeles", on_change=pick_city) \
                            .classes("w-full").props("dense outlined")
                        with ui.row().classes("w-full gap-3 no-wrap"):
                            simple(ui.number("Latitude", value=state["lat"], format="%.3f").classes("flex-1")
                                   .props("dense outlined"), "lat")
                            simple(ui.number("Longitude", value=state["lng"], format="%.3f").classes("flex-1")
                                   .props("dense outlined"), "lng")

                with ui.column().classes("panel p-5 gap-3").style("width:min(100%,26rem)"):
                    ui.label("Predicted severity").classes("sect")
                    with ui.row().classes("items-center gap-4 no-wrap"):
                        ui_out["badge"] = ui.label("-").classes("serif text-center") \
                            .style("width:88px;height:88px;line-height:88px;border-radius:10px;font-size:54px;font-weight:800;background:#C5F82A;color:#0B1530;box-shadow:0 0 0 6px rgba(197,248,42,.28)")
                        with ui.column().classes("gap-0"):
                            ui_out["name"] = ui.label("").classes("serif").style("font-size:26px;font-weight:700;color:#0B1530")
                            ui_out["meaning"] = ui.label("").classes("text-sm ink")
                            ui_out["share"] = ui.label("").classes("text-xs muted")
                    with ui.row().classes("w-full items-center no-wrap gap-2"):
                        ui.label("Severity scale").classes("mono").style("font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;color:#5B6784;width:112px;white-space:nowrap")
                        ui_out["segs"] = [ui.element("div").classes("seg") for _ in range(4)]
                    ui.separator().style("background:#DCE3EF")
                    ui.label("What the model believes").classes("sect")
                    for i, name in enumerate(CLASS_NAMES):
                        with ui.row().classes("w-full items-center no-wrap gap-3"):
                            ui.label(name).classes("text-sm ink").style("width:84px")
                            bars.append(ui.linear_progress(value=0, show_value=False, size="10px",
                                                           color="white").classes("flex-1").props("rounded track-color=track").style(f"color:{COLORS[i]} !important"))
                            prob_labels.append(ui.label("0%").classes("text-sm text-right mono").style("width:52px"))
                    ui.label("The big badge applies class weights tuned on validation data so rare, severe crashes are "
                             "not ignored. It can differ from the longest bar. Severity measures traffic delay, "
                             "not injuries.").classes("text-xs muted mt-1")

        with ui.tab_panel(t_perf):
            perf = P.performance()
            base = P.meta.get("majority_baseline_test_accuracy")
            with ui.column().classes("w-full gap-4"):
                ui.label(f"Held-out test set of {perf['n_test']:,} crashes the model never saw during training or tuning."
                         ).classes("text-sm muted")
                with ui.row().classes("w-full gap-4"):
                    acc_ok = tuned_res.get("accuracy", 0) >= 0.70
                    kpi("Accuracy (tuned)", f"{tuned_res.get('accuracy', 0) * 100:.1f}%",
                        "meets the 70% target" if acc_ok else "below the 70% target", good=acc_ok)
                    kpi("Macro-F1 (tuned)", f"{tuned_res.get('macro_f1', 0):.2f}", "average over all 4 classes")
                    kpi("Macro-F1 (plain)", f"{plain_res.get('macro_f1', 0):.2f}", "before tuning the weights")
                    kpi("Always guess Severity 2", f"{(base or 0) * 100:.1f}%", "accuracy of a useless model")
                with ui.column().classes("panel p-5 gap-2 w-full"):
                    ui.label("Per-class recall and F1: tuned vs plain").classes("sect")
                    rep_t = perf["tuned"]["report"]
                    rep_p = perf["plain"]["report"]
                    ui.echart({
                        "textStyle": {"color": "#0B1530", "fontFamily": "Instrument Sans, sans-serif"},
                        "tooltip": {"trigger": "axis"},
                        "legend": {"textStyle": {"color": "#0B1530", "fontFamily": "Instrument Sans, sans-serif"}, "top": 0},
                        "grid": {"left": 40, "right": 10, "bottom": 30, "top": 40},
                        "xAxis": {"type": "category", "data": CLASS_NAMES, "axisLabel": {"color": "#5B6784", "fontFamily": "JetBrains Mono, monospace", "fontSize": 11}},
                        "yAxis": {"type": "value", "max": 1, "axisLabel": {"color": "#5B6784", "fontFamily": "JetBrains Mono, monospace", "fontSize": 11},
                                  "splitLine": {"lineStyle": {"color": "#DCE3EF"}}},
                        "series": [
                            {"type": "bar", "name": "Recall (tuned)", "itemStyle": {"color": "#C5F82A", "borderColor": "#0B1530", "borderWidth": 1, "borderRadius": [3, 3, 0, 0]},
                             "data": [round(rep_t[n]["recall"], 3) for n in CLASS_NAMES]},
                            {"type": "bar", "name": "F1 (tuned)", "itemStyle": {"color": "#0B1530", "borderRadius": [3, 3, 0, 0]},
                             "data": [round(rep_t[n]["f1-score"], 3) for n in CLASS_NAMES]},
                            {"type": "bar", "name": "F1 (plain)", "itemStyle": {"color": "#9AA5BD", "borderRadius": [2, 2, 0, 0]},
                             "data": [round(rep_p[n]["f1-score"], 3) for n in CLASS_NAMES]},
                        ],
                    }).classes("w-full").style("height:300px")
                with ui.row().classes("w-full gap-4 items-start"):
                    with ui.column().classes("panel p-5 gap-2 flex-1").style("min-width:320px"):
                        ui.label("Metrics by class (tuned)").classes("sect")
                        report_table(rep_t)
                    with ui.column().classes("panel p-5 gap-2 flex-1").style("min-width:320px"):
                        ui.label("Confusion matrix (tuned)").classes("sect")
                        ui.html(confusion_html(perf["tuned"]["confusion"])).classes("w-full")
                with ui.expansion("Before tuning: plain argmax with balanced class weights", icon="compare_arrows") \
                        .classes("panel w-full ink"):
                    with ui.column().classes("w-full gap-3 p-3"):
                        report_table(rep_p)
                        ui.html(confusion_html(perf["plain"]["confusion"])).classes("w-full")
                shap_png = FIG_DIR / "shap_feature_importance.png"
                with ui.column().classes("panel p-5 gap-2 w-full"):
                    ui.label("What drives the predictions (SHAP)").classes("sect")
                    shap_json = FIG_DIR / "shap_importance.json"
                    if shap_json.exists():
                        import json as _json
                        sh = _json.loads(shap_json.read_text())
                        sev_cols = ["#7BC043", "#F2B705", "#F07B2C", "#D7263D"]
                        ui.echart({
                            "textStyle": {"fontFamily": "Instrument Sans, sans-serif", "color": "#0B1530"},
                            "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
                            "legend": {"top": 0, "textStyle": {"color": "#5B6784"}},
                            "grid": {"left": 130, "right": 20, "bottom": 30, "top": 36},
                            "xAxis": {"type": "value", "axisLabel": {"color": "#5B6784", "fontFamily": "JetBrains Mono, monospace", "fontSize": 11},
                                      "splitLine": {"lineStyle": {"color": "#DCE3EF"}}},
                            "yAxis": {"type": "category", "data": [f.replace("_", " ") for f in sh["features"]],
                                      "axisLabel": {"color": "#0B1530"}},
                            "series": [{"type": "bar", "stack": "s", "name": n, "itemStyle": {"color": sev_cols[i]},
                                        "data": [round(v, 4) for v in sh["values"][i]]} for i, n in enumerate(sh["classes"])],
                        }).classes("w-full").style("height:480px")
                    elif shap_png.exists():
                        ui.image("/figures/shap_feature_importance.png").classes("w-full").style("max-width:760px")
                    else:
                        ui.label("Run `python src/models/evaluate.py` to add the SHAP chart here.") \
                            .classes("text-sm muted")

        with ui.tab_panel(t_about):
            meta = P.meta
            notes = [
                ("Data", f"US Accidents (Kaggle), about 7.7 million raw records. This model trains on a random "
                         f"{meta.get('rows', 0):,}-row sample with a {meta.get('split', 'random')} 70/15/15 train, validation and test split."),
                ("Model", f"XGBoost multi-class classifier, tuned with Optuna ({meta.get('trials', '?')} trials) on validation "
                          "macro-F1. Class-balanced sample weights are used in training."),
                ("Tuned weights", "After training, per-class probability multipliers are searched on the validation set: "
                                  "maximise macro-F1 while keeping accuracy at 70% or higher. The test set is only used once at the end."),
                ("Reading the numbers", "Severity 2 is about 80% of all crashes, so accuracy alone is misleading. "
                                        "Macro-F1 and per-class recall show whether rare severe crashes are being caught."),
                ("Limitations", "Missing precipitation values are common in older records, so the model can use "
                                "'precipitation missing' as a proxy for the year or data source. Location features can also reflect "
                                "how each region reports crashes, not just what causes them."),
            ]
            with ui.column().classes("w-full gap-3"):
                for title, text in notes:
                    with ui.column().classes("panel p-5 gap-1 w-full"):
                        ui.label(title).classes("sect")
                        ui.label(text).classes("text-sm ink")
                with ui.row().classes("w-full gap-4 items-stretch"):
                    with ui.column().classes("panel p-5 gap-2 flex-1").style("min-width:300px"):
                        ui.label("Project team").classes("sect")
                        for n, r in TEAM:
                            with ui.row().classes("w-full justify-between items-center no-wrap"):
                                ui.label(n).classes("serif").style("font-size:17px;font-weight:600;color:#0B1530")
                                ui.label(r).classes("mono ink text-sm")
                    with ui.column().classes("panel p-5 gap-1 flex-1").style("min-width:300px"):
                        ui.label("Course requirements").classes("sect")
                        for t in ("Raw Kaggle data, 1,000,000 rows sampled", "Not clean: year-dependent gaps, mixed timestamps",
                                  "XGBoost + Optuna + decision-weight tuning", f"Accuracy above 70%: {tuned_res.get('accuracy', 0) * 100:.1f}%",
                                  "Follows 2025-26 published work"):
                            with ui.row().classes("items-center gap-2 no-wrap"):
                                ui.label("✓").classes("mono").style("color:#0B1530;font-weight:700")
                                ui.label(t).classes("text-sm ink")

update()

if __name__ in {"__main__", "__mp_main__"}:
    ui.run(title="US accident severity predictor", port=8080, reload=False, show=False, dark=False)
