from __future__ import annotations

import html
import json
from typing import Any


def render_cinematic_dashboard_html(payload: dict[str, Any]) -> str:
    """Render a standalone theme-switchable cinematic dashboard."""
    title = html.escape(str(payload.get("title", "CCS Operations")))
    subtitle = html.escape(str(payload.get("subtitle", "")))
    data_json = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).replace("</", "<\\/")

    template = """<!doctype html>
<html lang="en" data-theme="light-v2">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light dark">
  <title>__TITLE__</title>
  <script>
    try {
      const savedTheme = localStorage.getItem("ccs-rl-dashboard-theme");
      if (savedTheme === "light-v2" || savedTheme === "dark") {
        document.documentElement.dataset.theme = savedTheme;
      }
    } catch (_error) {
      // Local storage can be unavailable in privacy-restricted file views.
    }
  </script>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <style>
    :root {
      --night: #06131d;
      --night-raised: #0a1c28;
      --glass: rgba(8, 26, 38, .84);
      --glass-soft: rgba(9, 29, 42, .68);
      --ink: #edf7fa;
      --muted: #91a9b5;
      --line: rgba(151, 190, 203, .18);
      --cyan: #64c7c4;
      --mint: #72c7a0;
      --gold: #f4b942;
      --coral: #ef7d70;
      --blue: #7f9cf5;
      --shadow: 0 20px 55px rgba(0, 0, 0, .34);
    }
    * { box-sizing: border-box; }
    html, body { margin: 0; min-height: 100%; background: var(--night); }
    body {
      color: var(--ink);
      font-family: Manrope, Inter, Segoe UI, sans-serif;
      overflow: hidden;
    }
    button, input { font: inherit; }
    .app {
      display: grid;
      grid-template-rows: auto minmax(0, 1fr) 214px;
      height: 100vh;
      min-height: 660px;
      background:
        radial-gradient(circle at 70% 10%, rgba(38, 116, 126, .14), transparent 32%),
        var(--night);
    }
    .topbar {
      position: relative;
      z-index: 900;
      display: grid;
      grid-template-columns: minmax(260px, 1fr) minmax(360px, 1.2fr) auto;
      gap: 22px;
      align-items: center;
      min-height: 76px;
      padding: 13px 20px;
      border-bottom: 1px solid var(--line);
      background: rgba(6, 19, 29, .94);
      backdrop-filter: blur(18px);
    }
    .eyebrow {
      color: var(--cyan);
      font-size: 10px;
      font-weight: 700;
      letter-spacing: .14em;
      text-transform: uppercase;
    }
    h1 { margin: 3px 0 0; font-size: 18px; font-weight: 600; letter-spacing: -.02em; }
    .subtitle {
      margin-top: 3px;
      color: var(--muted);
      font-size: 10px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .timeline-control {
      display: grid;
      grid-template-columns: auto minmax(160px, 1fr) auto;
      gap: 12px;
      align-items: center;
    }
    .play-button {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 7px;
      min-width: 82px;
      height: 36px;
      padding: 0 14px;
      border: 1px solid rgba(244, 185, 66, .44);
      border-radius: 999px;
      background: rgba(244, 185, 66, .10);
      color: #ffe6a3;
      cursor: pointer;
    }
    .play-button:hover { background: rgba(244, 185, 66, .18); }
    .play-icon { width: 0; height: 0; border-top: 5px solid transparent; border-bottom: 5px solid transparent; border-left: 8px solid currentColor; }
    .play-button.is-playing .play-icon { width: 8px; height: 10px; border: 0; border-left: 3px solid currentColor; border-right: 3px solid currentColor; }
    input[type="range"] {
      width: 100%;
      height: 4px;
      border-radius: 999px;
      accent-color: var(--gold);
      cursor: pointer;
    }
    .speed-group { display: flex; gap: 4px; }
    .speed-button {
      min-width: 38px;
      height: 28px;
      padding: 0 8px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: transparent;
      color: var(--muted);
      cursor: pointer;
      font-size: 10px;
    }
    .speed-button[aria-pressed="true"] {
      border-color: rgba(100, 199, 196, .52);
      background: rgba(100, 199, 196, .12);
      color: var(--ink);
    }
    .clock { min-width: 125px; text-align: right; }
    .clock-main { font-size: 17px; font-weight: 600; font-variant-numeric: tabular-nums; }
    .clock-sub { margin-top: 2px; color: var(--muted); font-size: 10px; }
    .map-stage { position: relative; min-height: 0; overflow: hidden; }
    #cinematicMap { width: 100%; height: 100%; background: #071923; }
    .leaflet-container { font: 11px Manrope, sans-serif; }
    .leaflet-control-attribution {
      background: rgba(4, 16, 24, .68) !important;
      color: #6f8792 !important;
      font-size: 9px !important;
    }
    .leaflet-control-attribution a { color: #8eb9c2 !important; }
    .leaflet-control-zoom a {
      border-color: var(--line) !important;
      background: rgba(7, 25, 36, .88) !important;
      color: var(--ink) !important;
    }
    .components-panel {
      position: absolute;
      z-index: 720;
      top: 14px;
      bottom: 14px;
      left: 14px;
      width: 286px;
      border: 1px solid var(--line);
      border-radius: 15px;
      background: var(--glass);
      box-shadow: var(--shadow);
      backdrop-filter: blur(18px);
      overflow: hidden;
      transition: width .2s ease, height .2s ease;
    }
    .components-panel.is-collapsed {
      right: auto;
      bottom: auto;
      width: 44px;
      height: 44px;
    }
    .components-header {
      display: flex;
      min-height: 44px;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 10px 10px 9px 13px;
      border-bottom: 1px solid var(--line);
    }
    .components-panel.is-collapsed .components-header {
      justify-content: center;
      padding: 5px;
      border-bottom: 0;
    }
    .components-heading { font-size: 12px; font-weight: 600; }
    .components-count { margin-top: 2px; color: var(--muted); font-size: 8px; letter-spacing: .06em; text-transform: uppercase; }
    .components-toggle {
      display: grid;
      flex: 0 0 auto;
      width: 32px;
      height: 32px;
      place-items: center;
      border: 1px solid var(--line);
      border-radius: 9px;
      background: rgba(255,255,255,.025);
      color: var(--muted);
      cursor: pointer;
    }
    .components-toggle:hover { color: var(--ink); background: rgba(255,255,255,.06); }
    .components-toggle-glyph { display: block; font-size: 17px; transform: rotate(180deg); transition: transform .2s ease; }
    .components-panel.is-collapsed .components-toggle-glyph { transform: rotate(0); }
    .components-panel.is-collapsed .components-copy,
    .components-panel.is-collapsed .components-body { display: none; }
    .components-body {
      height: calc(100% - 44px);
      padding: 10px;
      overflow-y: auto;
      scrollbar-color: rgba(145,169,181,.28) transparent;
      scrollbar-width: thin;
    }
    .component-group + .component-group { margin-top: 13px; }
    .component-group-title {
      display: flex;
      align-items: center;
      gap: 7px;
      margin: 0 3px 7px;
      color: #bfd0d6;
      font-size: 8px;
      letter-spacing: .11em;
      text-transform: uppercase;
    }
    .component-group-title::before {
      width: 5px;
      height: 5px;
      border-radius: 50%;
      background: var(--group-color);
      box-shadow: 0 0 9px color-mix(in srgb, var(--group-color) 70%, transparent);
      content: "";
    }
    .component-list { display: grid; gap: 6px; }
    .component-item {
      display: grid;
      grid-template-columns: 8px minmax(0, 1fr);
      gap: 9px;
      width: 100%;
      padding: 8px 9px;
      border: 1px solid rgba(151,190,203,.12);
      border-radius: 10px;
      background: rgba(255,255,255,.024);
      color: var(--ink);
      text-align: left;
      cursor: pointer;
    }
    .component-item:hover,
    .component-item.is-selected {
      border-color: color-mix(in srgb, var(--item-color) 55%, transparent);
      background: color-mix(in srgb, var(--item-color) 10%, rgba(255,255,255,.018));
    }
    .component-status-dot {
      width: 7px;
      height: 7px;
      margin-top: 4px;
      border-radius: 50%;
      background: var(--item-color);
      box-shadow: 0 0 8px color-mix(in srgb, var(--item-color) 65%, transparent);
    }
    .component-item.is-alert .component-status-dot {
      background: var(--coral);
      animation: componentPulse 1.45s ease-out infinite;
    }
    .component-main { min-width: 0; }
    .component-line { display: flex; align-items: baseline; justify-content: space-between; gap: 8px; }
    .component-name { overflow: hidden; font-size: 9px; font-weight: 600; text-overflow: ellipsis; white-space: nowrap; }
    .component-value { flex: 0 0 auto; color: #d7e7eb; font-size: 8px; font-variant-numeric: tabular-nums; }
    .component-status { margin-top: 2px; overflow: hidden; color: var(--muted); font-size: 8px; text-overflow: ellipsis; white-space: nowrap; }
    .component-meter { height: 3px; margin-top: 6px; overflow: hidden; border-radius: 999px; background: rgba(255,255,255,.07); }
    .component-meter.is-hidden { display: none; }
    .component-meter-fill { height: 100%; border-radius: inherit; background: var(--item-color); }
    @keyframes componentPulse {
      0% { box-shadow: 0 0 0 0 rgba(239,125,112,.45); }
      100% { box-shadow: 0 0 0 9px rgba(239,125,112,0); }
    }
    .kpi-strip {
      position: absolute;
      z-index: 700;
      top: 14px;
      left: 314px;
      display: grid;
      grid-template-columns: repeat(3, minmax(112px, 1fr));
      gap: 8px;
      pointer-events: none;
      transition: left .2s ease;
    }
    .map-stage.components-collapsed .kpi-strip { left: 70px; }
    .kpi {
      min-width: 118px;
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: var(--glass-soft);
      box-shadow: 0 12px 26px rgba(0, 0, 0, .2);
      backdrop-filter: blur(14px);
    }
    .kpi-label { color: var(--muted); font-size: 9px; letter-spacing: .08em; text-transform: uppercase; }
    .kpi-value { margin-top: 3px; font-size: 17px; font-weight: 600; font-variant-numeric: tabular-nums; }
    .kpi-value.is-zero { color: #8be0b6; }
    .kpi-value.is-alert { color: #ff9b8f; }
    .event-pill {
      position: absolute;
      z-index: 700;
      top: 68px;
      left: 50%;
      max-width: min(430px, 45vw);
      padding: 9px 13px;
      transform: translateX(-50%);
      border: 1px solid rgba(244, 185, 66, .26);
      border-radius: 999px;
      background: rgba(8, 27, 39, .84);
      box-shadow: var(--shadow);
      color: #f4d78d;
      font-size: 10px;
      text-align: center;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      backdrop-filter: blur(12px);
      pointer-events: none;
    }
    .detail-panel {
      position: absolute;
      z-index: 700;
      top: 14px;
      right: 14px;
      width: 288px;
      max-height: calc(100% - 28px);
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 15px;
      background: var(--glass);
      box-shadow: var(--shadow);
      backdrop-filter: blur(18px);
      overflow: auto;
    }
    .detail-panel-title { margin-bottom: 8px; color: var(--muted); font-size: 8px; letter-spacing: .11em; text-transform: uppercase; }
    .vessel-tabs { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; }
    .vessel-tab {
      height: 31px;
      border: 1px solid var(--line);
      border-radius: 9px;
      background: rgba(255,255,255,.025);
      color: var(--muted);
      cursor: pointer;
      font-size: 9px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .vessel-tab[aria-pressed="true"] {
      border-color: var(--vessel-color);
      background: color-mix(in srgb, var(--vessel-color) 16%, transparent);
      color: var(--ink);
    }
    .detail-heading { display: flex; align-items: center; gap: 10px; margin-top: 14px; }
    .detail-swatch { width: 9px; height: 34px; border-radius: 999px; background: var(--selected-color); box-shadow: 0 0 20px color-mix(in srgb, var(--selected-color) 65%, transparent); }
    .detail-name { font-size: 15px; font-weight: 600; }
    .detail-state { margin-top: 2px; color: var(--muted); font-size: 10px; }
    .route-line { margin: 15px 0 12px; color: #c9d9de; font-size: 11px; }
    .detail-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .detail-cell { padding: 9px 10px; border: 1px solid var(--line); border-radius: 10px; background: rgba(255,255,255,.025); }
    .detail-cell span { display: block; color: var(--muted); font-size: 8px; letter-spacing: .07em; text-transform: uppercase; }
    .detail-cell strong { display: block; margin-top: 3px; font-size: 12px; font-weight: 600; font-variant-numeric: tabular-nums; }
    .progress-block { margin-top: 12px; }
    .progress-block.is-hidden { display: none; }
    .progress-header { display: flex; justify-content: space-between; gap: 10px; color: var(--muted); font-size: 9px; }
    .progress-track { height: 6px; margin-top: 6px; overflow: hidden; border-radius: 999px; background: rgba(255,255,255,.08); }
    .progress-fill { height: 100%; width: 0; border-radius: inherit; background: var(--selected-color); box-shadow: 0 0 12px color-mix(in srgb, var(--selected-color) 70%, transparent); transition: width .15s linear; }
    .legend {
      position: absolute;
      z-index: 700;
      left: 314px;
      bottom: 14px;
      display: flex;
      flex-wrap: wrap;
      gap: 8px 13px;
      max-width: 520px;
      padding: 8px 11px;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: rgba(7, 24, 35, .74);
      color: var(--muted);
      font-size: 9px;
      backdrop-filter: blur(12px);
      pointer-events: none;
      transition: left .2s ease;
    }
    .map-stage.components-collapsed .legend { left: 70px; }
    .legend-item { display: flex; align-items: center; gap: 6px; }
    .legend-line { width: 22px; height: 0; border-top: 2px solid var(--legend-color); }
    .legend-line.flow { border-top-width: 4px; border-top-style: dashed; }
    .legend-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--legend-color); }
    .grid-coordinate-label {
      border: 0 !important;
      background: transparent !important;
      color: #a9bec7;
      font-size: 8px;
      font-weight: 600;
      text-align: center;
      text-shadow: 0 1px 4px #031019, 0 0 7px #031019;
      white-space: nowrap;
      pointer-events: none;
    }
    .grid-coordinate-label span {
      display: inline-block;
      padding: 2px 4px;
      border: 1px solid rgba(151,190,203,.12);
      border-radius: 5px;
      background: rgba(4,17,25,.68);
    }
    .coordinate-readout {
      position: absolute;
      z-index: 705;
      right: 314px;
      bottom: 14px;
      min-width: 188px;
      padding: 7px 10px;
      border: 1px solid var(--line);
      border-radius: 9px;
      background: rgba(7,24,35,.78);
      color: #b9cbd1;
      font-size: 9px;
      font-variant-numeric: tabular-nums;
      text-align: center;
      backdrop-filter: blur(12px);
      pointer-events: none;
    }
    .facility-icon, .ship-icon { background: transparent; border: 0; }
    .facility-node { position: relative; display: grid; place-items: center; width: 30px; height: 30px; }
    .facility-core {
      position: relative;
      display: grid;
      place-items: center;
      width: 15px;
      height: 15px;
      border: 2px solid var(--facility-color);
      border-radius: 50%;
      background: rgba(6, 19, 29, .92);
      box-shadow: 0 0 0 4px color-mix(in srgb, var(--facility-color) 16%, transparent), 0 0 18px color-mix(in srgb, var(--facility-color) 34%, transparent);
    }
    .facility-node.is-critical .facility-core { animation: criticalPulse 1.45s ease-out infinite; }
    .facility-node.is-selected .facility-core {
      box-shadow: 0 0 0 5px color-mix(in srgb, var(--facility-color) 24%, transparent), 0 0 24px color-mix(in srgb, var(--facility-color) 75%, transparent);
    }
    .facility-node.terminal .facility-core { width: 19px; height: 19px; border-radius: 5px; }
    .facility-node.pipeline .facility-core { width: 13px; height: 13px; border-radius: 2px; transform: rotate(45deg); }
    .facility-node.manifold .facility-core { width: 17px; height: 10px; border-radius: 3px; }
    .facility-node.well .facility-core { width: 12px; height: 19px; border-radius: 999px; }
    .facility-node.reservoir .facility-core { width: 20px; height: 11px; border-style: dashed; border-radius: 50%; }
    .facility-label {
      position: absolute;
      top: 27px;
      left: 50%;
      padding: 3px 6px;
      transform: translateX(-50%);
      border: 1px solid rgba(153,190,203,.14);
      border-radius: 6px;
      background: rgba(5, 18, 27, .82);
      color: #dbe9ed;
      font-size: 8px;
      font-weight: 600;
      white-space: nowrap;
      box-shadow: 0 5px 14px rgba(0,0,0,.22);
    }
    @keyframes criticalPulse {
      0% { box-shadow: 0 0 0 0 rgba(239, 125, 112, .45), 0 0 16px rgba(239, 125, 112, .3); }
      100% { box-shadow: 0 0 0 18px rgba(239, 125, 112, 0), 0 0 20px rgba(239, 125, 112, .1); }
    }
    .ship-marker { position: relative; width: 46px; height: 52px; }
    .ship-glyph {
      position: absolute;
      left: 9px;
      top: 5px;
      width: 28px;
      height: 28px;
      transform: rotate(var(--bearing));
      transform-origin: center;
      filter: drop-shadow(0 0 8px color-mix(in srgb, var(--ship-color) 70%, transparent));
      transition: transform .12s linear;
    }
    .ship-glyph path { fill: var(--ship-color); stroke: #f7fbfc; stroke-width: .8; }
    .ship-tag {
      position: absolute;
      top: 36px;
      left: 50%;
      min-width: 30px;
      padding: 2px 5px;
      transform: translateX(-50%);
      border: 1px solid color-mix(in srgb, var(--ship-color) 55%, transparent);
      border-radius: 999px;
      background: rgba(5, 18, 27, .84);
      color: #f6fbfc;
      font-size: 8px;
      font-weight: 700;
      text-align: center;
      white-space: nowrap;
    }
    .ship-marker.is-selected .ship-glyph { filter: drop-shadow(0 0 13px var(--ship-color)); }
    .ship-marker.is-selected .ship-tag { background: color-mix(in srgb, var(--ship-color) 20%, rgba(5,18,27,.88)); }
    .leaflet-interactive.co2-flow { animation: flowDash 1.15s linear infinite; }
    @keyframes flowDash { to { stroke-dashoffset: -28; } }
    .chart-stage {
      position: relative;
      z-index: 800;
      display: grid;
      grid-template-rows: 31px minmax(0, 1fr);
      min-height: 0;
      border-top: 1px solid var(--line);
      background: linear-gradient(180deg, #091c28 0%, #071721 100%);
    }
    .chart-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      padding: 5px 20px 0;
      color: var(--muted);
      font-size: 9px;
    }
    .chart-title { color: #dbe8ec; font-size: 10px; font-weight: 600; letter-spacing: .03em; }
    .chart-legend { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 6px 15px; }
    .chart-legend-item { display: flex; align-items: center; gap: 5px; }
    .chart-swatch { width: 15px; height: 2px; background: var(--series-color); }
    #systemChart { display: block; width: 100%; height: 100%; min-height: 0; cursor: crosshair; }
    .theme-switcher {
      display: inline-flex;
      align-items: center;
      gap: 2px;
      margin-bottom: 6px;
      padding: 3px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: rgba(255, 255, 255, .04);
    }
    .theme-choice {
      min-width: 50px;
      padding: 4px 9px;
      border: 0;
      border-radius: 999px;
      background: transparent;
      color: var(--muted);
      font-size: 9px;
      font-weight: 700;
      cursor: pointer;
      transition: background .18s ease, color .18s ease, box-shadow .18s ease;
    }
    .theme-choice:hover { color: var(--ink); }
    .theme-choice[aria-pressed="true"] {
      background: rgba(100, 199, 196, .15);
      color: var(--cyan);
      box-shadow: inset 0 0 0 1px rgba(100, 199, 196, .22);
    }
    @media (max-width: 980px) {
      body { overflow: auto; }
      .app { height: auto; min-height: 100vh; grid-template-rows: auto 620px 240px; }
      .topbar { grid-template-columns: 1fr; gap: 10px; }
      .clock { position: absolute; top: 16px; right: 18px; }
      .detail-panel { width: 260px; }
      .coordinate-readout { right: 286px; }
    }
    @media (max-width: 680px) {
      .app { grid-template-rows: auto 660px 250px; }
      .topbar { padding: 12px; }
      .timeline-control { grid-template-columns: auto 1fr; }
      .speed-group { grid-column: 1 / -1; }
      .clock { position: static; text-align: left; }
      .kpi-strip { left: 70px; grid-template-columns: 1fr; }
      .event-pill { top: auto; bottom: 72px; max-width: 80vw; }
      .detail-panel { top: auto; right: 10px; bottom: 10px; left: 10px; width: auto; max-height: 265px; }
      .legend { display: none; }
      .coordinate-readout { display: none; }
      .chart-legend { gap: 5px 8px; }
    }
    @media (prefers-reduced-motion: reduce) {
      .leaflet-interactive.co2-flow, .facility-node.is-critical .facility-core, .component-item.is-alert .component-status-dot { animation: none; }
      .ship-glyph, .progress-fill { transition: none; }
    }
  </style>
  <style id="lightThemeStyles">
    /*
     * CCS_RL Physical Layer Dashboard — Light V2
     * The original cinematic rules remain above; this layer changes only presentation.
     */
    :root {
      color-scheme: light;
      --night: #f3f7f5;
      --night-raised: #ffffff;
      --glass: rgba(255, 255, 255, .91);
      --glass-soft: rgba(255, 255, 255, .84);
      --ink: #14343d;
      --muted: #5d747b;
      --line: rgba(24, 70, 79, .14);
      --cyan: #087f7b;
      --mint: #278b65;
      --gold: #a76500;
      --coral: #cf5e53;
      --blue: #4f6fc8;
      --shadow: 0 18px 48px rgba(31, 70, 71, .14);
    }
    html, body { background: var(--night); }
    body { color: var(--ink); }
    .app {
      background:
        radial-gradient(circle at 73% 8%, rgba(8, 127, 123, .09), transparent 34%),
        linear-gradient(180deg, #fbfcfa 0%, var(--night) 100%);
    }
    .topbar {
      border-bottom-color: rgba(24, 70, 79, .13);
      background: rgba(251, 253, 251, .94);
      box-shadow: 0 8px 28px rgba(31, 70, 71, .07);
    }
    .play-button {
      border-color: rgba(167, 101, 0, .35);
      background: rgba(230, 165, 52, .15);
      color: #865200;
    }
    .play-button:hover { background: rgba(230, 165, 52, .24); }
    .speed-button { background: rgba(255, 255, 255, .7); }
    .speed-button[aria-pressed="true"] {
      border-color: rgba(8, 127, 123, .42);
      background: rgba(8, 127, 123, .09);
    }
    .theme-switcher {
      border-color: rgba(24, 70, 79, .13);
      background: rgba(234, 241, 238, .78);
    }
    .theme-choice[aria-pressed="true"] {
      background: #ffffff;
      color: #087f7b;
      box-shadow: 0 2px 8px rgba(31, 70, 71, .12);
    }
    #cinematicMap { background: #e9efec; }
    .leaflet-control-attribution {
      background: rgba(255, 255, 255, .82) !important;
      color: #657c82 !important;
    }
    .leaflet-control-attribution a { color: #176f75 !important; }
    .leaflet-control-zoom {
      border: 0 !important;
      box-shadow: 0 8px 24px rgba(31, 70, 71, .13) !important;
    }
    .leaflet-control-zoom a {
      border-color: rgba(24, 70, 79, .12) !important;
      background: rgba(255, 255, 255, .94) !important;
      color: #244b54 !important;
    }
    .components-panel,
    .detail-panel {
      border-color: rgba(24, 70, 79, .13);
      background: var(--glass);
      box-shadow: var(--shadow);
    }
    .components-toggle,
    .component-item,
    .vessel-tab,
    .detail-cell {
      border-color: rgba(24, 70, 79, .11);
      background: rgba(248, 251, 249, .88);
    }
    .components-toggle:hover { background: rgba(8, 127, 123, .07); }
    .components-body { scrollbar-color: rgba(56, 96, 101, .3) transparent; }
    .component-group-title,
    .component-value,
    .route-line { color: #355861; }
    .component-item:hover,
    .component-item.is-selected {
      border-color: color-mix(in srgb, var(--item-color) 48%, rgba(24,70,79,.16));
      background: color-mix(in srgb, var(--item-color) 10%, white);
      box-shadow: 0 7px 18px rgba(31, 70, 71, .06);
    }
    .component-meter,
    .progress-track { background: rgba(24, 70, 79, .09); }
    .kpi {
      border-color: rgba(24, 70, 79, .12);
      background: rgba(255, 255, 255, .87);
      box-shadow: 0 12px 28px rgba(31, 70, 71, .1);
    }
    .kpi-value.is-zero { color: #19724f; }
    .kpi-value.is-alert { color: #b9473d; }
    .event-pill {
      border-color: rgba(167, 101, 0, .24);
      background: rgba(255, 249, 233, .93);
      color: #7c4c00;
      box-shadow: 0 12px 30px rgba(84, 69, 31, .11);
    }
    .legend,
    .coordinate-readout {
      border-color: rgba(24, 70, 79, .12);
      background: rgba(255, 255, 255, .88);
      color: #4e6c73;
      box-shadow: 0 10px 26px rgba(31, 70, 71, .09);
    }
    .grid-coordinate-label {
      color: #385c64;
      text-shadow: 0 1px 0 rgba(255,255,255,.9), 0 0 5px rgba(255,255,255,.95);
    }
    .grid-coordinate-label span {
      border-color: rgba(24, 70, 79, .12);
      background: rgba(255, 255, 255, .74);
    }
    .facility-core {
      background: rgba(255, 255, 255, .96);
      box-shadow: 0 0 0 4px color-mix(in srgb, var(--facility-color) 13%, transparent), 0 5px 16px rgba(25, 63, 66, .16);
    }
    .facility-label,
    .ship-tag {
      border-color: rgba(24, 70, 79, .15);
      background: rgba(255, 255, 255, .94);
      color: #173b44;
      box-shadow: 0 5px 14px rgba(31, 70, 71, .12);
    }
    .ship-glyph path {
      stroke: rgba(255, 255, 255, .94);
      stroke-width: 1.1;
    }
    .ship-marker.is-selected .ship-tag {
      background: color-mix(in srgb, var(--ship-color) 12%, white);
    }
    .chart-stage {
      border-top-color: rgba(24, 70, 79, .13);
      background: linear-gradient(180deg, rgba(237, 244, 241, .78), rgba(255, 255, 255, .96));
      box-shadow: 0 -8px 28px rgba(31, 70, 71, .06);
    }
    .chart-title { color: #244b54; }
    @media (max-width: 680px) {
      .topbar { background: rgba(251, 253, 251, .97); }
    }
  </style>
  <script>
    document.getElementById("lightThemeStyles").media =
      document.documentElement.dataset.theme === "dark" ? "not all" : "all";
  </script>
</head>
<body>
  <div class="app">
    <header class="topbar">
      <div>
        <div class="eyebrow">CCS fleet operations · E1 replay</div>
        <h1>__TITLE__</h1>
        <div class="subtitle">__SUBTITLE__</div>
      </div>
      <div class="timeline-control">
        <button id="playPause" class="play-button" type="button"><span class="play-icon" aria-hidden="true"></span><span id="playLabel">Play</span></button>
        <input id="timeline" type="range" min="0" max="720" step="0.05" value="0" aria-label="Simulation hour">
        <div class="speed-group" aria-label="Playback speed">
          <button class="speed-button" type="button" data-speed="12" aria-pressed="false">0.5×</button>
          <button class="speed-button" type="button" data-speed="24" aria-pressed="true">1×</button>
          <button class="speed-button" type="button" data-speed="48" aria-pressed="false">2×</button>
          <button class="speed-button" type="button" data-speed="96" aria-pressed="false">4×</button>
        </div>
      </div>
      <div class="clock">
        <div class="theme-switcher" role="group" aria-label="Visual theme">
          <button class="theme-choice" type="button" data-theme-choice="light-v2" aria-pressed="true">Light</button>
          <button class="theme-choice" type="button" data-theme-choice="dark" aria-pressed="false">Dark</button>
        </div>
        <div id="clockMain" class="clock-main">Day 00 · 00:00</div>
        <div id="clockSub" class="clock-sub">Hour 000 / 720</div>
      </div>
    </header>

    <main class="map-stage">
      <div id="cinematicMap" role="img" aria-label="Animated Northern Lights vessel and CO2 transport map"></div>
      <aside id="componentsPanel" class="components-panel" aria-label="Physical network components">
        <div class="components-header">
          <div class="components-copy">
            <div class="components-heading">Components</div>
            <div id="componentsCount" class="components-count">Live physical network</div>
          </div>
          <button id="componentsToggle" class="components-toggle" type="button" aria-expanded="true" aria-label="Collapse components panel">
            <span class="components-toggle-glyph" aria-hidden="true">›</span>
          </button>
        </div>
        <div id="componentsBody" class="components-body"></div>
      </aside>
      <div class="kpi-strip">
        <div class="kpi"><div class="kpi-label">Cumulative vent</div><div id="ventValue" class="kpi-value is-zero">0 t</div></div>
        <div class="kpi"><div class="kpi-label">CO₂ in transit</div><div id="transitValue" class="kpi-value">0 t</div></div>
        <div class="kpi"><div class="kpi-label">Terminal inventory</div><div id="terminalValue" class="kpi-value">0 t</div></div>
      </div>
      <div id="eventPill" class="event-pill">Representative E1 trajectory · playback ready</div>
      <aside class="detail-panel">
        <div class="detail-panel-title">Selected status</div>
        <div id="vesselTabs" class="vessel-tabs"></div>
        <div class="detail-heading">
          <span class="detail-swatch"></span>
          <div><div id="detailName" class="detail-name">Vessel</div><div id="detailState" class="detail-state">Loading</div></div>
        </div>
        <div id="detailRoute" class="route-line">Brevik → Øygarden</div>
        <div class="detail-grid">
          <div class="detail-cell"><span id="detailMetric1Label">Cargo</span><strong id="detailMetric1Value">0 t</strong></div>
          <div class="detail-cell"><span id="detailMetric2Label">Weather factor</span><strong id="detailMetric2Value">1.00×</strong></div>
          <div class="detail-cell"><span>Latitude</span><strong id="detailLat">0.000°</strong></div>
          <div class="detail-cell"><span>Longitude</span><strong id="detailLon">0.000°</strong></div>
        </div>
        <div id="primaryProgressBlock" class="progress-block">
          <div class="progress-header"><span id="primaryProgressLabel">Cargo utilisation</span><strong id="primaryProgressValue">0%</strong></div>
          <div class="progress-track"><div id="primaryProgress" class="progress-fill"></div></div>
        </div>
        <div id="secondaryProgressBlock" class="progress-block">
          <div class="progress-header"><span id="secondaryProgressLabel">Current leg</span><strong id="secondaryProgressValue">0%</strong></div>
          <div class="progress-track"><div id="secondaryProgress" class="progress-fill"></div></div>
        </div>
      </aside>
      <div id="coordinateReadout" class="coordinate-readout" aria-live="polite">Move cursor for coordinates</div>
      <div class="legend">
        <span class="legend-item"><span class="legend-line" style="--legend-color:#456d78"></span>Service corridor</span>
        <span class="legend-item"><span class="legend-line flow" style="--legend-color:#64c7c4"></span>Offshore CO₂ flow</span>
        <span class="legend-item"><span class="legend-dot" style="--legend-color:#72c7a0"></span>Capture / storage</span>
        <span class="legend-item"><span class="legend-dot" style="--legend-color:#ef7d70"></span>High buffer / venting</span>
      </div>
    </main>

    <section class="chart-stage">
      <div class="chart-header">
        <span id="chartTitle" class="chart-title">Selected component history · tonnes</span>
        <div id="chartLegend" class="chart-legend"></div>
      </div>
      <canvas id="systemChart" role="img" aria-label="System inventory and cumulative vent over the 720-hour trajectory"></canvas>
    </section>
  </div>

  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>window.__CINEMATIC_DATA__ = __DATA_JSON__;</script>
  <script>
    const data = window.__CINEMATIC_DATA__;
    const frames = data.frames;
    const events = data.events;
    const vesselById = Object.fromEntries(data.vessels.map(vessel => [vessel.id, vessel]));
    const emitterById = Object.fromEntries(data.emitters.map(emitter => [emitter.id, emitter]));
    const duration = Number(data.duration_hours);
    const timeline = document.getElementById("timeline");
    const playPause = document.getElementById("playPause");
    const playLabel = document.getElementById("playLabel");
    const chart = document.getElementById("systemChart");
    const chartContext = chart.getContext("2d");
    const mapStage = document.querySelector(".map-stage");
    const componentsPanel = document.getElementById("componentsPanel");
    const componentsToggle = document.getElementById("componentsToggle");
    const componentsBody = document.getElementById("componentsBody");
    const coordinateReadout = document.getElementById("coordinateReadout");
    const lightThemeStyles = document.getElementById("lightThemeStyles");
    const themeChoices = document.querySelectorAll("[data-theme-choice]");
    const vesselMarkers = {};
    const vesselTrails = {};
    const facilityMarkers = {};
    const routeLayers = {};
    const pipelineLayers = [];
    const injectionLayers = [];
    const componentNodes = {};
    let currentHour = 0;
    let selectedVessel = data.vessels[0].id;
    let selectedComponent = selectedVessel;
    let hoursPerSecond = 24;
    let isPlaying = false;
    let animationFrame = null;
    let previousTimestamp = null;
    let hoveredHour = null;
    let activeChartKey = null;

    const map = L.map("cinematicMap", {
      zoomControl: false,
      scrollWheelZoom: true,
      attributionControl: true,
      preferCanvas: false
    });
    L.control.zoom({position: "bottomright"}).addTo(map);
    const lightTileUrl = "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png";
    const darkTileUrl = "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png";
    const baseTileLayer = L.tileLayer(
      document.documentElement.dataset.theme === "dark" ? darkTileUrl : lightTileUrl,
      {
      subdomains: "abcd",
      maxZoom: 12,
      attribution: "&copy; OpenStreetMap contributors &copy; CARTO"
      }
    ).addTo(map);
    map.createPane("graticulePane");
    map.getPane("graticulePane").style.zIndex = 340;
    map.getPane("graticulePane").style.pointerEvents = "none";
    map.createPane("graticuleLabelPane");
    map.getPane("graticuleLabelPane").style.zIndex = 610;
    map.getPane("graticuleLabelPane").style.pointerEvents = "none";
    const graticuleLayer = L.layerGroup().addTo(map);
    const bbox = data.map.bbox;
    map.fitBounds([[bbox.min_lat, bbox.min_lon], [bbox.max_lat, bbox.max_lon]], {padding: [18, 18]});

    function isDarkTheme() {
      return document.documentElement.dataset.theme === "dark";
    }

    function themeColor(lightColor, darkColor) {
      return isDarkTheme() ? darkColor : lightColor;
    }

    function seriesColor(series) {
      return typeof series.color === "function" ? series.color() : series.color;
    }

    function formatTonnes(value) {
      const numeric = Number(value || 0);
      if (Math.abs(numeric) >= 1000000) return `${(numeric / 1000000).toFixed(2)} Mt`;
      if (Math.abs(numeric) >= 1000) return `${(numeric / 1000).toFixed(1)} kt`;
      return `${numeric.toFixed(numeric < 10 ? 1 : 0)} t`;
    }

    function friendly(value) {
      const names = {
        brevik: "Brevik",
        celsio: "Celsio",
        yara_sluiskil: "Yara Sluiskil",
        oygarden_terminal: "Øygarden",
        loading: "Loading",
        unloading: "Unloading",
        idle: "Idle",
        queued: "Queued",
        sailing_to_terminal: "Sailing → terminal",
        sailing_to_emitter: "Sailing → capture site"
      };
      return names[value] || String(value).replaceAll("_", " ").replace(/\\b\\w/g, char => char.toUpperCase());
    }

    function formatCoordinate(value, axis, decimals = 5) {
      const numeric = Number(value);
      const hemisphere = axis === "lat"
        ? (numeric >= 0 ? "N" : "S")
        : (numeric >= 0 ? "E" : "W");
      return `${Math.abs(numeric).toFixed(decimals)}°${hemisphere}`;
    }

    function gridStep(spanDegrees) {
      if (spanDegrees <= 1) return .1;
      if (spanDegrees <= 3) return .25;
      if (spanDegrees <= 8) return .5;
      if (spanDegrees <= 16) return 1;
      return 2;
    }

    function drawLatLonGrid() {
      graticuleLayer.clearLayers();
      const bounds = map.getBounds();
      const south = bounds.getSouth();
      const north = bounds.getNorth();
      const west = bounds.getWest();
      const east = bounds.getEast();
      const step = gridStep(Math.max(north - south, east - west));
      const decimals = step < 1 ? (step < .2 ? 2 : 1) : 0;
      const panelOffset = componentsPanel.classList.contains("is-collapsed")
        ? 13
        : componentsPanel.getBoundingClientRect().width + 23;
      const labelLon = Math.min(
        east - step * .08,
        Math.max(west + step * .08, map.containerPointToLatLng([panelOffset, 20]).lng)
      );
      const labelLat = Math.min(
        north - step * .08,
        Math.max(south + step * .08, map.containerPointToLatLng([20, map.getSize().y - 24]).lat)
      );
      for (let lat = Math.ceil(south / step) * step; lat <= north; lat += step) {
        const value = Number(lat.toFixed(decimals));
        L.polyline([[value, west], [value, east]], {
          pane: "graticulePane",
          color: themeColor("#54757c", "#8aa8b3"),
          weight: .8,
          opacity: .27,
          dashArray: "3 7",
          interactive: false
        }).addTo(graticuleLayer);
        L.marker([value, labelLon], {
          pane: "graticuleLabelPane",
          interactive: false,
          icon: L.divIcon({
            className: "grid-coordinate-label",
            html: `<span>${formatCoordinate(value, "lat", decimals)}</span>`,
            iconSize: [54, 18],
            iconAnchor: [0, 9]
          })
        }).addTo(graticuleLayer);
      }
      for (let lon = Math.ceil(west / step) * step; lon <= east; lon += step) {
        const value = Number(lon.toFixed(decimals));
        L.polyline([[south, value], [north, value]], {
          pane: "graticulePane",
          color: themeColor("#54757c", "#8aa8b3"),
          weight: .8,
          opacity: .27,
          dashArray: "3 7",
          interactive: false
        }).addTo(graticuleLayer);
        L.marker([labelLat, value], {
          pane: "graticuleLabelPane",
          interactive: false,
          icon: L.divIcon({
            className: "grid-coordinate-label",
            html: `<span>${formatCoordinate(value, "lon", decimals)}</span>`,
            iconSize: [54, 18],
            iconAnchor: [27, 18]
          })
        }).addTo(graticuleLayer);
      }
    }

    function setComponentsCollapsed(collapsed) {
      componentsPanel.classList.toggle("is-collapsed", collapsed);
      mapStage.classList.toggle("components-collapsed", collapsed);
      componentsToggle.setAttribute("aria-expanded", String(!collapsed));
      componentsToggle.setAttribute("aria-label", collapsed ? "Expand components panel" : "Collapse components panel");
      setTimeout(() => {
        map.invalidateSize();
        drawLatLonGrid();
      }, 220);
    }

    function interpolate(a, b, fraction) {
      return Number(a || 0) + (Number(b || 0) - Number(a || 0)) * fraction;
    }

    function interpolateAngle(a, b, fraction) {
      const delta = ((Number(b) - Number(a) + 540) % 360) - 180;
      return Number(a) + delta * fraction;
    }

    function framePair(hour) {
      const lower = Math.max(0, Math.min(frames.length - 1, Math.floor(hour)));
      const upper = Math.min(frames.length - 1, lower + 1);
      return {lower, upper, fraction: Math.max(0, Math.min(1, hour - lower))};
    }

    function vesselIcon(vessel, state, selected) {
      const shortName = vessel.label.split(" ").slice(-1)[0];
      return L.divIcon({
        className: "ship-icon",
        iconSize: [46, 52],
        iconAnchor: [23, 21],
        html: `<div class="ship-marker${selected ? " is-selected" : ""}" style="--ship-color:${vessel.color};--bearing:${state.bearing_deg}deg">
          <svg class="ship-glyph" viewBox="0 0 32 32" aria-hidden="true"><path d="M16 1.5 26.5 24 16 30.5 5.5 24 16 1.5Zm0 6.2-5.5 14.5 5.5 3.1 5.5-3.1L16 7.7Z"/></svg>
          <span class="ship-tag">${shortName}</span>
        </div>`
      });
    }

    function facilityIcon(label, color, type) {
      return L.divIcon({
        className: "facility-icon",
        iconSize: [30, 44],
        iconAnchor: [15, 15],
        html: `<div class="facility-node ${type}" style="--facility-color:${color}">
          <span class="facility-core"></span><span class="facility-label">${label}</span>
        </div>`
      });
    }

    Object.values(data.map.service_routes).forEach(route => {
      routeLayers[route.id] = L.polyline(route.coordinates, {
        color: themeColor("#5d7d83", "#456d78"),
        weight: 1.7,
        opacity: .42,
        className: "service-route"
      }).bindTooltip(`${route.label}<br>${route.distance_km.toFixed(0)} km`).addTo(map);
    });
    (data.map.pipeline_segments || []).forEach(segment => {
      const isSubsea = segment.style === "subsea_connection";
      const layer = L.polyline(segment.coordinates, {
        color: isSubsea
          ? themeColor("#4f6fc8", "#7f9cf5")
          : themeColor("#087f7b", "#64c7c4"),
        weight: isSubsea ? 2.5 : 5,
        opacity: isSubsea ? .62 : .9,
        dashArray: isSubsea ? "3 8" : "10 11",
        className: isSubsea ? "subsea-link" : "co2-flow"
      }).bindTooltip(segment.label).addTo(map);
      pipelineLayers.push({layer, isSubsea});
    });
    (data.map.injection_links || []).forEach(link => {
      const layer = L.polyline(link.coordinates, {
        color: themeColor("#278b65", "#72c7a0"),
        weight: 2,
        opacity: .48,
        dashArray: "2 8"
      }).addTo(map);
      injectionLayers.push(layer);
    });

    data.emitters.forEach(emitter => {
      const marker = L.marker([emitter.lat, emitter.lon], {
        icon: facilityIcon(emitter.label, emitter.color, "emitter")
      }).bindTooltip(`${emitter.label}<br>Capture buffer`).on("click", () => {
        selectedComponent = emitter.id;
        render(currentHour);
      });
      marker.addTo(map);
      facilityMarkers[emitter.id] = marker;
    });
    const terminal = data.terminal;
    facilityMarkers[terminal.id] = L.marker([terminal.lat, terminal.lon], {
      icon: facilityIcon("Øygarden", "#72c7a0", "terminal")
    }).bindTooltip(`${terminal.label}<br>Onshore receiving terminal`).on("click", () => {
      selectedComponent = terminal.id;
      render(currentHour);
    }).addTo(map);
    const infrastructureColors = {
      terminal: "#72c7a0",
      pipeline: "#64c7c4",
      manifold: "#7f9cf5",
      well: "#72c7a0",
      reservoir: "#b18ae8"
    };
    const componentById = Object.fromEntries([
      ...data.components.capture_sites,
      ...data.components.fleet,
      ...data.components.transport_storage
    ].map(component => [component.id, component]));
    data.components.transport_storage.forEach(component => {
      if (facilityMarkers[component.id]) return;
      const color = infrastructureColors[component.type] || "#91a9b5";
      facilityMarkers[component.id] = L.marker([component.lat, component.lon], {
        icon: facilityIcon(component.short_label, color, component.type)
      }).bindTooltip(
        `${component.label}<br>${formatCoordinate(component.lat, "lat")}, ${formatCoordinate(component.lon, "lon")}`
      ).on("click", () => {
        selectedComponent = component.id;
        render(currentHour);
      }).addTo(map);
    });

    data.vessels.forEach(vessel => {
      const initialState = frames[0].vessels[vessel.id];
      vesselTrails[vessel.id] = L.polyline([], {
        color: vessel.color,
        weight: vessel.id === selectedVessel ? 3.2 : 2,
        opacity: vessel.id === selectedVessel ? .78 : .42
      }).addTo(map);
      vesselMarkers[vessel.id] = L.marker([initialState.lat, initialState.lon], {
        icon: vesselIcon(vessel, initialState, vessel.id === selectedVessel),
        zIndexOffset: vessel.id === selectedVessel ? 600 : 400
      }).on("click", () => {
        selectedVessel = vessel.id;
        selectedComponent = vessel.id;
        render(currentHour);
      }).addTo(map);
    });

    function renderVesselTabs() {
      const tabs = document.getElementById("vesselTabs");
      if (!tabs.children.length) {
        data.vessels.forEach(vessel => {
          const button = document.createElement("button");
          button.type = "button";
          button.className = "vessel-tab";
          button.dataset.vesselId = vessel.id;
          button.style.setProperty("--vessel-color", vessel.color);
          button.textContent = vessel.label.replace(/^Northern\\s+/i, "");
          button.setAttribute("aria-label", `Select ${vessel.label}`);
          button.addEventListener("click", () => {
            selectedVessel = vessel.id;
            selectedComponent = vessel.id;
            render(currentHour);
          });
          tabs.appendChild(button);
        });
      }
      tabs.querySelectorAll(".vessel-tab").forEach(button => {
        button.setAttribute(
          "aria-pressed",
          String(button.dataset.vesselId === selectedComponent)
        );
      });
    }

    function focusComponent(component) {
      selectedComponent = component.id;
      if (vesselById[component.id]) {
        selectedVessel = component.id;
        const vesselState = frames[framePair(currentHour).lower].vessels[component.id];
        map.flyTo([vesselState.lat, vesselState.lon], Math.max(map.getZoom(), 6), {duration: .5});
        render(currentHour);
        return;
      }
      const marker = facilityMarkers[component.id];
      if (marker) {
        map.flyTo(marker.getLatLng(), Math.max(map.getZoom(), 7), {duration: .5});
        marker.openTooltip();
      } else if (Number.isFinite(component.lat) && Number.isFinite(component.lon)) {
        map.flyTo([component.lat, component.lon], Math.max(map.getZoom(), 7), {duration: .5});
      }
      render(currentHour);
    }

    function appendComponentGroup(title, color, items) {
      const group = document.createElement("section");
      group.className = "component-group";
      const heading = document.createElement("h2");
      heading.className = "component-group-title";
      heading.style.setProperty("--group-color", color);
      heading.textContent = title;
      group.appendChild(heading);
      const list = document.createElement("div");
      list.className = "component-list";
      items.forEach(item => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "component-item";
        button.style.setProperty("--item-color", item.color || color);
        button.setAttribute("aria-label", `Select ${item.label}`);
        button.addEventListener("click", () => focusComponent(item));

        const dot = document.createElement("span");
        dot.className = "component-status-dot";
        dot.setAttribute("aria-hidden", "true");
        button.appendChild(dot);

        const main = document.createElement("span");
        main.className = "component-main";
        const line = document.createElement("span");
        line.className = "component-line";
        const name = document.createElement("span");
        name.className = "component-name";
        name.textContent = item.label;
        const value = document.createElement("span");
        value.className = "component-value";
        line.append(name, value);
        const status = document.createElement("span");
        status.className = "component-status";
        main.append(line, status);
        const meter = document.createElement("span");
        meter.className = "component-meter";
        const fill = document.createElement("span");
        fill.className = "component-meter-fill";
        meter.appendChild(fill);
        main.appendChild(meter);
        button.appendChild(main);
        list.appendChild(button);
        componentNodes[item.id] = {button, value, status, meter, fill};
      });
      group.appendChild(list);
      componentsBody.appendChild(group);
    }

    function renderComponents(frame) {
      if (!componentsBody.children.length) {
        appendComponentGroup(
          "Capture sites",
          "#f4b942",
          data.components.capture_sites
        );
        appendComponentGroup(
          "Fleet",
          "#7f9cf5",
          data.components.fleet
        );
        appendComponentGroup(
          "Transport & storage",
          "#72c7a0",
          data.components.transport_storage.map(component => ({
            ...component,
            color: infrastructureColors[component.type]
          }))
        );
        const count = Object.keys(componentNodes).length;
        document.getElementById("componentsCount").textContent = `${count} live assets · click to select`;
      }
      const captureItems = data.components.capture_sites.map(emitter => {
        const state = frame.emitters[emitter.id];
        return {
          ...emitter,
          value: `${Math.round(state.fill_fraction * 100)}%`,
          status: state.capture_outage
            ? "Capture outage"
            : `${formatTonnes(state.inventory_t)} / ${formatTonnes(emitter.capacity_t)} · ${Math.round(state.capture_availability * 100)}% capture`,
          fraction: state.fill_fraction,
          alert: state.capture_outage || state.fill_fraction >= .9
        };
      });
      const fleetItems = data.components.fleet.map(vessel => {
        const state = frame.vessels[vessel.id];
        return {
          ...vessel,
          value: `${Math.round(state.fill_fraction * 100)}%`,
          status: `${friendly(state.operational_state)} · ${friendly(state.destination)}`,
          fraction: state.fill_fraction,
          alert: false
        };
      });
      const offshoreSegment = (data.map.pipeline_segments || []).find(segment => segment.style !== "subsea_connection");
      const storageItems = data.components.transport_storage.map(component => {
        if (component.type === "terminal") {
          return {
            ...component,
            color: infrastructureColors.terminal,
            value: `${Math.round(frame.terminal_fill_fraction * 100)}%`,
            status: `${formatTonnes(frame.terminal_inventory_t)} / ${formatTonnes(component.capacity_t)} · receiving hub`,
            fraction: frame.terminal_fill_fraction,
            alert: frame.terminal_fill_fraction >= .9
          };
        }
        if (component.type === "well") {
          const available = frame.well_available[component.id] !== false;
          return {
            ...component,
            color: infrastructureColors.well,
            value: available ? "Online" : "Offline",
            status: available ? "Injection capacity available" : "Injection unavailable",
            fraction: null,
            alert: !available
          };
        }
        const descriptions = {
          pipeline: offshoreSegment ? `${Number(offshoreSegment.length_km || 0).toFixed(1)} km offshore CO₂ trunkline` : "Offshore CO₂ trunkline",
          manifold: "Subsea distribution hub",
          reservoir: "Permanent geological storage"
        };
        return {
          ...component,
          color: infrastructureColors[component.type],
          value: "Active",
          status: descriptions[component.type],
          fraction: null,
          alert: false
        };
      });
      [...captureItems, ...fleetItems, ...storageItems].forEach(item => {
        const nodes = componentNodes[item.id];
        if (!nodes) return;
        nodes.button.classList.toggle("is-alert", Boolean(item.alert));
        nodes.button.classList.toggle("is-selected", item.id === selectedComponent);
        nodes.button.style.setProperty("--item-color", item.color || "#91a9b5");
        nodes.button.setAttribute("aria-label", `${item.label}: ${item.status}`);
        nodes.value.textContent = item.value || "";
        nodes.status.textContent = item.status;
        const hasFraction = item.fraction !== null && item.fraction !== undefined;
        nodes.meter.classList.toggle("is-hidden", !hasFraction);
        if (hasFraction) {
          nodes.fill.style.width = `${Math.max(0, Math.min(1, item.fraction)) * 100}%`;
        }
      });
    }

    function updateFacilities(frame) {
      data.emitters.forEach(emitter => {
        const state = frame.emitters[emitter.id];
        const markerElement = facilityMarkers[emitter.id].getElement();
        if (!markerElement) return;
        const node = markerElement.querySelector(".facility-node");
        node.classList.toggle("is-critical", state.fill_fraction >= .9 || state.capture_outage);
        const status = state.capture_outage ? "capture outage" : `${Math.round(state.fill_fraction * 100)}% buffer`;
        facilityMarkers[emitter.id].setTooltipContent(`${emitter.label}<br>${formatTonnes(state.inventory_t)} · ${status}`);
      });
      const terminalElement = facilityMarkers[terminal.id].getElement();
      if (terminalElement) terminalElement.querySelector(".facility-node").classList.toggle("is-critical", frame.terminal_fill_fraction >= .9);
      facilityMarkers[terminal.id].setTooltipContent(`${terminal.label}<br>${formatTonnes(frame.terminal_inventory_t)} · ${Math.round(frame.terminal_fill_fraction * 100)}% full`);
    }

    function updateFacilitySelection() {
      Object.entries(facilityMarkers).forEach(([componentId, marker]) => {
        const markerElement = marker.getElement();
        if (!markerElement) return;
        const node = markerElement.querySelector(".facility-node");
        if (node) node.classList.toggle("is-selected", componentId === selectedComponent);
      });
    }

    function updateRouteHighlights(state) {
      Object.values(data.map.service_routes).forEach(route => {
        const selectedIsVessel = Boolean(vesselById[selectedComponent]);
        const active = selectedIsVessel
          ? [route.origin, route.destination].includes(state.origin)
            && [route.origin, route.destination].includes(state.destination)
          : [route.origin, route.destination].includes(selectedComponent);
        const selectedColor = selectedIsVessel
          ? vesselById[selectedVessel].color
          : emitterById[selectedComponent]?.color || themeColor("#087f7b", "#64c7c4");
        routeLayers[route.id].setStyle({
          color: active ? selectedColor : themeColor("#5d7d83", "#456d78"),
          weight: active ? 3.2 : 1.7,
          opacity: active ? .82 : .34
        });
      });
    }

    function updateVessels(pair) {
      const currentFrame = frames[pair.lower];
      const nextFrame = frames[pair.upper];
      data.vessels.forEach(vessel => {
        const current = currentFrame.vessels[vessel.id];
        const next = nextFrame.vessels[vessel.id];
        const displayState = {
          lat: interpolate(current.lat, next.lat, pair.fraction),
          lon: interpolate(current.lon, next.lon, pair.fraction),
          bearing_deg: interpolateAngle(current.bearing_deg, next.bearing_deg, pair.fraction)
        };
        const selected = vessel.id === selectedComponent;
        const marker = vesselMarkers[vessel.id];
        marker.setLatLng([displayState.lat, displayState.lon]);
        marker.setZIndexOffset(selected ? 600 : 400);
        const markerElement = marker.getElement();
        if (markerElement) {
          const wrapper = markerElement.querySelector(".ship-marker");
          wrapper.classList.toggle("is-selected", selected);
          wrapper.style.setProperty("--bearing", `${displayState.bearing_deg}deg`);
        }
        const trailStart = Math.max(0, pair.lower - 24);
        const trailCoordinates = frames.slice(trailStart, pair.lower + 1).map(frame => {
          const point = frame.vessels[vessel.id];
          return [point.lat, point.lon];
        });
        trailCoordinates.push([displayState.lat, displayState.lon]);
        vesselTrails[vessel.id].setLatLngs(trailCoordinates);
        vesselTrails[vessel.id].setStyle({
          weight: selected ? 3.4 : 1.8,
          opacity: selected ? .82 : .36
        });
      });
      return {
        current: currentFrame.vessels[selectedVessel],
        next: nextFrame.vessels[selectedVessel],
        displayLat: interpolate(currentFrame.vessels[selectedVessel].lat, nextFrame.vessels[selectedVessel].lat, pair.fraction),
        displayLon: interpolate(currentFrame.vessels[selectedVessel].lon, nextFrame.vessels[selectedVessel].lon, pair.fraction)
      };
    }

    function setDetailMetric(index, label, value) {
      document.getElementById(`detailMetric${index}Label`).textContent = label;
      document.getElementById(`detailMetric${index}Value`).textContent = value;
    }

    function setDetailProgress(prefix, label, value, fraction, visible = true) {
      document.getElementById(`${prefix}ProgressBlock`).classList.toggle("is-hidden", !visible);
      document.getElementById(`${prefix}ProgressLabel`).textContent = label;
      document.getElementById(`${prefix}ProgressValue`).textContent = value;
      document.getElementById(`${prefix}Progress`).style.width = `${Math.max(0, Math.min(1, Number(fraction || 0))) * 100}%`;
    }

    function updateDetail(frame, vesselDisplay, pair) {
      const component = componentById[selectedComponent];
      const vesselState = vesselDisplay.current;
      let color = "#91a9b5";
      let stateLabel = "Operational";
      let contextLabel = "Physical network component";
      let latitude = Number(component?.lat || 0);
      let longitude = Number(component?.lon || 0);

      if (vesselById[selectedComponent]) {
        const vessel = vesselById[selectedComponent];
        const current = vesselDisplay.current;
        const next = vesselDisplay.next;
        const cargo = interpolate(current.inventory_t, next.inventory_t, pair.fraction);
        const cargoFraction = interpolate(current.fill_fraction, next.fill_fraction, pair.fraction);
        const legProgress = current.operational_state.startsWith("sailing")
          ? interpolate(current.progress, next.progress, pair.fraction)
          : 1;
        color = vessel.color;
        stateLabel = friendly(current.operational_state);
        contextLabel = `${friendly(current.origin)} → ${friendly(current.destination)}`;
        latitude = vesselDisplay.displayLat;
        longitude = vesselDisplay.displayLon;
        setDetailMetric(1, "Cargo", `${formatTonnes(cargo)} / ${formatTonnes(vessel.capacity_t)}`);
        setDetailMetric(2, "Weather factor", `${Number(frame.weather_speed_factor).toFixed(2)}×`);
        setDetailProgress("primary", "Cargo utilisation", `${Math.round(cargoFraction * 100)}%`, cargoFraction);
        setDetailProgress(
          "secondary",
          "Current leg",
          current.operational_state.startsWith("sailing") ? `${Math.round(legProgress * 100)}%` : friendly(current.operational_state),
          legProgress
        );
      } else if (emitterById[selectedComponent]) {
        const emitter = emitterById[selectedComponent];
        const state = frame.emitters[selectedComponent];
        color = emitter.color;
        stateLabel = state.capture_outage
          ? "Capture outage"
          : state.capture_high_output ? "High capture output" : "Capture available";
        contextLabel = "Capture site · CO₂ buffer";
        setDetailMetric(1, "Buffer", `${formatTonnes(state.inventory_t)} / ${formatTonnes(emitter.capacity_t)}`);
        setDetailMetric(2, "Capture availability", `${Math.round(state.capture_availability * 100)}%`);
        setDetailProgress("primary", "Buffer utilisation", `${Math.round(state.fill_fraction * 100)}%`, state.fill_fraction);
        setDetailProgress("secondary", "Capture availability", `${Math.round(state.capture_availability * 100)}%`, state.capture_availability);
      } else if (selectedComponent === terminal.id) {
        color = infrastructureColors.terminal;
        stateLabel = "Receiving terminal";
        contextLabel = "Onshore aggregation · offshore export";
        setDetailMetric(1, "Inventory", `${formatTonnes(frame.terminal_inventory_t)} / ${formatTonnes(terminal.capacity_t)}`);
        setDetailMetric(2, "Storage capacity", formatTonnes(terminal.capacity_t));
        setDetailProgress("primary", "Storage utilisation", `${Math.round(frame.terminal_fill_fraction * 100)}%`, frame.terminal_fill_fraction);
        setDetailProgress("secondary", "", "", 0, false);
      } else if (component?.type === "well") {
        const available = frame.well_available[selectedComponent] !== false;
        color = infrastructureColors.well;
        stateLabel = available ? "Injection online" : "Injection unavailable";
        contextLabel = "Aurora storage complex · injection well";
        setDetailMetric(1, "Availability", available ? "100%" : "0%");
        setDetailMetric(2, "Component type", "Injection well");
        setDetailProgress("primary", "Operational availability", available ? "100%" : "0%", available ? 1 : 0);
        setDetailProgress("secondary", "", "", 0, false);
      } else {
        const offshoreSegment = (data.map.pipeline_segments || []).find(segment => segment.style !== "subsea_connection");
        const descriptions = {
          pipeline: "Offshore CO₂ transport",
          manifold: "Subsea distribution hub",
          reservoir: "Permanent geological storage"
        };
        color = infrastructureColors[component?.type] || "#91a9b5";
        stateLabel = "Infrastructure active";
        contextLabel = descriptions[component?.type] || "Transport and storage";
        setDetailMetric(
          1,
          component?.type === "pipeline" ? "Route length" : "Component type",
          component?.type === "pipeline" && offshoreSegment
            ? `${Number(offshoreSegment.length_km || 0).toFixed(1)} km`
            : friendly(component?.type || "infrastructure")
        );
        setDetailMetric(2, "Data view", "System context");
        setDetailProgress("primary", "", "", 0, false);
        setDetailProgress("secondary", "", "", 0, false);
      }

      document.querySelector(".detail-panel").style.setProperty("--selected-color", color);
      document.getElementById("detailName").textContent = component?.label || friendly(selectedComponent);
      document.getElementById("detailState").textContent = stateLabel;
      document.getElementById("detailRoute").textContent = contextLabel;
      document.getElementById("detailLat").textContent = formatCoordinate(latitude, "lat", 3);
      document.getElementById("detailLon").textContent = formatCoordinate(longitude, "lon", 3);
      updateRouteHighlights(vesselState);
    }

    function updateEventPill(hour, frame) {
      const recent = [...events].reverse().find(event => event.hour <= hour && hour - event.hour <= 5);
      const label = recent
        ? `H${String(Math.round(recent.hour)).padStart(3, "0")} · ${recent.label}`
        : frame.cumulative_vent_t <= 1e-9
          ? "Zero venting maintained in this representative trajectory"
          : `Cumulative vent ${formatTonnes(frame.cumulative_vent_t)}`;
      document.getElementById("eventPill").textContent = label;
    }

    function render(hour) {
      currentHour = Math.max(0, Math.min(duration, Number(hour)));
      const pair = framePair(currentHour);
      const frame = frames[pair.lower];
      const next = frames[pair.upper];
      const cumulativeVent = interpolate(frame.cumulative_vent_t, next.cumulative_vent_t, pair.fraction);
      const transit = interpolate(frame.total_vessel_inventory_t, next.total_vessel_inventory_t, pair.fraction);
      const terminalInventory = interpolate(frame.terminal_inventory_t, next.terminal_inventory_t, pair.fraction);
      timeline.value = String(currentHour);
      const day = Math.floor(currentHour / 24);
      const hourOfDay = Math.floor(currentHour % 24);
      document.getElementById("clockMain").textContent = `Day ${String(day).padStart(2, "0")} · ${String(hourOfDay).padStart(2, "0")}:00`;
      document.getElementById("clockSub").textContent = `Hour ${String(Math.floor(currentHour)).padStart(3, "0")} / ${duration}`;
      const ventValue = document.getElementById("ventValue");
      ventValue.textContent = formatTonnes(cumulativeVent);
      ventValue.classList.toggle("is-zero", cumulativeVent <= 1e-9);
      ventValue.classList.toggle("is-alert", cumulativeVent > 1e-9);
      document.getElementById("transitValue").textContent = formatTonnes(transit);
      document.getElementById("terminalValue").textContent = formatTonnes(terminalInventory);
      updateFacilities(frame);
      updateFacilitySelection();
      renderComponents(frame);
      const vesselDisplay = updateVessels(pair);
      updateDetail(frame, vesselDisplay, pair);
      updateEventPill(currentHour, frame);
      renderVesselTabs();
      drawChart();
    }

    const systemChartSeries = [
      {id: "source-buffers", label: "Source buffers", color: () => themeColor("#a76500", "#f4b942"), value: frame => frame.total_emitter_inventory_t},
      {id: "vessel-cargo", label: "Vessel cargo", color: () => themeColor("#4f6fc8", "#7f9cf5"), value: frame => frame.total_vessel_inventory_t},
      {id: "terminal", label: "Terminal", color: () => themeColor("#278b65", "#72c7a0"), value: frame => frame.terminal_inventory_t},
      {id: "cumulative-vent", label: "Cumulative vent", color: () => themeColor("#cf5e53", "#ef7d70"), value: frame => frame.cumulative_vent_t}
    ];
    const systemChartMaximum = Math.max(
      1,
      ...frames.flatMap(frame => systemChartSeries.map(series => Number(series.value(frame) || 0)))
    );

    function niceScaleMaximum(value) {
      const target = Math.max(1, Number(value));
      const magnitude = 10 ** Math.floor(Math.log10(target));
      const normalized = target / magnitude;
      const nice = normalized <= 1
        ? 1
        : normalized <= 2
          ? 2
          : normalized <= 2.5
            ? 2.5
            : normalized <= 5
              ? 5
              : 10;
      return nice * magnitude;
    }

    function inventoryScaleMaximum(capacity, value) {
      const observedMaximum = Math.max(
        0,
        ...frames.map(frame => Number(value(frame) || 0))
      );
      if (observedMaximum > Number(capacity) + 1e-6) {
        return niceScaleMaximum(observedMaximum * 1.08);
      }
      return Math.max(1, Number(capacity));
    }

    function chartConfigForSelection() {
      if (emitterById[selectedComponent]) {
        const emitter = emitterById[selectedComponent];
        const inventoryValue = frame => frame.emitters[selectedComponent].inventory_t;
        return {
          key: `emitter:${selectedComponent}`,
          title: `${emitter.label} · buffer inventory`,
          format: formatTonnes,
          maximum: inventoryScaleMaximum(emitter.capacity_t, inventoryValue),
          series: [
            {id: "inventory", label: "Buffer inventory", color: emitter.color, value: inventoryValue},
            {id: "capacity", label: "Buffer capacity", color: () => themeColor("#667f86", "#91a9b5"), dash: [5, 5], value: () => emitter.capacity_t}
          ]
        };
      }
      if (vesselById[selectedComponent]) {
        const vessel = vesselById[selectedComponent];
        const inventoryValue = frame => frame.vessels[selectedComponent].inventory_t;
        return {
          key: `vessel:${selectedComponent}`,
          title: `${vessel.label} · cargo inventory`,
          format: formatTonnes,
          maximum: inventoryScaleMaximum(vessel.capacity_t, inventoryValue),
          series: [
            {id: "cargo", label: "Cargo inventory", color: vessel.color, value: inventoryValue},
            {id: "capacity", label: "Vessel capacity", color: () => themeColor("#667f86", "#91a9b5"), dash: [5, 5], value: () => vessel.capacity_t}
          ]
        };
      }
      if (selectedComponent === terminal.id) {
        const inventoryValue = frame => frame.terminal_inventory_t;
        return {
          key: `terminal:${selectedComponent}`,
          title: `${terminal.label} · terminal inventory`,
          format: formatTonnes,
          maximum: inventoryScaleMaximum(terminal.capacity_t, inventoryValue),
          series: [
            {id: "inventory", label: "Terminal inventory", color: infrastructureColors.terminal, value: inventoryValue},
            {id: "capacity", label: "Storage capacity", color: () => themeColor("#667f86", "#91a9b5"), dash: [5, 5], value: () => terminal.capacity_t}
          ]
        };
      }
      const component = componentById[selectedComponent];
      if (component?.type === "well") {
        return {
          key: `well:${selectedComponent}`,
          title: `${component.label} · operational availability`,
          format: value => `${Math.round(value)}%`,
          maximum: 100,
          series: [
            {
              id: "availability",
              label: "Well availability",
              color: infrastructureColors.well,
              value: frame => frame.well_available[selectedComponent] === false ? 0 : 100
            }
          ]
        };
      }
      return {
        key: `system:${selectedComponent}`,
        title: `${component?.label || "System"} · connected system context`,
        format: formatTonnes,
        maximum: systemChartMaximum,
        series: systemChartSeries
      };
    }

    function updateChartHeader(config) {
      if (activeChartKey === config.key) return;
      activeChartKey = config.key;
      document.getElementById("chartTitle").textContent = `${config.title} · ${config.format === formatTonnes ? "tonnes" : "percent"}`;
      chart.setAttribute("aria-label", `${config.title} over the 720-hour trajectory`);
      const legend = document.getElementById("chartLegend");
      legend.innerHTML = "";
      config.series.forEach(series => {
        const item = document.createElement("span");
        item.className = "chart-legend-item";
        const swatch = document.createElement("span");
        swatch.className = "chart-swatch";
        const color = seriesColor(series);
        swatch.style.setProperty("--series-color", color);
        if (series.dash) {
          swatch.style.background = `repeating-linear-gradient(90deg, ${color} 0 5px, transparent 5px 9px)`;
        }
        const label = document.createElement("span");
        label.textContent = series.label;
        item.append(swatch, label);
        legend.appendChild(item);
      });
    }

    function resizeChart() {
      const ratio = window.devicePixelRatio || 1;
      const rect = chart.getBoundingClientRect();
      const width = Math.max(420, Math.floor(rect.width));
      const height = Math.max(150, Math.floor(rect.height));
      chart.width = Math.floor(width * ratio);
      chart.height = Math.floor(height * ratio);
      chartContext.setTransform(ratio, 0, 0, ratio, 0, 0);
      return {width, height};
    }

    function chartX(hour, plot) {
      return plot.left + (Number(hour) / duration) * (plot.right - plot.left);
    }

    function chartY(value, plot, maximum) {
      return plot.bottom - (Number(value) / maximum) * (plot.bottom - plot.top);
    }

    function drawSeries(series, plot, maximum) {
      chartContext.save();
      chartContext.beginPath();
      chartContext.rect(plot.left, plot.top, plot.right - plot.left, plot.bottom - plot.top);
      chartContext.clip();
      chartContext.beginPath();
      frames.forEach((frame, index) => {
        const x = chartX(frame.hour, plot);
        const y = chartY(series.value(frame), plot, maximum);
        if (index === 0) chartContext.moveTo(x, y); else chartContext.lineTo(x, y);
      });
      chartContext.strokeStyle = seriesColor(series);
      chartContext.globalAlpha = series.id === "cumulative-vent" ? .95 : .82;
      chartContext.lineWidth = series.id === "cumulative-vent" ? 2.2 : 1.7;
      chartContext.setLineDash(series.dash || []);
      chartContext.stroke();
      chartContext.setLineDash([]);
      chartContext.globalAlpha = 1;
      chartContext.restore();
    }

    function drawChart() {
      const config = chartConfigForSelection();
      updateChartHeader(config);
      const {width, height} = resizeChart();
      const plot = {left: 58, top: 12, right: width - 18, bottom: height - 27};
      chartContext.clearRect(0, 0, width, height);
      chartContext.font = "9px Manrope";
      chartContext.textBaseline = "middle";
      for (let index = 0; index <= 4; index += 1) {
        const value = config.maximum * index / 4;
        const y = chartY(value, plot, config.maximum);
        chartContext.beginPath();
        chartContext.moveTo(plot.left, y);
        chartContext.lineTo(plot.right, y);
        chartContext.strokeStyle = themeColor("rgba(24,70,79,.12)", "rgba(151,190,203,.12)");
        chartContext.lineWidth = 1;
        chartContext.stroke();
        chartContext.fillStyle = themeColor("#607a81", "#77919c");
        chartContext.textAlign = "right";
        chartContext.fillText(config.format(value), plot.left - 8, y);
      }
      events.filter(event => ["vent", "outage", "vessel"].includes(event.type)).forEach(event => {
        const x = chartX(event.hour, plot);
        chartContext.beginPath();
        chartContext.moveTo(x, plot.top);
        chartContext.lineTo(x, plot.top + (event.type === "vent" ? 10 : 5));
        chartContext.strokeStyle = event.type === "vent"
          ? themeColor("#cf5e53", "#ef7d70")
          : themeColor("rgba(67,96,102,.38)", "rgba(145,169,181,.38)");
        chartContext.lineWidth = event.type === "vent" ? 1.8 : 1;
        chartContext.stroke();
      });
      config.series.forEach(series => drawSeries(series, plot, config.maximum));
      const cursorHour = hoveredHour === null ? currentHour : hoveredHour;
      const cursorX = chartX(cursorHour, plot);
      chartContext.beginPath();
      chartContext.moveTo(cursorX, plot.top);
      chartContext.lineTo(cursorX, plot.bottom);
      chartContext.strokeStyle = hoveredHour === null
        ? themeColor("#a76500", "#f4b942")
        : themeColor("#315a63", "#d8e7eb");
      chartContext.lineWidth = 1.3;
      chartContext.stroke();
      if (cursorHour > 1 && cursorHour < duration - 1) {
        chartContext.fillStyle = themeColor("#315a63", "#c9dbe0");
        chartContext.textAlign = "center";
        chartContext.fillText(`H${Math.round(cursorHour)}`, cursorX, plot.bottom + 16);
      }
      chartContext.fillStyle = themeColor("#607a81", "#77919c");
      chartContext.textAlign = "left";
      chartContext.fillText("H0", plot.left, plot.bottom + 16);
      chartContext.textAlign = "right";
      chartContext.fillText(`H${duration}`, plot.right, plot.bottom + 16);
    }

    function applyTheme(theme, persist = true) {
      const normalized = theme === "dark" ? "dark" : "light-v2";
      document.documentElement.dataset.theme = normalized;
      lightThemeStyles.media = normalized === "dark" ? "not all" : "all";
      themeChoices.forEach(button => {
        button.setAttribute(
          "aria-pressed",
          String(button.dataset.themeChoice === normalized)
        );
      });
      baseTileLayer.setUrl(normalized === "dark" ? darkTileUrl : lightTileUrl);
      Object.values(routeLayers).forEach(layer => {
        layer.setStyle({color: themeColor("#5d7d83", "#456d78")});
      });
      pipelineLayers.forEach(({layer, isSubsea}) => {
        layer.setStyle({
          color: isSubsea
            ? themeColor("#4f6fc8", "#7f9cf5")
            : themeColor("#087f7b", "#64c7c4")
        });
      });
      injectionLayers.forEach(layer => {
        layer.setStyle({color: themeColor("#278b65", "#72c7a0")});
      });
      activeChartKey = null;
      drawLatLonGrid();
      render(currentHour);
      if (persist) {
        try {
          localStorage.setItem("ccs-rl-dashboard-theme", normalized);
        } catch (_error) {
          // Theme still applies for the current view when storage is unavailable.
        }
      }
    }

    function stopPlayback() {
      isPlaying = false;
      previousTimestamp = null;
      if (animationFrame !== null) cancelAnimationFrame(animationFrame);
      animationFrame = null;
      playPause.classList.remove("is-playing");
      playPause.setAttribute("aria-label", "Play trajectory");
      playLabel.textContent = "Play";
    }

    function playbackTick(timestamp) {
      if (!isPlaying) return;
      if (previousTimestamp === null) previousTimestamp = timestamp;
      const elapsedSeconds = Math.min(.1, (timestamp - previousTimestamp) / 1000);
      previousTimestamp = timestamp;
      currentHour += elapsedSeconds * hoursPerSecond;
      if (currentHour >= duration) {
        currentHour = duration;
        render(currentHour);
        stopPlayback();
        return;
      }
      render(currentHour);
      animationFrame = requestAnimationFrame(playbackTick);
    }

    playPause.addEventListener("click", () => {
      if (isPlaying) {
        stopPlayback();
        return;
      }
      if (currentHour >= duration) currentHour = 0;
      isPlaying = true;
      playPause.classList.add("is-playing");
      playPause.setAttribute("aria-label", "Pause trajectory");
      playLabel.textContent = "Pause";
      animationFrame = requestAnimationFrame(playbackTick);
    });
    timeline.max = String(duration);
    themeChoices.forEach(button => {
      button.addEventListener("click", () => {
        applyTheme(button.dataset.themeChoice);
      });
    });
    window.addEventListener("storage", event => {
      if (
        event.key === "ccs-rl-dashboard-theme"
        && (event.newValue === "light-v2" || event.newValue === "dark")
      ) {
        applyTheme(event.newValue, false);
      }
    });
    window.addEventListener("message", event => {
      if (
        event.data?.type === "ccs-rl-theme"
        && (event.data.theme === "light-v2" || event.data.theme === "dark")
      ) {
        applyTheme(event.data.theme, false);
      }
    });
    componentsToggle.addEventListener("click", () => {
      setComponentsCollapsed(!componentsPanel.classList.contains("is-collapsed"));
    });
    map.on("moveend zoomend", drawLatLonGrid);
    map.on("mousemove", event => {
      coordinateReadout.textContent = `${formatCoordinate(event.latlng.lat, "lat")}  ·  ${formatCoordinate(event.latlng.lng, "lon")}`;
    });
    map.getContainer().addEventListener("mouseleave", () => {
      coordinateReadout.textContent = "Move cursor for coordinates";
    });
    timeline.addEventListener("input", event => {
      stopPlayback();
      render(Number(event.target.value));
    });
    document.querySelectorAll(".speed-button").forEach(button => {
      button.addEventListener("click", () => {
        hoursPerSecond = Number(button.dataset.speed);
        document.querySelectorAll(".speed-button").forEach(peer => peer.setAttribute("aria-pressed", String(peer === button)));
      });
    });
    chart.addEventListener("mousemove", event => {
      const rect = chart.getBoundingClientRect();
      const plotLeft = 58;
      const plotRight = rect.width - 18;
      const x = Math.max(plotLeft, Math.min(plotRight, event.clientX - rect.left));
      hoveredHour = ((x - plotLeft) / (plotRight - plotLeft)) * duration;
      drawChart();
    });
    chart.addEventListener("mouseleave", () => {
      hoveredHour = null;
      drawChart();
    });
    chart.addEventListener("click", event => {
      const rect = chart.getBoundingClientRect();
      const plotLeft = 58;
      const plotRight = rect.width - 18;
      const x = Math.max(plotLeft, Math.min(plotRight, event.clientX - rect.left));
      stopPlayback();
      render(((x - plotLeft) / (plotRight - plotLeft)) * duration);
    });
    window.addEventListener("resize", () => {
      map.invalidateSize();
      drawLatLonGrid();
      drawChart();
    });

    if (window.matchMedia("(max-width: 980px)").matches) {
      setComponentsCollapsed(true);
    }
    applyTheme(document.documentElement.dataset.theme, false);
    setTimeout(() => {
      map.invalidateSize();
      drawLatLonGrid();
    }, 0);
  </script>
</body>
</html>
"""
    return (
        template.replace("__TITLE__", title)
        .replace("__SUBTITLE__", subtitle)
        .replace("__DATA_JSON__", data_json)
    )
