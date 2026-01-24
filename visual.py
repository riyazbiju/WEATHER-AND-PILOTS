import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import folium
from streamlit_folium import st_folium
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from datetime import datetime, timedelta, date

# ==========================================
# 0. PAGE CONFIG & STYLING
# ==========================================
st.set_page_config(
    page_title="SKYBRIEF Visualizer",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    /* Dark Theme Setup */
    .stApp { background-color: #0f172a; color: #e2e8f0; }
    h1, h2, h3, h4, h5, h6 { color: #f8fafc !important; font-family: 'Inter', sans-serif; }
    
    /* Inputs */
    .stTextInput > div > div > input, 
    .stNumberInput > div > div > input, 
    .stSelectbox > div > div > div, 
    .stDateInput > div > div > input,
    .stTimeInput > div > div > input {
        background-color: #1e293b; color: white; border: 1px solid #334155; border-radius: 8px;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #0f172a; color: #0ea5e9; border: 1px solid #0ea5e9; border-radius: 8px; font-weight: 600;
        width: 100%;
    }
    .stButton > button:hover { background-color: #0ea5e9; color: white; }
    
    /* Slider & Radio */
    .stSlider > div > div > div > div { background-color: #0ea5e9; }
    .stRadio > div { color: white; }
    
    .block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================

def calculate_time_at_dist(start_time_obj, speed_kts, dist_nm):
    """Calculates the time at a specific distance."""
    if start_time_obj is None: return "N/A"
    hours_flown = dist_nm / speed_kts
    t_current = start_time_obj + timedelta(hours=hours_flown)
    return t_current.strftime("%H:%MZ")

def get_mock_weather_html(title, alt, is_point=False):
    """Generates HTML content for Map Popups."""
    temp = round(-2.0 - (alt/1000)*2 + np.random.randint(-2, 2), 1)
    wind_spd = int(10 + (alt/1000) + np.random.randint(0, 20))
    wind_dir = np.random.randint(240, 300)
    pres = int(1013 - (alt/30)) if alt < 10000 else int(250)
    vis = "10+ SM" if alt > 5000 else f"{np.random.randint(3,10)} SM"
    
    html = f"""
    <div style="font-family: sans-serif; color: #333; min-width: 180px;">
        <h4 style="margin:0 0 5px 0; border-bottom:1px solid #ccc; padding-bottom:5px;">{title}</h4>
        <div style="font-size: 13px; line-height: 1.4;">
            <b>Altitude:</b> {alt} ft<br>
            <b>Temperature:</b> {temp} °C<br>
            <b>Wind:</b> {wind_spd} kt @ {wind_dir}°<br>
            <b>Visibility:</b> {vis}<br>
            <b>Pressure:</b> {pres} hPa
        </div>
    </div>
    """
    return html

# ==========================================
# 2. VISUALIZATION FUNCTIONS
# ==========================================

@st.cache_data
def create_route_map(start, end, lat1, lon1, lat2, lon2, alt):
    """Top View Map with interactive waypoints."""
    center_lat, center_lon = (lat1 + lat2) / 2, (lon1 + lon2) / 2
    m = folium.Map(location=[center_lat, center_lon], zoom_start=4, tiles="CartoDB dark_matter")

    coords = [
        [lat1, lon1],
        [lat1 + (lat2-lat1)*0.25, lon1 + (lon2-lon1)*0.25], 
        [lat1 + (lat2-lat1)*0.5, lon1 + (lon2-lon1)*0.5], 
        [lat1 + (lat2-lat1)*0.75, lon1 + (lon2-lon1)*0.75], 
        [lat2, lon2]
    ]
    folium.PolyLine(coords, color="#0ea5e9", weight=4, opacity=0.8).add_to(m)

    start_popup = get_mock_weather_html(f"DEP: {start}", alt)
    folium.Marker([lat1, lon1], popup=folium.Popup(start_popup, max_width=250), icon=folium.Icon(color="blue", icon="plane", prefix="fa")).add_to(m)

    end_popup = get_mock_weather_html(f"ARR: {end}", alt)
    folium.Marker([lat2, lon2], popup=folium.Popup(end_popup, max_width=250), icon=folium.Icon(color="green", icon="flag")).add_to(m)

    mid_points = coords[1:-1]
    for i, point in enumerate(mid_points):
        lat, lon = point
        popup_html = get_mock_weather_html(f"Waypoint {i+1}", alt)
        folium.CircleMarker(
            location=[lat, lon], radius=6, color="#facc15", fill=True, fill_color="#facc15", fill_opacity=1.0,
            popup=folium.Popup(popup_html, max_width=250)
        ).add_to(m)

    return m

def plot_side_view_advanced(cruise_alt_ft, view_mode, inspect_dist_nm=None, flight_speed=450, start_datetime=None):
    """Side View with Dual Legends and Bottom Compass Rose."""
    plt.style.use('dark_background')
    
    # Increase height slightly to accommodate the bottom legend comfortably
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8.5), gridspec_kw={'height_ratios': [5, 1.5]})
    fig.patch.set_facecolor('#0f172a') 
    plt.subplots_adjust(hspace=0.3) 
    
    # --- MOCK DATA ---
    max_dist = 800
    dist_km = np.linspace(0, max_dist, 100)
    altitude_ft = np.linspace(0, 50000, 50)
    X, Y = np.meshgrid(dist_km, altitude_ft)
    
    temp_data = 15 - (Y / 1000) * 2 + np.sin(X/200)*2
    wind_speed = 10 + (Y / 1000) * 1.5 + 20 * np.sin(X/300)
    vis_data = 10 + 5 * np.sin(X/50) * np.cos(Y/5000)
    vis_data = np.clip(vis_data, 0, 10) 
    
    path_x = dist_km
    path_y = np.zeros_like(dist_km)
    path_y[:20] = np.linspace(0, cruise_alt_ft, 20)      
    path_y[20:80] = cruise_alt_ft                        
    path_y[80:] = np.linspace(cruise_alt_ft, 0, 20)      
    path_pressure = 1013.25 * (1 - (2.25577e-5 * path_y * 0.3048))**5.25588

    # --- PLOTTING ---
    cbar_bottom = None
    cbar_bottom_label = ""
    
    if view_mode == "Temperature + Wind":
        # 1. Background: Temperature
        c_plot = ax1.contourf(X, Y, temp_data, cmap='RdBu_r', levels=50, alpha=0.8)
        
        # 2. VERTICAL Legend for Temperature
        cbar_temp = plt.colorbar(c_plot, ax=ax1, location='right', pad=0.01, aspect=30)
        cbar_temp.set_label("Temperature (°C)", color='white', fontsize=9)
        cbar_temp.ax.yaxis.set_tick_params(color='white', labelcolor='white')
        
        ax1.set_title("Temperature and Wind", fontsize=13, color='white', fontweight='bold')
        
        # 3. Overlay: Wind Barbs
        skip_x, skip_y = 8, 5
        barb_u = wind_speed * np.sin(X/300)
        barb_v = wind_speed * np.cos(X/300)
        ax1.barbs(X[::skip_y, ::skip_x], Y[::skip_y, ::skip_x], 
                  barb_u[::skip_y, ::skip_x], barb_v[::skip_y, ::skip_x], 
                  length=5, color='white', alpha=0.6, pivot='middle')
        
        ax1.plot(path_x, path_y, color='#0ea5e9', linewidth=3, zorder=10)

        # 4. WIND LEGEND (In ax2 - Below the Graph)
        ax2.set_xlim(0, 100)
        ax2.set_ylim(0, 1)
        ax2.axis('off')

        # Draw a subtle box around the legend area
        rect = patches.Rectangle((5, 0.05), 90, 0.9, linewidth=1, edgecolor='#334155', facecolor='#1e293b', alpha=0.5, zorder=0)
        ax2.add_patch(rect)
        
        # Title
        ax2.text(50, 0.85, "Wind Legend", color='white', fontsize=11, fontweight='bold', ha='center', zorder=1)

        # --- Section 1: Speed (Left) ---
        # 10kt
        ax2.barbs([20], [0.5], [10], [0], length=6, color='white', pivot='middle', sizes=dict(emptybarb=0.0), zorder=1)
        ax2.text(23, 0.5, "10 kt", color='white', fontsize=10, va='center', zorder=1)
        # 50kt
        ax2.barbs([35], [0.5], [50], [0], length=6, color='white', pivot='middle', sizes=dict(emptybarb=0.0), zorder=1)
        ax2.text(38, 0.5, "50 kt", color='white', fontsize=10, va='center', zorder=1)

        # --- Section 2: Compass Rose (Right) ---
        cx, cy = 70, 0.45 # Center position
        
        # Crosshairs
        ax2.plot([cx, cx], [cy-0.25, cy+0.25], color='#94a3b8', linewidth=1, zorder=1) # N-S
        ax2.plot([cx-3, cx+3], [cy, cy], color='#94a3b8', linewidth=1, zorder=1)     # E-W
        
        # Directions
        ax2.text(cx, cy+0.3, "N 000°", color='#0ea5e9', fontsize=9, ha='center', va='bottom', fontweight='bold', zorder=1)
        ax2.text(cx, cy-0.3, "S 180°", color='#0ea5e9', fontsize=9, ha='center', va='top', fontweight='bold', zorder=1)
        ax2.text(cx+4, cy, "E 090°", color='#0ea5e9', fontsize=9, ha='left', va='center', fontweight='bold', zorder=1)
        ax2.text(cx-4, cy, "W 270°", color='#0ea5e9', fontsize=9, ha='right', va='center', fontweight='bold', zorder=1)
        
        # Diagonals
        ax2.text(cx+3, cy+0.2, "NE 045°", color='white', fontsize=8, zorder=1)
        ax2.text(cx+3, cy-0.2, "SE 135°", color='white', fontsize=8, zorder=1)
        ax2.text(cx-3, cy-0.2, "SW 225°", color='white', fontsize=8, ha='right', zorder=1)
        ax2.text(cx-3, cy+0.2, "NW 315°", color='white', fontsize=8, ha='right', zorder=1)

    elif view_mode == "Visibility + Pressure":
        # Background: Visibility
        vis_plot = ax1.contourf(X, Y, vis_data, cmap='bone', levels=20, alpha=0.7)
        cbar_vis = plt.colorbar(vis_plot, ax=ax1, location='right', pad=0.01, aspect=30)
        cbar_vis.set_label("Visibility (SM)", color='white', fontsize=9)
        cbar_vis.ax.yaxis.set_tick_params(color='white', labelcolor='white')
        
        ax1.set_title("Visibility and Pressure", fontsize=13, color='white', fontweight='bold')
        
        # Line: Pressure
        points = np.array([path_x, path_y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(path_pressure.min(), path_pressure.max())
        lc = LineCollection(segments, cmap='plasma', norm=norm)
        lc.set_array(path_pressure)
        lc.set_linewidth(5)
        ax1.add_collection(lc)
        
        cbar_bottom = lc
        cbar_bottom_label = "Pressure along Route (hPa)"

        # Use ax2 for bottom legend (Pressure)
        ax2.set_xlim(0, max_dist)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        if cbar_bottom:
            cbar = plt.colorbar(cbar_bottom, ax=[ax1, ax2], location='bottom', pad=0.12, fraction=0.05, aspect=40)
            cbar.ax.xaxis.set_tick_params(color='white', labelcolor='white')
            cbar.set_label(cbar_bottom_label, color='white', fontsize=10, fontweight='bold')

    # Styling
    terrain = 1000 + 2000 * np.sin(dist_km/60)**2
    ax1.fill_between(dist_km, 0, terrain, color='#22c55e', alpha=0.8, zorder=5)
    ax1.set_ylim(0, 50000)
    ax1.set_ylabel("Altitude (ft)", color='white')
    ax1.set_xlim(0, max_dist)
    ax1.set_xlabel("Distance (NM)", color='#94a3b8', fontweight='bold')
    ax1.tick_params(colors='white')
    ax1.grid(True, linestyle=':', alpha=0.2, color='white')
    ax1.set_facecolor('#1e293b')

    # Interactive Cursor Box
    if inspect_dist_nm is not None:
        ax1.axvline(x=inspect_dist_nm, color='white', linestyle='--', linewidth=1.5, alpha=0.9, zorder=20)
        idx = (np.abs(dist_km - inspect_dist_nm)).argmin()
        curr_alt = path_y[idx]
        
        curr_time_str = calculate_time_at_dist(start_datetime, flight_speed, inspect_dist_nm)
        val_p = path_pressure[idx]
        val_v = vis_data[int(curr_alt/1000), idx]
        
        box_text = (
            f"TIME: {curr_time_str}\n"
            f"DIST: {int(inspect_dist_nm)} NM\n"
            f"ALT:  {int(curr_alt)} FT\n"
            f"--------------\n"
            f"PRES: {int(val_p)} hPa\n"
            f"VIS:  {int(val_v)} SM"
        )
        
        box_x = inspect_dist_nm + 20 if inspect_dist_nm < 400 else inspect_dist_nm - 180
        props = dict(boxstyle='round', facecolor='#0f172a', alpha=0.95, edgecolor='#0ea5e9', linewidth=2)
        ax1.text(box_x, 40000, box_text, fontsize=10, color='#0ea5e9', fontfamily='monospace', fontweight='bold', bbox=props, zorder=25)
        ax1.plot(inspect_dist_nm, curr_alt, 'wo', markersize=8, markeredgecolor='#0ea5e9', zorder=25)

    return fig

@st.cache_data
def create_point_map(lat, lon, alt, date_val, time_val):
    """Map for Unknown Point Analysis."""
    m = folium.Map(location=[lat, lon], zoom_start=8, tiles="CartoDB dark_matter")
    popup_html = get_mock_weather_html("Analysis Point", alt)
    folium.Marker([lat, lon], popup=folium.Popup(popup_html, max_width=250), icon=folium.Icon(color="red", icon="cloud", prefix="fa")).add_to(m)
    return m

# ==========================================
# 3. APP LOGIC
# ==========================================

if 'page' not in st.session_state: st.session_state.page = 'home'

# --- PAGE: HOME ---
if st.session_state.page == 'home':
    st.markdown("<div style='text-align: center; padding: 40px 0;'>", unsafe_allow_html=True)
    st.title("✈️ SKYBRIEF Visualizer")
    st.markdown("</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🗺️ Find Weather Along Route"):
            st.session_state.page = 'route_input'
            st.rerun()
        st.write("")
        if st.button("📍 Find Weather at Unknown Point"):
            st.session_state.page = 'point_input'
            st.rerun()

# --- PAGE: ROUTE INPUT ---
elif st.session_state.page == 'route_input':
    st.markdown("## 🛫 Plan Your Route")
    with st.form("route_form"):
        c1, c2 = st.columns(2)
        start = c1.text_input("Start Station (ICAO)", "CYYZ")
        alt = c1.number_input("Cruise Altitude (ft)", 30000, step=1000)
        date_input = c1.date_input("Date of Flight", date.today())
        
        end = c2.text_input("End Station (ICAO)", "CYYR")
        speed = c2.number_input("Cruise Speed (kts)", 450, step=10)
        time_input = c2.time_input("Departure Time", datetime.now().time())
        
        st.write("")
        if st.form_submit_button("🚀 Analyze Route"):
            dt_obj = datetime.combine(date_input, time_input)
            st.session_state.route_data = {
                'start': start, 'end': end, 'alt': alt, 'speed': speed,
                'datetime': dt_obj,
                'lat1': 43.67, 'lon1': -79.62, 
                'lat2': 53.3, 'lon2': -60.4
            }
            st.session_state.page = 'route_result'
            st.rerun()
    
    if st.button("← Back"): st.session_state.page = 'home'; st.rerun()

# --- PAGE: ROUTE RESULT ---
elif st.session_state.page == 'route_result':
    d = st.session_state.route_data
    time_str = d['datetime'].strftime("%Y-%m-%d %H:%M")
    
    st.markdown(f"""
    <div style="background-color: #1e293b; padding: 15px; border-radius: 10px; border: 1px solid #334155; margin-bottom: 20px;">
        <h3 style="margin:0;">{d['start']} ➝ {d['end']}</h3>
        <code style="color: #94a3b8;">FL{int(d['alt']/100)} | {d['speed']} KTS | {time_str}</code>
    </div>
    """, unsafe_allow_html=True)
    
    view = st.radio("Select View:", ["Top View (Map)", "Side View (Vertical Profile)"], horizontal=True)
    
    if view == "Top View (Map)":
        st.info("💡 Click on the Departure, Arrival, or yellow waypoints to see Temp, Wind, Visibility & Pressure.")
        map_obj = create_route_map(d['start'], d['end'], d['lat1'], d['lon1'], d['lat2'], d['lon2'], d['alt'])
        st_folium(map_obj, height=500, use_container_width=True, returned_objects=[])
        
    else: # SIDE VIEW
        c_ctrl, _ = st.columns([2, 1])
        with c_ctrl:
            combo_mode = st.selectbox("Select Data Layers:", ["Temperature + Wind", "Visibility + Pressure"])
            
        st.write("")
        st.markdown("**🔍 Interactive Flight Inspector** (Slide to view Time & Data)")
        inspect_val = st.slider("Distance (NM)", 0, 800, 0, step=10, label_visibility="collapsed")
        
        fig = plot_side_view_advanced(d['alt'], combo_mode, inspect_val, d['speed'], d['datetime'])
        st.pyplot(fig, use_container_width=True)
        
    st.markdown("---")
    if st.button("↺ New Query"): st.session_state.page = 'home'; st.rerun()

# --- PAGE: POINT INPUT ---
elif st.session_state.page == 'point_input':
    st.markdown("## 📍 Single Point Analysis")
    with st.form("point_form"):
        c1, c2, c3 = st.columns(3)
        lat = c1.number_input("Latitude", 43.65, format="%.4f")
        lon = c2.number_input("Longitude", -79.38, format="%.4f")
        alt = c3.number_input("Altitude (ft)", 5000, step=100)
        
        c4, c5 = st.columns(2)
        p_date = c4.date_input("Date", date.today())
        p_time = c5.time_input("Time", datetime.now().time())
        
        st.write("")
        if st.form_submit_button("🔍 Analyze Location"):
            st.session_state.point_data = {
                'lat': lat, 'lon': lon, 'alt': alt,
                'datetime': datetime.combine(p_date, p_time)
            }
            st.session_state.page = 'point_result'
            st.rerun()

    if st.button("← Back"): st.session_state.page = 'home'; st.rerun()

# --- PAGE: POINT RESULT ---
elif st.session_state.page == 'point_result':
    d = st.session_state.point_data
    
    st.markdown(f"""
    <div style="background-color: #1e293b; padding: 15px; border-radius: 10px; border: 1px solid #334155; margin-bottom: 20px; text-align: center;">
        <h3 style="margin:0;">Analysis Result</h3>
        <code style="color: #0ea5e9;">Lat {d['lat']} / Lon {d['lon']} @ {d['alt']} ft</code><br>
        <small style="color: #94a3b8;">{d['datetime'].strftime("%Y-%m-%d %H:%M")}</small>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("💡 Click the red marker on the map to view detailed weather variables.")
    map_obj = create_point_map(d['lat'], d['lon'], d['alt'], d['datetime'].date(), d['datetime'].time())
    st_folium(map_obj, height=500, use_container_width=True, returned_objects=[])
    
    st.markdown("---")
    if st.button("↺ New Query"): st.session_state.page = 'home'; st.rerun()