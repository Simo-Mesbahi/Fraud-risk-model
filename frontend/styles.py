CUSTOM_CSS = """
<style>

/* ==========================================================================
   GLOBAL
   ========================================================================== */

:root {
    --bg-main: #070B14;
    --bg-secondary: #0A1020;
    --surface: rgba(17, 24, 39, 0.82);
    --surface-hover: rgba(25, 34, 52, 0.92);

    --border: rgba(255, 255, 255, 0.08);
    --border-hover: rgba(255, 255, 255, 0.14);

    --text-primary: #F8FAFC;
    --text-secondary: #97A4BA;
    --text-muted: #66738A;

    --primary: #4C8DFF;
    --secondary: #8B5CF6;

    --success: #61E7A6;
    --warning: #FFD166;
    --high: #FF8A5B;
    --critical: #FF5C7A;

    --radius-sm: 10px;
    --radius-md: 16px;
    --radius-lg: 22px;
}


/* ==========================================================================
   APP BACKGROUND
   ========================================================================== */

html,
body,
[class*="css"] {
    font-family:
        Inter,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;
}


.stApp {
    background:
        radial-gradient(
            circle at 85% 0%,
            rgba(139, 92, 246, 0.12),
            transparent 34%
        ),
        radial-gradient(
            circle at 15% 5%,
            rgba(76, 141, 255, 0.10),
            transparent 30%
        ),
        linear-gradient(
            180deg,
            #070B14 0%,
            #080D18 55%,
            #060A12 100%
        );

    color: var(--text-primary);
}


.block-container {
    max-width: 1500px;
    padding-top: 2rem;
    padding-bottom: 4rem;
    padding-left: 3.2rem;
    padding-right: 3.2rem;
}


/* ==========================================================================
   REMOVE STREAMLIT NATIVE NAVIGATION
   ========================================================================== */

[data-testid="stSidebarNav"] {
    display: none !important;
}

[data-testid="stSidebarNavItems"] {
    display: none !important;
}

[data-testid="stSidebarNavSeparator"] {
    display: none !important;
}


/* Optional native UI cleanup */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}


/* ==========================================================================
   SIDEBAR
   ========================================================================== */

[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #080D18 0%,
            #070B14 100%
        );

    border-right:
        1px solid rgba(255,255,255,0.065);
}


[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.7rem;
}


[data-testid="stSidebar"] * {
    color: #E8EEF8;
}


[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.08);
}


/* ==========================================================================
   RADIO NAVIGATION
   ========================================================================== */

[data-testid="stRadio"] label {
    padding:
        0.36rem
        0.20rem;

    border-radius:
        10px;

    transition:
        all
        0.18s
        ease;
}


[data-testid="stRadio"] label:hover {
    background:
        rgba(
            255,
            255,
            255,
            0.035
        );
}


[data-testid="stRadio"] p {
    font-size:
        0.94rem;

    font-weight:
        550;
}


/* ==========================================================================
   HEADER
   ========================================================================== */

.dashboard-title {
    margin: 0;

    font-size:
        clamp(
            2.2rem,
            4vw,
            3.35rem
        );

    line-height:
        1.05;

    font-weight:
        850;

    letter-spacing:
        -0.045em;

    background:
        linear-gradient(
            90deg,
            #FFFFFF 0%,
            #D8EAFF 48%,
            #A8D4FF 100%
        );

    -webkit-background-clip:
        text;

    -webkit-text-fill-color:
        transparent;
}


.dashboard-subtitle {
    margin-top:
        0.8rem;

    margin-bottom:
        2.8rem;

    color:
        var(--text-secondary);

    font-size:
        1rem;

    line-height:
        1.6;
}


/* ==========================================================================
   TYPOGRAPHY
   ========================================================================== */

h1,
h2,
h3,
h4 {
    letter-spacing:
        -0.025em;

    color:
        var(--text-primary);
}


h3 {
    margin-bottom:
        0.25rem;
}


p {
    line-height:
        1.6;
}


/* ==========================================================================
   GLASS CARDS
   ========================================================================== */

.glass-card {
    position:
        relative;

    min-height:
        132px;

    padding:
        1.35rem
        1.45rem;

    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,0.065),
            rgba(255,255,255,0.022)
        );

    border:
        1px solid
        var(--border);

    border-radius:
        var(--radius-lg);

    box-shadow:
        0 18px 45px
        rgba(0,0,0,0.23);

    backdrop-filter:
        blur(14px);

    transition:
        transform
        0.18s
        ease,
        border-color
        0.18s
        ease,
        background
        0.18s
        ease;
}


.glass-card:hover {
    transform:
        translateY(-2px);

    border-color:
        var(--border-hover);

    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,0.078),
            rgba(255,255,255,0.028)
        );
}


/* ==========================================================================
   METRIC CARDS
   ========================================================================== */

.metric-label {
    color:
        #8997AE;

    font-size:
        0.76rem;

    font-weight:
        700;

    text-transform:
        uppercase;

    letter-spacing:
        0.105em;
}


.metric-value {
    margin-top:
        0.55rem;

    color:
        #FFFFFF;

    font-size:
        2rem;

    font-weight:
        850;

    letter-spacing:
        -0.035em;
}


.metric-helper {
    margin-top:
        0.55rem;

    color:
        #75839A;

    font-size:
        0.79rem;

    line-height:
        1.4;
}


/* ==========================================================================
   STATUS BADGES
   ========================================================================== */

.status-ok {
    display:
        inline-flex;

    align-items:
        center;

    gap:
        0.3rem;

    padding:
        0.42rem
        0.72rem;

    border:
        1px solid
        rgba(
            97,
            231,
            166,
            0.36
        );

    border-radius:
        999px;

    background:
        rgba(
            97,
            231,
            166,
            0.10
        );

    color:
        var(--success);

    font-size:
        0.76rem;

    font-weight:
        800;

    letter-spacing:
        0.03em;
}


.status-offline {
    display:
        inline-flex;

    padding:
        0.42rem
        0.72rem;

    border:
        1px solid
        rgba(
            255,
            92,
            122,
            0.35
        );

    border-radius:
        999px;

    background:
        rgba(
            255,
            92,
            122,
            0.10
        );

    color:
        var(--critical);

    font-size:
        0.76rem;

    font-weight:
        800;
}


/* ==========================================================================
   EMPTY STATES
   ========================================================================== */

.empty-state {
    padding:
        3.2rem
        2rem;

    text-align:
        center;

    border:
        1px dashed
        rgba(
            255,
            255,
            255,
            0.12
        );

    border-radius:
        var(--radius-lg);

    background:
        rgba(
            255,
            255,
            255,
            0.018
        );
}


.empty-title {
    color:
        #F5F8FC;

    font-size:
        1.15rem;

    font-weight:
        750;
}


.empty-message {
    max-width:
        600px;

    margin:
        0.5rem auto 0;

    color:
        var(--text-secondary);

    font-size:
        0.88rem;
}


/* ==========================================================================
   STREAMLIT METRIC
   ========================================================================== */

div[data-testid="stMetric"] {
    padding:
        1rem
        1.1rem;

    border:
        1px solid
        var(--border);

    border-radius:
        var(--radius-md);

    background:
        rgba(
            255,
            255,
            255,
            0.025
        );
}


/* ==========================================================================
   BUTTONS
   ========================================================================== */

.stButton > button,
.stDownloadButton > button {
    min-height:
        46px;

    border:
        1px solid
        rgba(
            100,
            160,
            255,
            0.30
        );

    border-radius:
        12px;

    background:
        linear-gradient(
            135deg,
            #286CF4 0%,
            #6E55EF 100%
        );

    color:
        #FFFFFF;

    font-weight:
        750;

    transition:
        all
        0.18s
        ease;

    box-shadow:
        0 8px 24px
        rgba(
            50,
            100,
            240,
            0.13
        );
}


.stButton > button:hover,
.stDownloadButton > button:hover {
    transform:
        translateY(-1px);

    border-color:
        rgba(
            255,
            255,
            255,
            0.42
        );

    box-shadow:
        0 10px 30px
        rgba(
            70,
            100,
            255,
            0.20
        );
}


/* ==========================================================================
   INPUTS
   ========================================================================== */

div[data-baseweb="input"] > div,
div[data-baseweb="textarea"] > div,
div[data-baseweb="select"] > div {
    background:
        rgba(
            255,
            255,
            255,
            0.032
        ) !important;

    border-color:
        rgba(
            255,
            255,
            255,
            0.085
        ) !important;

    border-radius:
        12px !important;
}


/* ==========================================================================
   FILE UPLOADER
   ========================================================================== */

[data-testid="stFileUploaderDropzone"] {
    padding:
        2rem;

    background:
        rgba(
            255,
            255,
            255,
            0.025
        );

    border:
        1px dashed
        rgba(
            255,
            255,
            255,
            0.14
        );

    border-radius:
        var(--radius-lg);
}


/* ==========================================================================
   DATAFRAME
   ========================================================================== */

[data-testid="stDataFrame"] {
    border:
        1px solid
        var(--border);

    border-radius:
        var(--radius-md);

    overflow:
        hidden;
}


/* ==========================================================================
   TABS
   ========================================================================== */

button[data-baseweb="tab"] {
    font-weight:
        650;

    color:
        #8E9BB0;
}


button[data-baseweb="tab"][aria-selected="true"] {
    color:
        #FFFFFF;
}


/* ==========================================================================
   ALERTS
   ========================================================================== */

[data-testid="stAlert"] {
    border-radius:
        var(--radius-md);

    border:
        1px solid
        rgba(
            255,
            255,
            255,
            0.08
        );
}


/* ==========================================================================
   EXPANDER
   ========================================================================== */

[data-testid="stExpander"] {
    border:
        1px solid
        var(--border);

    border-radius:
        var(--radius-md);

    overflow:
        hidden;
}


/* ==========================================================================
   DIVIDERS
   ========================================================================== */

hr {
    border:
        none;

    border-top:
        1px solid
        rgba(
            255,
            255,
            255,
            0.07
        );
}


/* ==========================================================================
   RESPONSIVE
   ========================================================================== */

@media (
    max-width:
    900px
) {

    .block-container {
        padding-left:
            1.2rem;

        padding-right:
            1.2rem;
    }

    .dashboard-title {
        font-size:
            2.2rem;
    }

}


/* ==========================================================================
   SCROLLBAR
   ========================================================================== */

::-webkit-scrollbar {
    width:
        8px;

    height:
        8px;
}


::-webkit-scrollbar-track {
    background:
        transparent;
}


::-webkit-scrollbar-thumb {
    background:
        rgba(
            255,
            255,
            255,
            0.12
        );

    border-radius:
        999px;
}


::-webkit-scrollbar-thumb:hover {
    background:
        rgba(
            255,
            255,
            255,
            0.20
        );
}

</style>
"""