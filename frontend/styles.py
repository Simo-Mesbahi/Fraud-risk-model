CUSTOM_CSS = """
<style>

/* ==========================================================================
   DESIGN TOKENS
   ========================================================================== */

:root {
    --bg-main: #060A12;
    --bg-secondary: #080D18;
    --bg-tertiary: #0B1220;

    --surface-01: rgba(255, 255, 255, 0.020);
    --surface-02: rgba(255, 255, 255, 0.032);
    --surface-03: rgba(255, 255, 255, 0.050);
    --surface-hover: rgba(255, 255, 255, 0.065);

    --border-soft: rgba(255, 255, 255, 0.060);
    --border: rgba(255, 255, 255, 0.085);
    --border-strong: rgba(255, 255, 255, 0.140);

    --text-primary: #F8FAFC;
    --text-secondary: #98A6BC;
    --text-muted: #66758D;
    --text-disabled: #46546A;

    --primary: #4C8DFF;
    --primary-soft: rgba(76, 141, 255, 0.12);

    --secondary: #8B5CF6;
    --secondary-soft: rgba(139, 92, 246, 0.12);

    --success: #61E7A6;
    --success-soft: rgba(97, 231, 166, 0.10);

    --warning: #FFD166;
    --warning-soft: rgba(255, 209, 102, 0.10);

    --high: #FF8A5B;
    --high-soft: rgba(255, 138, 91, 0.10);

    --critical: #FF5C7A;
    --critical-soft: rgba(255, 92, 122, 0.10);

    --radius-xs: 8px;
    --radius-sm: 10px;
    --radius-md: 14px;
    --radius-lg: 18px;
    --radius-xl: 22px;

    --shadow-sm:
        0 8px 24px rgba(0, 0, 0, 0.14);

    --shadow-md:
        0 18px 45px rgba(0, 0, 0, 0.22);

    --shadow-lg:
        0 24px 70px rgba(0, 0, 0, 0.28);

    --transition-fast:
        150ms cubic-bezier(.2,.8,.2,1);

    --transition:
        200ms cubic-bezier(.2,.8,.2,1);
}


/* ==========================================================================
   UNIVERSAL SAFETY
   ========================================================================== */

*,
*::before,
*::after {
    box-sizing: border-box;
}


html,
body {
    max-width: 100%;
    overflow-x: hidden;
}


html {
    scroll-behavior: smooth;
}


body {
    margin: 0;

    color: var(--text-primary);

    font-family:
        Inter,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        Roboto,
        Helvetica,
        Arial,
        sans-serif;

    -webkit-font-smoothing: antialiased;
    text-rendering: optimizeLegibility;
}


[class*="css"] {
    font-family:
        Inter,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        Roboto,
        Helvetica,
        Arial,
        sans-serif;
}


/* ==========================================================================
   GLOBAL APP SHELL
   ========================================================================== */

.stApp {
    min-height: 100vh;

    background:
        radial-gradient(
            circle at 88% -5%,
            rgba(139, 92, 246, 0.13),
            transparent 31%
        ),
        radial-gradient(
            circle at 12% 0%,
            rgba(76, 141, 255, 0.11),
            transparent 28%
        ),
        radial-gradient(
            circle at 60% 100%,
            rgba(34, 197, 235, 0.035),
            transparent 34%
        ),
        linear-gradient(
            180deg,
            #070B14 0%,
            #080D18 48%,
            #060A12 100%
        );

    color: var(--text-primary);
}


.block-container {
    width: 100%;
    max-width: 1540px;

    padding-top: 1.7rem;
    padding-bottom: 4rem;
    padding-left: 3rem;
    padding-right: 3rem;
}


/* ==========================================================================
   STREAMLIT CHROME
   ========================================================================== */

[data-testid="stSidebarNav"],
[data-testid="stSidebarNavItems"],
[data-testid="stSidebarNavSeparator"] {
    display: none !important;
}


#MainMenu {
    visibility: hidden;
}


footer {
    visibility: hidden;
}


header[data-testid="stHeader"] {
    background: transparent;
}


/* ==========================================================================
   GLOBAL LAYOUT SAFETY
   ========================================================================== */

[data-testid="stHorizontalBlock"] {
    width: 100%;
}


[data-testid="stHorizontalBlock"] > div,
[data-testid="stColumn"],
[data-testid="column"] {
    min-width: 0 !important;
}


[data-testid="stColumn"] {
    overflow: visible !important;
}


[data-testid="stColumn"] > div,
[data-testid="stColumn"] [data-testid="stVerticalBlock"],
[data-testid="stColumn"] [data-testid="stVerticalBlockBorderWrapper"] {
    min-width: 0 !important;
    max-width: 100% !important;
}


/* ==========================================================================
   SIDEBAR
   ========================================================================== */

[data-testid="stSidebar"] {
    background:
        radial-gradient(
            circle at 0% 0%,
            rgba(76, 141, 255, 0.055),
            transparent 28%
        ),
        linear-gradient(
            180deg,
            #080D18 0%,
            #070B14 100%
        );

    border-right:
        1px solid var(--border-soft);
}


[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.55rem;
}


[data-testid="stSidebar"] * {
    color: #E8EEF8;
}


[data-testid="stSidebar"] hr {
    margin-top: 1.15rem;
    margin-bottom: 1.15rem;

    border-color: var(--border-soft);
}


[data-testid="stSidebar"] h2 {
    margin-bottom: 0.15rem;

    font-size: 1.25rem;
    font-weight: 800;
    letter-spacing: -0.025em;
}


.sidebar-product-meta {
    color: #59677E;

    font-size: .66rem;
    line-height: 1.65;

    text-transform: uppercase;
    letter-spacing: .085em;
}


/* ==========================================================================
   SIDEBAR NAVIGATION
   ========================================================================== */

[data-testid="stSidebar"] [data-testid="stRadio"] > div {
    gap: .18rem;
}


[data-testid="stSidebar"] [data-testid="stRadio"] label {
    min-height: 42px;

    padding:
        .54rem
        .68rem;

    border:
        1px solid transparent;

    border-radius:
        11px;

    transition:
        background var(--transition-fast),
        border-color var(--transition-fast),
        transform var(--transition-fast);
}


[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    transform: translateX(2px);

    background:
        rgba(255,255,255,.038);

    border-color:
        rgba(255,255,255,.045);
}


[data-testid="stSidebar"] [data-testid="stRadio"] p {
    font-size: .90rem;
    font-weight: 600;

    white-space: normal !important;
}


/* ==========================================================================
   APPLICATION HEADER
   ========================================================================== */

.dashboard-title {
    max-width: 100%;

    margin: 0;

    font-size:
        clamp(
            2.1rem,
            4vw,
            3.45rem
        );

    line-height: 1.02;

    font-weight: 850;

    letter-spacing: -0.052em;

    overflow-wrap: anywhere;

    background:
        linear-gradient(
            92deg,
            #FFFFFF 0%,
            #E6F0FF 40%,
            #BBD9FF 72%,
            #C4B5FD 100%
        );

    -webkit-background-clip: text;
    background-clip: text;

    -webkit-text-fill-color: transparent;
}


.dashboard-subtitle {
    max-width: 800px;

    margin-top: .75rem;
    margin-bottom: 1.3rem;

    color: var(--text-secondary);

    font-size: .98rem;
    line-height: 1.65;

    overflow-wrap: anywhere;
}


.page-context-strip {
    display: inline-flex;

    align-items: center;

    flex-wrap: wrap;

    gap: .52rem;

    max-width: 100%;

    padding:
        .40rem
        .72rem;

    margin-bottom: 1.05rem;

    border:
        1px solid var(--border-soft);

    border-radius: 999px;

    background:
        rgba(255,255,255,.022);

    color: #7F8DA4;

    font-size: .73rem;

    backdrop-filter: blur(10px);

    overflow-wrap: anywhere;
}


.page-context-strip strong {
    color: #DCE7F7;
    font-weight: 750;
}


/* ==========================================================================
   TYPOGRAPHY
   ========================================================================== */

h1,
h2,
h3,
h4,
h5 {
    max-width: 100%;

    color: var(--text-primary);

    letter-spacing: -.025em;

    white-space: normal;

    overflow-wrap: anywhere;
}


h1 {
    font-weight: 850;
}


h2,
h3 {
    font-weight: 780;
}


h4 {
    font-weight: 720;
}


p {
    max-width: 100%;

    line-height: 1.62;

    overflow-wrap: anywhere;
}


small {
    color: var(--text-muted);
}


[data-testid="stCaptionContainer"] {
    max-width: 100%;

    color: var(--text-secondary);

    white-space: normal !important;

    overflow-wrap: anywhere;
}


/* ==========================================================================
   SECTION HEADINGS
   ========================================================================== */

.section-heading {
    max-width: 100%;

    margin-top: .45rem;
    margin-bottom: 1.05rem;
}


.section-eyebrow {
    margin-bottom: .28rem;

    color: #70A6FF;

    font-size: .66rem;
    font-weight: 800;

    letter-spacing: .14em;

    text-transform: uppercase;
}


.section-title {
    max-width: 100%;

    color: #F5F8FD;

    font-size:
        clamp(
            1.05rem,
            1.8vw,
            1.28rem
        );

    line-height: 1.25;

    font-weight: 800;

    letter-spacing: -.028em;

    white-space: normal;

    overflow-wrap: anywhere;
}


.section-subtitle {
    max-width: 900px;

    margin-top: .34rem;

    color: #8593A9;

    font-size: .83rem;
    line-height: 1.55;

    white-space: normal;

    overflow-wrap: anywhere;
}


/* ==========================================================================
   PAGE INTRO
   ========================================================================== */

.page-intro {
    max-width: 100%;

    margin-bottom: 1.55rem;
}


.page-intro-eyebrow {
    max-width: 100%;

    margin-bottom: .32rem;

    color: #70A6FF;

    font-size: .67rem;

    font-weight: 850;

    letter-spacing: .14em;

    text-transform: uppercase;

    white-space: normal;

    overflow-wrap: anywhere;
}


.page-intro-title {
    max-width: 100%;

    color: #F8FAFC;

    font-size:
        clamp(
            1.4rem,
            2.5vw,
            1.72rem
        );

    font-weight: 850;

    letter-spacing: -.04em;

    white-space: normal;

    overflow-wrap: anywhere;
}


.page-intro-subtitle {
    max-width: 850px;

    margin-top: .35rem;

    color: #8996AA;

    font-size: .90rem;
    line-height: 1.6;

    white-space: normal;

    overflow-wrap: anywhere;
}


/* ==========================================================================
   GENERIC GLASS CARDS
   ========================================================================== */

.glass-card {
    position: relative;

    width: 100%;
    min-width: 0;

    min-height: 128px;
    height: auto;

    overflow: hidden;

    padding:
        1.3rem
        1.4rem;

    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,.058),
            rgba(255,255,255,.020)
        );

    border:
        1px solid var(--border);

    border-radius:
        var(--radius-xl);

    box-shadow:
        var(--shadow-md);

    backdrop-filter:
        blur(16px);

    -webkit-backdrop-filter:
        blur(16px);

    transition:
        transform var(--transition),
        border-color var(--transition),
        background var(--transition),
        box-shadow var(--transition);
}


.glass-card::after {
    content: "";

    position: absolute;

    inset: 0;

    pointer-events: none;

    border-radius: inherit;

    background:
        linear-gradient(
            120deg,
            rgba(255,255,255,.035),
            transparent 35%
        );
}


.glass-card:hover {
    transform: translateY(-2px);

    border-color: var(--border-strong);

    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,.071),
            rgba(255,255,255,.026)
        );

    box-shadow:
        var(--shadow-lg);
}


/* ==========================================================================
   METRIC CARDS
   ========================================================================== */

.metric-card-pro {
    width: 100%;

    min-width: 0;

    height: auto;

    overflow: hidden;
}


.metric-accent {
    position: absolute;

    top: 0;
    left: 0;

    width: 100%;
    height: 2px;

    opacity: .88;
}


.metric-label {
    position: relative;
    z-index: 1;

    max-width: 100%;

    color: #8997AE;

    font-size: .72rem;

    font-weight: 750;

    text-transform: uppercase;

    letter-spacing: .105em;

    white-space: normal;

    overflow-wrap: anywhere;
}


.metric-value {
    position: relative;
    z-index: 1;

    max-width: 100%;

    margin-top: .52rem;

    color: #FFFFFF;

    font-size:
        clamp(
            1.25rem,
            2.2vw,
            2rem
        );

    line-height: 1.12;

    font-weight: 850;

    letter-spacing: -.04em;

    white-space: normal !important;

    overflow: visible !important;

    text-overflow: clip !important;

    overflow-wrap: anywhere;
}


.metric-helper {
    position: relative;
    z-index: 1;

    min-height: 1.15rem;

    margin-top: .52rem;

    color: #748299;

    font-size: .76rem;

    line-height: 1.42;

    white-space: normal;

    overflow-wrap: anywhere;
}


/* ==========================================================================
   MINI METRICS
   ========================================================================== */

.mini-metric {
    --mini-accent: #A6B2C7;

    position: relative;

    width: 100%;
    min-width: 0;

    min-height: 96px;

    overflow: hidden;

    padding:
        .92rem
        1rem;

    border:
        1px solid var(--border-soft);

    border-radius:
        var(--radius-md);

    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,.032),
            rgba(255,255,255,.014)
        );

    box-shadow:
        var(--shadow-sm);

    transition:
        transform var(--transition-fast),
        border-color var(--transition-fast),
        background var(--transition-fast);
}


.mini-metric::before {
    content: "";

    position: absolute;

    top: 0;
    left: 0;

    width: 100%;
    height: 2px;

    background:
        var(--mini-accent);

    opacity: .82;
}


.mini-metric:hover {
    transform: translateY(-1px);

    border-color:
        var(--border-strong);

    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,.042),
            rgba(255,255,255,.018)
        );
}


.mini-metric-label {
    max-width: 100%;

    color: #7F8DA3;

    font-size: .67rem;

    font-weight: 780;

    letter-spacing: .095em;

    text-transform: uppercase;

    white-space: normal;

    overflow-wrap: anywhere;
}


.mini-metric-value {
    max-width: 100%;

    margin-top: .34rem;

    color: #F7FAFF;

    font-size:
        clamp(
            1rem,
            1.8vw,
            1.35rem
        );

    line-height: 1.2;

    font-weight: 820;

    letter-spacing: -.025em;

    white-space: normal !important;

    overflow: visible !important;

    text-overflow: clip !important;

    overflow-wrap: anywhere !important;
}


.mini-metric-helper {
    max-width: 100%;

    margin-top: .32rem;

    color: #69788F;

    font-size: .70rem;

    line-height: 1.42;

    white-space: normal;

    overflow-wrap: anywhere;
}


/* ==========================================================================
   SURFACE CARD
   ========================================================================== */

.surface-card-heading {
    --surface-accent: #A6B2C7;

    position: relative;

    width: 100%;
    min-width: 0;

    margin:
        .35rem
        0
        .9rem;

    padding:
        .9rem
        1rem
        .9rem
        1.15rem;

    overflow: hidden;

    border:
        1px solid var(--border-soft);

    border-radius:
        var(--radius-md);

    background:
        linear-gradient(
            135deg,
            rgba(255,255,255,.028),
            rgba(255,255,255,.012)
        );
}


.surface-card-heading::before {
    content: "";

    position: absolute;

    top: .75rem;
    bottom: .75rem;
    left: 0;

    width: 3px;

    border-radius: 999px;

    background:
        var(--surface-accent);
}


.surface-card-title {
    max-width: 100%;

    color: #EDF3FC;

    font-size: .92rem;

    line-height: 1.35;

    font-weight: 780;

    letter-spacing: -.015em;

    white-space: normal;

    overflow-wrap: anywhere;
}


.surface-card-subtitle {
    max-width: 900px;

    margin-top: .25rem;

    color: #7E8CA2;

    font-size: .78rem;

    line-height: 1.5;

    white-space: normal;

    overflow-wrap: anywhere;
}


/* ==========================================================================
   NATIVE STREAMLIT METRICS
   ========================================================================== */

div[data-testid="stMetric"] {
    width: 100%;

    min-width: 0;

    min-height: 124px;
    height: auto;

    overflow: visible;

    padding:
        1rem
        1.08rem;

    border:
        1px solid var(--border);

    border-radius:
        var(--radius-md);

    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,.040),
            rgba(255,255,255,.018)
        );

    box-shadow:
        var(--shadow-sm);
}


div[data-testid="stMetric"] > div {
    min-width: 0;
    max-width: 100%;
}


div[data-testid="stMetric"] label {
    max-width: 100%;

    color: #8492A8 !important;

    white-space: normal !important;

    overflow: visible !important;

    text-overflow: clip !important;

    overflow-wrap: anywhere;
}


div[data-testid="stMetricValue"] {
    max-width: 100%;

    color: #FFFFFF;

    font-size:
        clamp(
            1.45rem,
            2.5vw,
            2.45rem
        ) !important;

    line-height: 1.08 !important;

    font-weight: 820;

    letter-spacing: -.035em;

    white-space: normal !important;

    overflow: visible !important;

    text-overflow: clip !important;

    overflow-wrap: anywhere !important;

    word-break: normal !important;
}


div[data-testid="stMetricValue"] > div {
    max-width: 100%;

    white-space: normal !important;

    overflow: visible !important;

    text-overflow: clip !important;

    overflow-wrap: anywhere !important;
}


div[data-testid="stMetricDelta"] {
    font-size: .75rem;

    white-space: normal;
}


/* ==========================================================================
   APPLICATION BADGES
   ========================================================================== */

.app-badge {
    --badge-color: #A6B2C7;

    display: inline-flex;

    align-items: center;
    justify-content: center;

    gap: .32rem;

    max-width: 100%;

    padding:
        .42rem
        .72rem;

    border:
        1px solid
        var(--badge-color);

    border-radius:
        999px;

    background:
        rgba(255,255,255,.025);

    color:
        var(--badge-color);

    font-size:
        .72rem;

    font-weight:
        800;

    letter-spacing:
        .055em;

    line-height:
        1.2;

    white-space: normal;

    overflow-wrap: anywhere;

    text-align: center;

    box-shadow:
        inset 0 0 0 1px
        rgba(255,255,255,.015);
}


/* ==========================================================================
   LEGACY STATUS BADGES
   ========================================================================== */

.status-ok,
.status-offline {
    display: inline-flex;

    align-items: center;

    gap: .32rem;

    max-width: 100%;

    padding:
        .42rem
        .72rem;

    border-radius: 999px;

    font-size: .72rem;

    font-weight: 800;

    letter-spacing: .055em;

    white-space: normal;

    overflow-wrap: anywhere;
}


.status-ok {
    border:
        1px solid
        rgba(97,231,166,.34);

    background:
        var(--success-soft);

    color:
        var(--success);
}


.status-offline {
    border:
        1px solid
        rgba(255,92,122,.34);

    background:
        var(--critical-soft);

    color:
        var(--critical);
}


/* ==========================================================================
   RISK BADGE
   ========================================================================== */

.risk-badge-wrapper {
    width: 100%;

    display: flex;

    justify-content: center;

    margin-top:
        .65rem;
}


.risk-badge {
    --risk-color: #61E7A6;

    display: inline-flex;

    align-items: center;
    justify-content: center;

    max-width: 100%;

    min-width:
        92px;

    padding:
        .50rem
        1rem;

    border:
        1px solid
        var(--risk-color);

    border-radius:
        999px;

    background:
        rgba(255,255,255,.025);

    color:
        var(--risk-color);

    font-size:
        .77rem;

    font-weight:
        850;

    letter-spacing:
        .135em;

    text-transform:
        uppercase;

    white-space: normal;

    overflow-wrap: anywhere;

    text-align: center;

    box-shadow:
        0 8px 24px
        rgba(0,0,0,.14);
}


/* ==========================================================================
   RISK GAUGE
   ========================================================================== */

.risk-gauge-wrapper {
    width: 100%;

    display: flex;

    align-items: center;
    justify-content: center;

    padding:
        1.35rem
        0;
}


.risk-gauge-ring {
    --risk-color: #61E7A6;
    --risk-degrees: 0deg;

    position: relative;

    width:
        clamp(
            190px,
            18vw,
            238px
        );

    max-width: 100%;

    aspect-ratio: 1;

    display: flex;

    align-items: center;
    justify-content: center;

    border-radius:
        50%;

    background:
        conic-gradient(
            from 0deg,
            var(--risk-color) 0deg,
            var(--risk-color) var(--risk-degrees),
            rgba(255,255,255,.065) var(--risk-degrees),
            rgba(255,255,255,.065) 360deg
        );

    box-shadow:
        0 0 58px rgba(76,141,255,.10),
        0 18px 50px rgba(0,0,0,.22),
        inset 0 0 22px rgba(255,255,255,.025);

    transition:
        transform var(--transition),
        box-shadow var(--transition);
}


.risk-gauge-ring::before {
    content: "";

    position: absolute;

    inset: -1px;

    border:
        1px solid
        rgba(255,255,255,.075);

    border-radius:
        inherit;

    pointer-events:
        none;
}


.risk-gauge-ring::after {
    content: "";

    position: absolute;

    inset: 0;

    border-radius: inherit;

    background:
        radial-gradient(
            circle at 50% 0%,
            rgba(255,255,255,.08),
            transparent 34%
        );

    pointer-events:
        none;
}


.risk-gauge-ring:hover {
    transform:
        scale(1.015);

    box-shadow:
        0 0 65px rgba(76,141,255,.13),
        0 22px 55px rgba(0,0,0,.28),
        inset 0 0 24px rgba(255,255,255,.03);
}


.risk-gauge-inner {
    position: relative;
    z-index: 1;

    width: 76.5%;

    aspect-ratio: 1;

    display: flex;

    flex-direction: column;

    align-items: center;
    justify-content: center;

    border:
        1px solid
        rgba(255,255,255,.09);

    border-radius:
        50%;

    background:
        radial-gradient(
            circle at 50% 35%,
            rgba(255,255,255,.035),
            transparent 48%
        ),
        linear-gradient(
            145deg,
            #0B1220,
            #070B14
        );

    box-shadow:
        inset 0 0 35px
        rgba(0,0,0,.20);
}


.risk-gauge-label {
    max-width: 90%;

    color:
        #8794AA;

    font-size:
        .68rem;

    font-weight:
        750;

    letter-spacing:
        .14em;

    text-transform:
        uppercase;

    text-align: center;

    white-space: normal;
}


.risk-gauge-value {
    max-width: 90%;

    margin-top:
        .16rem;

    color:
        #FFFFFF;

    font-size:
        clamp(
            1.8rem,
            3vw,
            2.35rem
        );

    line-height:
        1.1;

    font-weight:
        850;

    letter-spacing:
        -.045em;

    white-space: normal;

    text-align: center;

    overflow-wrap: anywhere;
}


.risk-gauge-tier {
    max-width: 90%;

    margin-top:
        .28rem;

    font-size:
        .78rem;

    font-weight:
        850;

    letter-spacing:
        .135em;

    text-transform:
        uppercase;

    text-align: center;

    white-space: normal;

    overflow-wrap: anywhere;
}


/* ==========================================================================
   INFORMATION PANELS
   ========================================================================== */

.info-panel {
    --panel-color: #63A7FF;

    position: relative;

    width: 100%;
    min-width: 0;

    overflow: hidden;

    padding:
        1rem
        1.15rem;

    border:
        1px solid
        rgba(255,255,255,.08);

    border-left:
        3px solid
        var(--panel-color);

    border-radius:
        var(--radius-md);

    background:
        linear-gradient(
            135deg,
            rgba(255,255,255,.035),
            rgba(255,255,255,.015)
        );

    box-shadow:
        var(--shadow-sm);
}


.info-panel::before {
    content: "";

    position: absolute;

    width: 90px;
    height: 90px;

    left: -40px;
    top: -40px;

    border-radius: 50%;

    background:
        var(--panel-color);

    opacity: .05;

    filter:
        blur(22px);

    pointer-events: none;
}


.info-panel-title {
    position: relative;

    max-width: 100%;

    color:
        var(--panel-color);

    font-size:
        .72rem;

    font-weight:
        850;

    letter-spacing:
        .075em;

    text-transform:
        uppercase;

    white-space: normal;

    overflow-wrap: anywhere;
}


.info-panel-message {
    position: relative;

    max-width: 1000px;

    margin-top:
        .38rem;

    color:
        #A9B5C8;

    font-size:
        .86rem;

    line-height:
        1.58;

    white-space: normal;

    overflow-wrap: anywhere;
}


/* ==========================================================================
   DECISION PANEL
   ========================================================================== */

.decision-panel {
    --decision-color: #63A7FF;

    position: relative;

    width: 100%;
    min-width: 0;

    padding:
        1.05rem
        1.15rem;

    overflow: hidden;

    border:
        1px solid var(--border);

    border-radius:
        var(--radius-lg);

    background:
        linear-gradient(
            135deg,
            rgba(255,255,255,.040),
            rgba(255,255,255,.014)
        );

    box-shadow:
        var(--shadow-sm);
}


.decision-panel::before {
    content: "";

    position: absolute;

    width: 130px;
    height: 130px;

    top: -75px;
    right: -60px;

    border-radius: 50%;

    background:
        var(--decision-color);

    opacity: .055;

    filter:
        blur(28px);

    pointer-events: none;
}


.decision-panel-header {
    position: relative;

    display: flex;

    align-items: center;

    gap: .55rem;

    min-width: 0;
}


.decision-panel-dot {
    flex: 0 0 auto;

    width: 8px;
    height: 8px;

    border-radius: 50%;

    background:
        var(--decision-color);

    box-shadow:
        0 0 16px
        var(--decision-color);
}


.decision-panel-title {
    min-width: 0;

    max-width: 100%;

    color:
        var(--decision-color);

    font-size: .74rem;

    font-weight: 850;

    letter-spacing: .075em;

    text-transform: uppercase;

    white-space: normal;

    overflow-wrap: anywhere;
}


.decision-panel-message {
    position: relative;

    max-width: 1000px;

    margin-top: .48rem;

    color:
        #B2BED0;

    font-size: .88rem;

    line-height: 1.62;

    white-space: normal;

    overflow-wrap: anywhere;
}


.decision-panel-caption {
    position: relative;

    max-width: 1000px;

    margin-top: .55rem;

    color:
        #718096;

    font-size: .72rem;

    line-height: 1.5;

    white-space: normal;

    overflow-wrap: anywhere;
}


/* ==========================================================================
   TREE SHAP DRIVER CARDS
   ========================================================================== */

.driver-card {
    --driver-color: #A6B2C7;

    position: relative;

    width: 100%;
    min-width: 0;

    margin-bottom: .62rem;

    padding:
        .95rem
        1rem;

    overflow: hidden;

    border:
        1px solid var(--border-soft);

    border-left:
        3px solid
        var(--driver-color);

    border-radius:
        var(--radius-md);

    background:
        linear-gradient(
            135deg,
            rgba(255,255,255,.032),
            rgba(255,255,255,.012)
        );

    box-shadow:
        0 8px 26px
        rgba(0,0,0,.10);

    transition:
        transform var(--transition-fast),
        border-color var(--transition-fast),
        background var(--transition-fast),
        box-shadow var(--transition-fast);
}


.driver-card::before {
    content: "";

    position: absolute;

    width: 90px;
    height: 90px;

    top: -45px;
    left: -45px;

    border-radius: 50%;

    background:
        var(--driver-color);

    opacity: .045;

    filter:
        blur(18px);

    pointer-events: none;
}


.driver-card:hover {
    transform:
        translateY(-1px);

    border-color:
        var(--border);

    border-left-color:
        var(--driver-color);

    background:
        linear-gradient(
            135deg,
            rgba(255,255,255,.043),
            rgba(255,255,255,.017)
        );

    box-shadow:
        0 12px 32px
        rgba(0,0,0,.14);
}


.driver-direction {
    position: relative;

    max-width: 100%;

    color:
        var(--driver-color);

    font-size:
        .62rem;

    font-weight:
        850;

    letter-spacing:
        .105em;

    text-transform:
        uppercase;

    white-space:
        normal;

    overflow-wrap:
        anywhere;
}


.driver-label {
    position: relative;

    max-width: 100%;

    margin-top: .28rem;

    color:
        #EEF4FC;

    font-size:
        .90rem;

    line-height:
        1.4;

    font-weight:
        750;

    white-space:
        normal;

    overflow-wrap:
        anywhere;
}


.driver-meta {
    position: relative;

    display: flex;

    flex-wrap: wrap;

    gap:
        .55rem
        1.25rem;

    margin-top:
        .65rem;
}


.driver-value,
.driver-contribution {
    min-width: 0;

    display: flex;

    flex-direction: column;

    gap: .12rem;
}


.driver-value span,
.driver-contribution span {
    color:
        #69778D;

    font-size:
        .64rem;

    font-weight:
        700;

    letter-spacing:
        .06em;

    text-transform:
        uppercase;
}


.driver-value strong,
.driver-contribution strong {
    max-width: 100%;

    color:
        #C9D4E3;

    font-size:
        .77rem;

    line-height:
        1.4;

    font-weight:
        720;

    white-space:
        normal;

    overflow-wrap:
        anywhere;
}


.driver-contribution strong {
    color:
        var(--driver-color);
}


/* ==========================================================================
   KEY VALUE ROW
   ========================================================================== */

.key-value-row {
    width: 100%;
    min-width: 0;

    display: grid;

    grid-template-columns:
        minmax(130px, .75fr)
        minmax(0, 1.25fr);

    gap: .75rem;

    align-items: start;

    padding:
        .72rem
        .15rem;

    border-bottom:
        1px solid
        rgba(255,255,255,.045);
}


.key-value-row:last-child {
    border-bottom: none;
}


.key-value-label {
    min-width: 0;

    color:
        #77869C;

    font-size:
        .72rem;

    line-height:
        1.45;

    font-weight:
        700;

    white-space:
        normal;

    overflow-wrap:
        anywhere;
}


.key-value-value {
    min-width: 0;

    max-width: 100%;

    color:
        #DCE5F1;

    font-size:
        .78rem;

    line-height:
        1.48;

    font-weight:
        650;

    text-align: right;

    white-space:
        normal;

    overflow:
        visible;

    text-overflow:
        clip;

    overflow-wrap:
        anywhere;
}


.key-value-monospace {
    font-family:
        "SFMono-Regular",
        Consolas,
        "Liberation Mono",
        Menlo,
        monospace;

    font-size:
        .72rem;

    word-break:
        break-word;
}


/* ==========================================================================
   SOFT DIVIDER
   ========================================================================== */

.soft-divider {
    width: 100%;
    min-width: 0;

    display: flex;

    align-items: center;

    gap: .65rem;

    margin:
        .9rem
        0;
}


.soft-divider-line {
    flex: 1 1 auto;

    min-width: 10px;

    height: 1px;

    background:
        var(--border-soft);
}


.soft-divider-label {
    flex: 0 1 auto;

    max-width: 70%;

    color:
        #64738A;

    font-size:
        .64rem;

    font-weight:
        750;

    letter-spacing:
        .085em;

    text-transform:
        uppercase;

    text-align:
        center;

    white-space:
        normal;

    overflow-wrap:
        anywhere;
}


/* ==========================================================================
   BUTTONS
   ========================================================================== */

.stButton > button,
.stDownloadButton > button,
[data-testid="stFormSubmitButton"] > button {
    min-width: 0;

    min-height: 45px;
    height: auto;

    padding:
        .58rem
        1.05rem;

    border:
        1px solid
        rgba(111,158,255,.32);

    border-radius:
        12px;

    background:
        linear-gradient(
            135deg,
            #286CF4 0%,
            #6255EE 100%
        );

    color:
        #FFFFFF;

    font-weight:
        750;

    letter-spacing:
        -.005em;

    white-space: normal;

    overflow-wrap: anywhere;

    box-shadow:
        0 8px 25px
        rgba(50,100,240,.14);

    transition:
        transform var(--transition-fast),
        border-color var(--transition-fast),
        box-shadow var(--transition-fast),
        filter var(--transition-fast);
}


.stButton > button:hover,
.stDownloadButton > button:hover,
[data-testid="stFormSubmitButton"] > button:hover {
    transform:
        translateY(-1px);

    border-color:
        rgba(255,255,255,.40);

    box-shadow:
        0 12px 32px
        rgba(70,100,255,.22);

    filter:
        brightness(1.06);
}


.stButton > button:active,
.stDownloadButton > button:active {
    transform:
        translateY(0);
}


.stButton > button:disabled {
    opacity:
        .45;

    cursor:
        not-allowed;

    transform:
        none;
}


button[kind="secondary"] {
    background:
        rgba(255,255,255,.035) !important;

    border:
        1px solid var(--border) !important;

    color:
        #D8E1EE !important;

    box-shadow:
        none !important;
}


button[kind="secondary"]:hover {
    background:
        rgba(255,255,255,.065) !important;

    border-color:
        var(--border-strong) !important;
}


.stButton > button *,
.stDownloadButton > button *,
[data-testid="stFormSubmitButton"] > button * {
    max-width: 100%;

    white-space: normal !important;

    overflow: visible !important;

    text-overflow: clip !important;

    overflow-wrap: anywhere !important;

    text-align: center;
}


/* ==========================================================================
   FORM LABELS
   ========================================================================== */

[data-testid="stWidgetLabel"] p {
    max-width: 100%;

    color:
        #C4CDDA;

    font-size:
        .81rem;

    font-weight:
        650;

    white-space: normal;

    overflow-wrap: anywhere;
}


/* ==========================================================================
   INPUTS
   ========================================================================== */

div[data-baseweb="input"] > div,
div[data-baseweb="textarea"] > div,
div[data-baseweb="select"] > div {
    min-width: 0;

    background:
        rgba(255,255,255,.030)
        !important;

    border-color:
        var(--border)
        !important;

    border-radius:
        11px
        !important;

    transition:
        border-color var(--transition-fast),
        box-shadow var(--transition-fast),
        background var(--transition-fast);
}


div[data-baseweb="input"] > div:hover,
div[data-baseweb="textarea"] > div:hover,
div[data-baseweb="select"] > div:hover {
    border-color:
        var(--border-strong)
        !important;

    background:
        rgba(255,255,255,.040)
        !important;
}


div[data-baseweb="input"] > div:focus-within,
div[data-baseweb="textarea"] > div:focus-within,
div[data-baseweb="select"] > div:focus-within {
    border-color:
        rgba(76,141,255,.62)
        !important;

    box-shadow:
        0 0 0 3px
        rgba(76,141,255,.10)
        !important;
}


input,
textarea {
    min-width: 0;

    color:
        #EEF4FC !important;

    caret-color:
        var(--primary);
}


input::placeholder,
textarea::placeholder {
    color:
        #536176 !important;
}


/* ==========================================================================
   SELECTBOX / MULTISELECT
   ========================================================================== */

[data-baseweb="select"] {
    min-width: 0;

    border-radius:
        11px;
}


[data-baseweb="select"] * {
    min-width: 0;

    max-width: 100%;
}


[data-baseweb="select"] span,
[data-baseweb="select"] div {
    max-width: 100%;
}


[data-baseweb="select"] p {
    white-space:
        normal !important;

    overflow:
        visible !important;

    text-overflow:
        clip !important;

    overflow-wrap:
        anywhere !important;
}


[data-baseweb="popover"] {
    border-radius:
        var(--radius-md);

    max-width:
        min(
            92vw,
            760px
        ) !important;
}


[data-baseweb="popover"] > div {
    max-width:
        min(
            92vw,
            760px
        ) !important;
}


[data-baseweb="menu"] {
    max-width:
        min(
            92vw,
            760px
        ) !important;

    overflow-x:
        hidden;

    background:
        #0C1321 !important;

    border:
        1px solid var(--border);

    box-shadow:
        var(--shadow-lg);
}


[data-baseweb="menu"] li,
[data-baseweb="menu"] [role="option"] {
    min-width: 0;

    max-width: 100%;

    height:
        auto !important;

    min-height:
        38px;

    white-space:
        normal !important;

    overflow:
        visible !important;

    text-overflow:
        clip !important;

    overflow-wrap:
        anywhere !important;
}


/* ==========================================================================
   CHECKBOX / TOGGLE
   ========================================================================== */

[data-testid="stCheckbox"] label,
[data-testid="stToggle"] label {
    color:
        #C6D0DE;

    white-space:
        normal;
}


/* ==========================================================================
   SLIDERS
   ========================================================================== */

[data-testid="stSlider"] {
    padding-top:
        .1rem;
}


[data-testid="stSlider"] [role="slider"] {
    box-shadow:
        0 0 0 4px
        rgba(76,141,255,.10);
}


/* ==========================================================================
   FILE UPLOADER
   ========================================================================== */

[data-testid="stFileUploaderDropzone"] {
    min-width: 0;

    padding:
        2.15rem
        1.5rem;

    background:
        linear-gradient(
            145deg,
            rgba(76,141,255,.035),
            rgba(255,255,255,.018)
        );

    border:
        1px dashed
        rgba(139,169,220,.22);

    border-radius:
        var(--radius-xl);

    transition:
        background var(--transition),
        border-color var(--transition);
}


[data-testid="stFileUploaderDropzone"]:hover {
    background:
        linear-gradient(
            145deg,
            rgba(76,141,255,.060),
            rgba(255,255,255,.026)
        );

    border-color:
        rgba(76,141,255,.42);
}


[data-testid="stFileUploaderDropzoneInstructions"] {
    color:
        var(--text-secondary);

    white-space:
        normal;

    overflow-wrap:
        anywhere;
}


/* ==========================================================================
   FORMS
   ========================================================================== */

[data-testid="stForm"] {
    min-width: 0;

    padding:
        1.15rem;

    border:
        1px solid var(--border-soft);

    border-radius:
        var(--radius-lg);

    background:
        rgba(255,255,255,.014);
}


[data-testid="stForm"] *,
[data-testid="stFileUploader"] *,
[data-testid="stCheckbox"] *,
[data-testid="stToggle"] * {
    max-width: 100%;
}


/* ==========================================================================
   TABS
   ========================================================================== */

[data-testid="stTabs"] {
    min-width: 0;

    margin-top:
        .25rem;
}


[data-baseweb="tab-list"] {
    gap:
        .28rem;

    border-bottom:
        1px solid var(--border-soft);

    overflow-x:
        auto;
}


button[data-baseweb="tab"] {
    min-height:
        43px;

    padding:
        .5rem
        .8rem;

    color:
        #8491A7;

    font-weight:
        650;

    border-radius:
        9px 9px 0 0;

    white-space:
        normal;

    transition:
        color var(--transition-fast),
        background var(--transition-fast);
}


button[data-baseweb="tab"]:hover {
    color:
        #D6DFEC;

    background:
        rgba(255,255,255,.025);
}


button[data-baseweb="tab"][aria-selected="true"] {
    color:
        #FFFFFF;

    background:
        linear-gradient(
            180deg,
            rgba(76,141,255,.075),
            transparent
        );
}


button[data-baseweb="tab"] *,
[data-baseweb="tab-panel"],
[data-baseweb="tab-panel"] * {
    max-width:
        100%;

    white-space:
        normal;

    overflow-wrap:
        anywhere;
}


/* ==========================================================================
   EXPANDERS
   ========================================================================== */

[data-testid="stExpander"] {
    min-width: 0;

    overflow:
        hidden;

    border:
        1px solid var(--border);

    border-radius:
        var(--radius-md);

    background:
        rgba(255,255,255,.018);
}


[data-testid="stExpander"] summary {
    white-space:
        normal;

    transition:
        background var(--transition-fast);
}


[data-testid="stExpander"] summary:hover {
    background:
        rgba(255,255,255,.025);
}


[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary *,
[data-testid="stExpanderDetails"],
[data-testid="stExpanderDetails"] * {
    max-width: 100%;

    white-space:
        normal !important;

    overflow-wrap:
        anywhere !important;
}


/* ==========================================================================
   ALERTS
   ========================================================================== */

[data-testid="stAlert"] {
    min-width: 0;

    border:
        1px solid var(--border);

    border-radius:
        var(--radius-md);

    backdrop-filter:
        blur(10px);

    box-shadow:
        0 8px 28px
        rgba(0,0,0,.10);
}


[data-testid="stAlert"] * {
    white-space:
        normal;

    overflow-wrap:
        anywhere;
}


/* ==========================================================================
   DATAFRAMES / TABLES
   ========================================================================== */

[data-testid="stDataFrame"] {
    max-width: 100%;

    overflow:
        auto;

    border:
        1px solid var(--border);

    border-radius:
        var(--radius-lg);

    background:
        rgba(255,255,255,.018);

    box-shadow:
        var(--shadow-sm);
}


[data-testid="stDataFrame"] > div {
    max-width:
        100% !important;
}


[data-testid="stDataFrame"] canvas {
    max-width:
        none;
}


[data-testid="stTable"] {
    max-width: 100%;

    overflow:
        auto;

    border:
        1px solid var(--border);

    border-radius:
        var(--radius-md);
}


/* ==========================================================================
   JSON / CODE
   ========================================================================== */

[data-testid="stJson"] {
    max-width: 100%;

    overflow:
        auto;

    border:
        1px solid var(--border);

    border-radius:
        var(--radius-md);

    background:
        rgba(4,8,16,.52);
}


[data-testid="stCode"] {
    max-width: 100%;

    overflow:
        auto;

    border:
        1px solid var(--border);

    border-radius:
        var(--radius-md);
}


code,
pre {
    max-width: 100%;

    overflow-x:
        auto;

    white-space:
        pre-wrap;

    overflow-wrap:
        anywhere;
}


/* ==========================================================================
   CHART CONTAINERS
   ========================================================================== */

[data-testid="stVegaLiteChart"],
[data-testid="stPlotlyChart"] {
    width: 100%;

    max-width: 100%;

    overflow:
        hidden;

    padding:
        .3rem;

    border:
        1px solid var(--border-soft);

    border-radius:
        var(--radius-lg);

    background:
        rgba(255,255,255,.010);
}


/* ==========================================================================
   IMAGES
   ========================================================================== */

[data-testid="stImage"] {
    max-width: 100%;
}


[data-testid="stImage"] img {
    max-width: 100%;

    height: auto;

    border-radius:
        var(--radius-lg);
}


/* ==========================================================================
   EMPTY STATES
   ========================================================================== */

.empty-state {
    position: relative;

    width: 100%;
    min-width: 0;

    overflow: hidden;

    padding:
        3.1rem
        2rem;

    text-align:
        center;

    border:
        1px dashed
        rgba(150,170,205,.18);

    border-radius:
        var(--radius-xl);

    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,.020),
            rgba(255,255,255,.010)
        );
}


.empty-state::before {
    content: "";

    position: absolute;

    width: 180px;
    height: 180px;

    left: 50%;
    top: -120px;

    transform:
        translateX(-50%);

    border-radius:
        50%;

    background:
        rgba(76,141,255,.06);

    filter:
        blur(30px);
}


.empty-title {
    position: relative;

    max-width: 100%;

    color:
        #F5F8FC;

    font-size:
        1.08rem;

    font-weight:
        780;

    white-space:
        normal;

    overflow-wrap:
        anywhere;
}


.empty-message {
    position: relative;

    max-width:
        620px;

    margin:
        .52rem
        auto
        0;

    color:
        var(--text-secondary);

    font-size:
        .84rem;

    line-height:
        1.6;

    white-space:
        normal;

    overflow-wrap:
        anywhere;
}


.empty-hint {
    position: relative;

    margin-top:
        .9rem;

    color:
        #66758D;

    font-size:
        .74rem;

    white-space:
        normal;

    overflow-wrap:
        anywhere;
}


/* ==========================================================================
   PROGRESS
   ========================================================================== */

[data-testid="stProgress"] > div > div {
    border-radius:
        999px;
}


/* ==========================================================================
   TOAST
   ========================================================================== */

[data-testid="stToast"] {
    max-width:
        min(
            420px,
            90vw
        );

    border:
        1px solid var(--border);

    border-radius:
        var(--radius-md);

    background:
        rgba(10,16,30,.94);

    backdrop-filter:
        blur(16px);

    box-shadow:
        var(--shadow-lg);
}


[data-testid="stToast"] * {
    white-space:
        normal;

    overflow-wrap:
        anywhere;
}


/* ==========================================================================
   DIALOGS
   ========================================================================== */

[data-testid="stDialog"] > div {
    max-width:
        min(
            920px,
            94vw
        );

    border:
        1px solid var(--border);

    border-radius:
        var(--radius-xl);

    background:
        #0A101D;
}


/* ==========================================================================
   TOOLTIPS
   ========================================================================== */

[data-baseweb="tooltip"],
[role="tooltip"] {
    max-width:
        min(
            420px,
            90vw
        ) !important;

    white-space:
        normal !important;

    overflow-wrap:
        anywhere !important;
}


/* ==========================================================================
   LINKS
   ========================================================================== */

a {
    max-width: 100%;

    overflow-wrap:
        anywhere;

    word-break:
        break-word;
}


/* ==========================================================================
   DIVIDERS
   ========================================================================== */

hr {
    border: none;

    border-top:
        1px solid var(--border-soft);
}


/* ==========================================================================
   FOCUS / ACCESSIBILITY
   ========================================================================== */

button:focus-visible,
input:focus-visible,
textarea:focus-visible,
[tabindex]:focus-visible {
    outline:
        2px solid
        rgba(76,141,255,.75);

    outline-offset:
        2px;
}


/* ==========================================================================
   SELECTION
   ========================================================================== */

::selection {
    color:
        #FFFFFF;

    background:
        rgba(76,141,255,.38);
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
        rgba(255,255,255,.115);

    border-radius:
        999px;
}


::-webkit-scrollbar-thumb:hover {
    background:
        rgba(255,255,255,.20);
}


/* ==========================================================================
   GLOBAL CONTENT SAFETY
   ========================================================================== */

.metric-value,
.mini-metric-label,
.mini-metric-value,
.mini-metric-helper,
.risk-gauge-value,
.risk-gauge-tier,
.section-title,
.page-intro-title,
.page-intro-eyebrow,
.dashboard-title,
.info-panel-title,
.info-panel-message,
.surface-card-title,
.surface-card-subtitle,
.decision-panel-title,
.decision-panel-message,
.decision-panel-caption,
.driver-direction,
.driver-label,
.driver-value,
.driver-contribution,
.key-value-label,
.key-value-value,
.empty-title,
.empty-message,
[data-testid="stMetricValue"],
[data-testid="stMetricLabel"],
[data-testid="stCaptionContainer"] {
    overflow:
        visible !important;

    text-overflow:
        clip !important;

    white-space:
        normal !important;

    overflow-wrap:
        anywhere !important;

    word-break:
        normal !important;
}


/* Generic business-value safety */

.model-name,
.model-value,
.action-value,
.status-value,
.claim-id,
.technical-value,
.business-value,
.business-label,
.card-value,
.card-label {
    max-width:
        100%;

    overflow:
        visible !important;

    text-overflow:
        clip !important;

    white-space:
        normal !important;

    overflow-wrap:
        anywhere !important;

    word-break:
        normal !important;
}


/* Never force important cards to one line */

.glass-card,
.metric-card-pro,
.mini-metric,
.surface-card-heading,
.info-panel,
.decision-panel,
.driver-card,
.empty-state,
[data-testid="stMetric"],
[data-testid="stAlert"],
[data-testid="stExpander"],
[data-testid="stForm"] {
    min-width:
        0;

    overflow-wrap:
        anywhere;
}


/* ==========================================================================
   RESPONSIVE — LARGE TABLET / LAPTOP
   ========================================================================== */

@media (max-width: 1300px) {

    .block-container {
        padding-left:
            2rem;

        padding-right:
            2rem;
    }


    div[data-testid="stMetricValue"] {
        font-size:
            clamp(
                1.35rem,
                2.3vw,
                2.1rem
            ) !important;
    }


    .metric-value {
        font-size:
            clamp(
                1.2rem,
                2vw,
                1.75rem
            );
    }

}


/* ==========================================================================
   RESPONSIVE — TABLET
   ========================================================================== */

@media (max-width: 1100px) {

    .block-container {
        padding-left:
            1.6rem;

        padding-right:
            1.6rem;
    }


    .glass-card {
        min-height:
            118px;
    }


    .mini-metric {
        min-height:
            90px;
    }


    .metric-value {
        font-size:
            clamp(
                1.2rem,
                2.5vw,
                1.65rem
            );
    }


    div[data-testid="stMetricValue"] {
        font-size:
            clamp(
                1.3rem,
                2.8vw,
                1.95rem
            ) !important;
    }


    .risk-gauge-ring {
        width:
            210px;
    }


    .driver-meta {
        gap:
            .5rem
            1rem;
    }


    .key-value-row {
        grid-template-columns:
            minmax(110px, .7fr)
            minmax(0, 1.3fr);
    }

}


/* ==========================================================================
   RESPONSIVE — MOBILE
   ========================================================================== */

@media (max-width: 700px) {

    .block-container {
        padding-top:
            1rem;

        padding-left:
            1rem;

        padding-right:
            1rem;

        padding-bottom:
            2.5rem;
    }


    .dashboard-title {
        font-size:
            2.05rem;

        line-height:
            1.08;
    }


    .dashboard-subtitle {
        margin-bottom:
            1rem;

        font-size:
            .88rem;
    }


    .page-context-strip {
        display:
            flex;

        width:
            fit-content;

        flex-wrap:
            wrap;
    }


    .page-intro-eyebrow {
        font-size:
            .63rem;
    }


    .glass-card,
    .metric-card-pro,
    div[data-testid="stMetric"] {
        min-height:
            auto;

        padding:
            1.1rem;
    }


    .surface-card-heading {
        padding:
            .82rem
            .9rem
            .82rem
            1.05rem;
    }


    .mini-metric {
        min-height:
            auto;

        padding:
            .85rem
            .9rem;
    }


    .mini-metric-value {
        font-size:
            1.15rem;
    }


    .metric-value {
        font-size:
            1.4rem;
    }


    div[data-testid="stMetricValue"] {
        font-size:
            1.6rem !important;
    }


    .section-title {
        font-size:
            1.15rem;
    }


    [data-testid="stFileUploaderDropzone"] {
        padding:
            1.4rem
            1rem;
    }


    .risk-gauge-wrapper {
        padding:
            1rem
            0;
    }


    .risk-gauge-ring {
        width:
            min(
                205px,
                70vw
            );
    }


    .risk-gauge-value {
        font-size:
            1.9rem;
    }


    .risk-badge {
        min-width:
            84px;
    }


    .info-panel {
        padding:
            .9rem
            1rem;
    }


    .decision-panel {
        padding:
            .92rem
            1rem;
    }


    .driver-card {
        padding:
            .88rem
            .92rem;
    }


    .driver-meta {
        flex-direction:
            column;

        gap:
            .48rem;
    }


    .key-value-row {
        grid-template-columns:
            1fr;

        gap:
            .22rem;

        padding:
            .65rem
            .1rem;
    }


    .key-value-value {
        text-align:
            left;
    }


    .soft-divider-label {
        max-width:
            82%;
    }

}


/* ==========================================================================
   VERY SMALL MOBILE
   ========================================================================== */

@media (max-width: 420px) {

    .block-container {
        padding-left:
            .75rem;

        padding-right:
            .75rem;
    }


    .dashboard-title {
        font-size:
            1.75rem;
    }


    .surface-card-title {
        font-size:
            .86rem;
    }


    .metric-value {
        font-size:
            1.25rem;
    }


    .mini-metric-value {
        font-size:
            1.05rem;
    }


    div[data-testid="stMetricValue"] {
        font-size:
            1.4rem !important;
    }


    .risk-gauge-ring {
        width:
            min(
                185px,
                72vw
            );
    }


    .decision-panel-message {
        font-size:
            .82rem;
    }


    .driver-label {
        font-size:
            .84rem;
    }

}


/* ==========================================================================
   REDUCED MOTION
   ========================================================================== */

@media (
    prefers-reduced-motion:
    reduce
) {

    *,
    *::before,
    *::after {
        scroll-behavior:
            auto !important;

        transition-duration:
            0.01ms !important;

        animation-duration:
            0.01ms !important;

        animation-iteration-count:
            1 !important;
    }

}

</style>
"""