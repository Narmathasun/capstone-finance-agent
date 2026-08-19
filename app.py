"""
Streamlit UI — three tabs:
1. Chat: conversational interface backed by the LangGraph multi-agent app
2. Portfolio: enter holdings -> visual dashboard (allocation, P/L, sector mix)
3. Market: watchlist of popular stocks + real-time quote lookup + price chart
"""
import json
import uuid
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config import settings
from src.graph import invoke as run_assistant
from src.tools.market_data import get_live_quote, get_price_history, get_portfolio_quotes, MarketDataError

st.set_page_config(page_title="AI Financial Assistant", page_icon="💰", layout="wide")

STOCK_UNIVERSE_PATH = Path(__file__).parent / "sample_data" / "stock_universe.json"


@st.cache_data
def load_stock_universe() -> dict:
    if STOCK_UNIVERSE_PATH.exists():
        with open(STOCK_UNIVERSE_PATH) as f:
            return json.load(f)
    return {"stocks": [], "default_watchlist": []}


# ---------- authentication ----------
# Two modes, chosen via ENABLE_MULTI_USER_AUTH in .env:
#
# 1. Multi-user accounts (ENABLE_MULTI_USER_AUTH=true): real per-person
#    signup + login via streamlit-authenticator, with each user's
#    portfolio persisted separately (src/auth/user_data.py) and their
#    conversation memory scoped to their own username.
#
# 2. Single shared password (default, ENABLE_MULTI_USER_AUTH=false):
#    the original lightweight gate — one shared APP_PASSWORD for
#    everyone, no individual accounts. Kept as the default so existing
#    deployments (e.g. Streamlit Cloud) don't change behavior unless you
#    deliberately opt in to real accounts.
current_username = "guest"
current_display_name = "Guest"

if settings.ENABLE_MULTI_USER_AUTH:
    from src.auth.auth_manager import get_authenticator

    authenticator = get_authenticator()

    st.title("💰 AI Financial Assistant")

    auth_tab_login, auth_tab_register = st.tabs(["Log in", "Create account"])
    with auth_tab_login:
        authenticator.login(location="main")
    with auth_tab_register:
        try:
            new_email, new_username, new_name = authenticator.register_user(
                location="main", captcha=False, roles=["user"]
            )
            if new_username:
                st.success(f"Account created for {new_username} — you can now log in on the Log in tab.")
        except Exception as e:
            st.error(f"Registration error: {e}")

    auth_status = st.session_state.get("authentication_status")
    if auth_status is False:
        st.error("Username or password is incorrect.")
        st.stop()
    elif auth_status is None:
        st.info("Please log in or create an account to continue.")
        st.stop()

    # auth_status is True from here on
    current_username = st.session_state.get("username", "guest")
    current_display_name = st.session_state.get("name", current_username)
else:
    # ---------- legacy single shared-password gate ----------
    _app_password = settings.APP_PASSWORD
    if _app_password:
        if "authenticated" not in st.session_state:
            st.session_state.authenticated = False
        if not st.session_state.authenticated:
            st.title("💰 AI Financial Assistant")
            st.caption("This demo is password-protected to prevent unrestricted API usage.")
            entered = st.text_input("Enter access password", type="password")
            if st.button("Enter"):
                if entered == _app_password:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Incorrect password.")
            st.stop()

# ---------- session state ----------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "portfolio" not in st.session_state:
    if settings.ENABLE_MULTI_USER_AUTH:
        # Real accounts get their saved portfolio back on every login,
        # rather than starting from empty each session.
        from src.auth.user_data import load_user_portfolio
        st.session_state.portfolio = load_user_portfolio(current_username)
    else:
        st.session_state.portfolio = []

# Conversation memory is scoped per-user when multi-user auth is on, so two
# different people never share a thread_id and can't see each other's
# chat history even if a session_id were somehow guessed or reused.
_graph_session_id = (
    f"{current_username}:{st.session_state.session_id}"
    if settings.ENABLE_MULTI_USER_AUTH else st.session_state.session_id
)


def _persist_portfolio():
    if settings.ENABLE_MULTI_USER_AUTH:
        from src.auth.user_data import save_user_portfolio
        save_user_portfolio(current_username, st.session_state.portfolio)


st.title("💰 AI Financial Assistant")
st.caption("Multi-agent system: Finance Q&A · Portfolio Analysis · Market Analysis · "
           "Goal Planning · News Synthesizer · Tax Education")
if settings.ENABLE_MULTI_USER_AUTH:
    st.caption(f"Logged in as **{current_display_name}**")

# ---------- persistent compliance disclaimer ----------
# Shown on every page load, every tab — not something a user can dismiss and
# lose track of. This is a structural control, not just prompt-level text,
# since LLM-generated disclaimers inside chat responses can't be guaranteed
# to appear on every single turn.
st.warning(
    "⚠️ **Educational tool only — not a registered investment adviser, broker, "
    "or tax professional.** This assistant provides general financial "
    "education and does not provide personalized investment, tax, or legal "
    "advice. Responses are generated by AI and may contain errors or outdated "
    "information — always verify important figures independently and consult "
    "a licensed professional before making financial decisions. No account "
    "data entered here is verified, insured, or professionally reviewed.",
    icon="⚠️",
)

tab_chat, tab_portfolio, tab_market = st.tabs(["💬 Chat", "📊 Portfolio", "📈 Market"])

# ================= CHAT TAB =================
with tab_chat:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("route"):
                st.caption(f"routed to: `{msg['route']}`" + (
                    f" · sources: {', '.join(msg['sources'][:3])}" if msg.get("sources") else ""
                ))

    user_query = st.chat_input("Ask about investing, your portfolio, market data, taxes, or goals...")
    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = run_assistant(
                        query=user_query,
                        user_id=current_username,
                        session_id=_graph_session_id,
                        portfolio=st.session_state.portfolio,
                    )
                    st.markdown(result["response"])
                    if result.get("route"):
                        st.caption(f"routed to: `{result['route']}`" + (
                            f" · sources: {', '.join(result['sources'][:3])}" if result.get("sources") else ""
                        ))
                    st.session_state.chat_history.append({
                        "role": "assistant", "content": result["response"],
                        "route": result.get("route"), "sources": result.get("sources"),
                    })
                except Exception as e:
                    err_msg = f"Something went wrong: {e}"
                    st.error(err_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": err_msg})

# ================= PORTFOLIO TAB =================
with tab_portfolio:
    st.subheader("Your Holdings")
    universe = load_stock_universe()
    ticker_options = [s["ticker"] for s in universe.get("stocks", [])]

    with st.form("add_holding"):
        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
        if ticker_options:
            picked = c1.selectbox("Ticker", options=["(type below)"] + ticker_options)
            typed = c1.text_input("Or type a ticker not in the list", "").upper().strip()
            ticker = typed if typed else (picked if picked != "(type below)" else "")
        else:
            ticker = c1.text_input("Ticker").upper().strip()
        shares = c2.number_input("Shares", min_value=0.0, step=1.0)
        cost_basis = c3.number_input("Cost Basis / share ($)", min_value=0.0, step=1.0)
        submitted = c4.form_submit_button("Add")
        if submitted and ticker and shares > 0:
            st.session_state.portfolio.append(
                {"ticker": ticker, "shares": shares, "cost_basis": cost_basis}
            )
            _persist_portfolio()
            st.success(f"Added {shares} shares of {ticker}")

    if st.session_state.portfolio:
        df = pd.DataFrame(st.session_state.portfolio)
        st.dataframe(df, use_container_width=True)

        if st.button("🗑️ Clear Portfolio"):
            st.session_state.portfolio = []
            _persist_portfolio()
            st.rerun()

        if st.button("🔍 Analyze Portfolio", type="primary"):
            with st.spinner("Fetching live data and analyzing..."):
                rows, errors = [], []
                for h in st.session_state.portfolio:
                    try:
                        q = get_live_quote(h["ticker"])
                        market_value = q["price"] * h["shares"]
                        cost_value = h["cost_basis"] * h["shares"]
                        rows.append({
                            "Ticker": h["ticker"], "Shares": h["shares"],
                            "Price": q["price"], "Market Value": round(market_value, 2),
                            "Cost Basis": round(cost_value, 2),
                            "Gain/Loss": round(market_value - cost_value, 2),
                            "Gain/Loss %": round((market_value - cost_value) / cost_value * 100, 2) if cost_value else 0,
                        })
                    except MarketDataError as e:
                        errors.append(f"{h['ticker']}: {e}")

                if errors:
                    st.warning("Some tickers failed to load:\n" + "\n".join(errors))

                if rows:
                    result_df = pd.DataFrame(rows)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Value", f"${result_df['Market Value'].sum():,.2f}")
                        fig_alloc = px.pie(result_df, values="Market Value", names="Ticker",
                                            title="Allocation by Holding")
                        st.plotly_chart(fig_alloc, use_container_width=True)
                    with col2:
                        total_gl = result_df["Gain/Loss"].sum()
                        st.metric("Total Gain/Loss", f"${total_gl:,.2f}")
                        fig_gl = go.Figure(go.Bar(
                            x=result_df["Ticker"], y=result_df["Gain/Loss %"],
                            marker_color=["green" if v >= 0 else "red" for v in result_df["Gain/Loss %"]],
                        ))
                        fig_gl.update_layout(title="Gain/Loss % by Holding")
                        st.plotly_chart(fig_gl, use_container_width=True)

                    st.dataframe(result_df, use_container_width=True)

                    with st.spinner("Generating AI analysis..."):
                        result = run_assistant(
                            query="Please analyze my current portfolio for diversification and risk.",
                            user_id=current_username,
                            session_id=_graph_session_id,
                            portfolio=st.session_state.portfolio,
                        )
                        st.markdown("### AI Analysis")
                        st.markdown(result["response"])
    else:
        st.info("Add holdings above to see your portfolio dashboard.")

# ================= MARKET TAB =================
with tab_market:
    st.subheader("📋 Market Watchlist")
    st.caption("Live prices for a curated list of well-known stocks and ETFs across sectors.")
    universe = load_stock_universe()
    watchlist_tickers = universe.get("default_watchlist", [])

    if watchlist_tickers:
        if st.button("🔄 Refresh Watchlist"):
            st.cache_data.clear()
        with st.spinner("Loading watchlist quotes..."):
            batch = get_portfolio_quotes(watchlist_tickers)
        name_by_ticker = {s["ticker"]: s["name"] for s in universe.get("stocks", [])}
        sector_by_ticker = {s["ticker"]: s["sector"] for s in universe.get("stocks", [])}
        watch_rows = []
        for tk, q in batch["quotes"].items():
            watch_rows.append({
                "Ticker": tk,
                "Name": name_by_ticker.get(tk, ""),
                "Sector": sector_by_ticker.get(tk, ""),
                "Price": q.get("price"),
                "Change %": q.get("change_percent"),
            })
        if watch_rows:
            st.dataframe(pd.DataFrame(watch_rows), use_container_width=True, hide_index=True)
        if batch["errors"]:
            st.caption(f"Unavailable right now: {', '.join(batch['errors'].keys())}")

    st.divider()
    st.subheader("🔍 Individual Stock Lookup")
    mcol1, mcol2 = st.columns([1, 3])
    lookup_ticker = mcol1.text_input("Ticker", value="AAPL", key="market_ticker").upper().strip()
    period = mcol2.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)

    if st.button("Get Market Data", type="primary"):
        try:
            quote = get_live_quote(lookup_ticker)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Price", f"${quote['price']}", quote["change_percent"])
            c2.metric("Volume", f"{quote.get('volume', 0):,}")
            c3.metric("Day High", quote.get("day_high", "—"))
            c4.metric("Day Low", quote.get("day_low", "—"))

            hist = get_price_history(lookup_ticker, period=period)
            fig = go.Figure(go.Candlestick(
                x=hist.index, open=hist["Open"], high=hist["High"],
                low=hist["Low"], close=hist["Close"],
            ))
            fig.update_layout(title=f"{lookup_ticker} — {period}", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
        except MarketDataError as e:
            st.error(str(e))

with st.sidebar:
    st.header("⚙️ Session")
    if settings.ENABLE_MULTI_USER_AUTH:
        st.text(f"User: {current_display_name} ({current_username})")
        authenticator.logout(button_name="Log out", location="sidebar")
    st.text(f"Session ID: {st.session_state.session_id[:8]}...")
    if st.button("New Session"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.chat_history = []
        st.rerun()
    st.divider()
    st.caption(
        "🎓 **Educational tool.** Not registered investment, tax, or legal "
        "advice. AI-generated content may be inaccurate — verify independently."
    )
    st.divider()
    st.caption(f"LLM: {settings.OPENAI_MODEL}")
    st.caption(f"Vector backend: {settings.VECTOR_BACKEND}")
