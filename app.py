from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import talib
import os
import json
from werkzeug.utils import secure_filename
from sqlalchemy.orm import joinedload
from datetime import datetime, timedelta
import hashlib


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///stock.db'
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'xlsx', 'xls'}
db = SQLAlchemy(app)

# Initialize data cache
data_cache = {}

# Database Models (same as before)
class Company(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    symbol = db.Column(db.String(10), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    stock_data = db.relationship('StockData', backref='company', lazy=True, order_by='StockData.date.desc()')

class StockData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    company_id = db.Column(db.Integer, db.ForeignKey('company.id'), nullable=False)
    date = db.Column(db.DateTime, nullable=False, index=True)
    open = db.Column(db.Float, nullable=False)
    high = db.Column(db.Float, nullable=False)
    low = db.Column(db.Float, nullable=False)
    close = db.Column(db.Float, nullable=False)
    volume = db.Column(db.BigInteger, nullable=False)

# Helper Functions (same as before)
def convert_volume(volume_str):
    if isinstance(volume_str, (int, float)):
        return int(volume_str)
    volume_str = str(volume_str).upper().replace(',', '')
    if 'M' in volume_str:
        return int(float(volume_str.replace('M', '')) * 1_000_000)
    elif 'K' in volume_str:
        return int(float(volume_str.replace('K', '')) * 1_000)
    return int(float(volume_str))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_stock_data(symbol):
    """Get ALL stock data from database for a symbol"""
    cache_key = f"{symbol}_all"
    if cache_key in data_cache:
        return data_cache[cache_key]
    
    company = Company.query.filter_by(symbol=symbol).first()
    if not company:
        return None
    
    stock_data = StockData.query.filter(
        StockData.company_id == company.id
    ).order_by(StockData.date).all()
    
    if not stock_data or len(stock_data) < 20:
        return None
    
    data = [{
        'date': s.date.strftime('%Y-%m-%d'),
        'open': float(s.open),
        'high': float(s.high),
        'low': float(s.low),
        'close': float(s.close),
        'volume': int(s.volume)
    } for s in stock_data]
    
    df = pd.DataFrame(data)
    df = calculate_technical_indicators(df)
    
    data_cache[cache_key] = df
    return df

def calculate_technical_indicators(df):
    """Calculate all technical indicators"""
    df = df.copy()
    
    closes = np.array(df['close'], dtype=np.float64)
    highs = np.array(df['high'], dtype=np.float64)
    lows = np.array(df['low'], dtype=np.float64)
    volumes = np.array(df['volume'], dtype=np.float64)
    
    # Moving Averages
    df['SMA20'] = talib.SMA(closes, timeperiod=20)
    df['SMA50'] = talib.SMA(closes, timeperiod=50)
    df['SMA200'] = talib.SMA(closes, timeperiod=200)
    
    # Exponential Moving Averages
    df['EMA9'] = talib.EMA(closes, timeperiod=9)
    df['EMA21'] = talib.EMA(closes, timeperiod=21)
    
    # RSI
    df['RSI'] = talib.RSI(closes, timeperiod=14)
    
    # MACD
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(
        closes, fastperiod=12, slowperiod=26, signalperiod=9)
    
    # Bollinger Bands
    df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(
        closes, timeperiod=20, nbdevup=2, nbdevdn=2)
    df['BB_%B'] = (closes - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
    
    # Stochastic Oscillator
    df['STOCH_K'], df['STOCH_D'] = talib.STOCH(
        highs, lows, closes, 
        fastk_period=14, slowk_period=3, 
        slowk_matype=0, slowd_period=3, slowd_matype=0)
    
    # ATR
    df['ATR'] = talib.ATR(highs, lows, closes, timeperiod=14)
    
    # ADX
    df['ADX'] = talib.ADX(highs, lows, closes, timeperiod=14)
    
    # OBV
    df['OBV'] = talib.OBV(closes, volumes.astype(np.float64))
    
    # Volume SMA
    df['Volume_SMA20'] = talib.SMA(volumes, timeperiod=20)
    
    df['trend_strength'] = np.clip(df['ADX'], 0, 100)
    
    return df.dropna()

def generate_trading_plan(df):
    """Generate trading plan with entry, stop loss, and targets"""
    latest = df.iloc[-1]
    atr = latest['ATR']
    entry = latest['close']
    stop_loss = latest['low'] - (atr * 1.5)
    risk_per_share = entry - stop_loss
    target = entry + (risk_per_share * 3)  # 3:1 reward ratio
    
    return {
        'entry': entry,
        'stop_loss': stop_loss,
        'target': target,
        'atr_value': atr,
        'risk_reward': 3,
        'target_pct': round(((target / entry) - 1) * 100, 2)
    }

def get_trend_analysis(df):
    """Determine market trends at different timeframes"""
    latest = df.iloc[-1]
    
    # Short-term trend (EMA9 vs EMA21)
    short_term = "Bullish" if latest['EMA9'] > latest['EMA21'] else "Bearish"
    
    # Medium-term trend (Price vs SMA50)
    medium_term = "Bullish" if latest['close'] > latest['SMA50'] else "Bearish"
    
    # Long-term trend (Price vs SMA200)
    long_term = "Bullish" if latest['close'] > latest['SMA200'] else "Bearish"
    
    # Overall trend (using ADX)
    if latest['ADX'] > 25:  # Strong trend
        if short_term == "Bullish" and medium_term == "Bullish" and long_term == "Bullish":
            overall = "Strong Bullish"
        elif short_term == "Bearish" and medium_term == "Bearish" and long_term == "Bearish":
            overall = "Strong Bearish"
        else:
            overall = "Strong Mixed"
    else:  # Weak trend
        overall = "Weak/Choppy"
    
    return {
        'overall_trend': overall,
        'short_term_trend': short_term,
        'medium_term_trend': medium_term,
        'long_term_trend': long_term
    }

def calculate_fib_levels(df, lookback=180):
    """Calculate Fibonacci retracement levels"""
    recent_data = df.tail(lookback)
    swing_high = recent_data['high'].max()
    swing_low = recent_data['low'].min()
    diff = swing_high - swing_low
    
    levels = {
        '0.0': swing_high,
        '0.236': swing_high - diff * 0.236,
        '0.382': swing_high - diff * 0.382,
        '0.5': swing_high - diff * 0.5,
        '0.618': swing_high - diff * 0.618,
        '0.786': swing_high - diff * 0.786,
        '1.0': swing_low,
    }
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    current_price = df.iloc[-1]['close']
    current_level = None
    
    # Determine between which levels the current price is
    sorted_levels = sorted(levels.items(), key=lambda x: x[1], reverse=True)
    for i in range(len(sorted_levels)-1):
        if sorted_levels[i][1] >= current_price >= sorted_levels[i+1][1]:
            current_level = f"{sorted_levels[i][0]} - {sorted_levels[i+1][0]}"
            break
    
    return {
        'swing_high': swing_high,
        'swing_low': swing_low,
        'levels': levels,
        'current_level': current_level
    }

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    if request.method == 'POST':
        symbol = request.form.get('symbol', 'AAPL').strip().upper()
    else:
        symbol = request.args.get('symbol', 'AAPL').strip().upper()
    
    if not symbol or not symbol.isalpha():
        return render_template('error.html', message="Invalid stock symbol")
    
    try:
        df = get_stock_data(symbol)
        if df is None or len(df) < 20:
            return render_template('error.html', message=f"Insufficient data for {symbol}")
        
        company = Company.query.filter_by(symbol=symbol).first()
        company_name = company.name if company else symbol
        
        trading_plan = generate_trading_plan(df)
        trend_analysis = get_trend_analysis(df)
        fib_levels = calculate_fib_levels(df)
        
        latest = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                 for k, v in df.iloc[-1].to_dict().items()}
        
        prev_close = float(df.iloc[-2]['close']) if len(df) > 1 else float(latest['close'])
        price_change = ((latest['close'] - prev_close) / prev_close) * 100
        
        indicators = {
            'dates': df['date'].tolist(),
            'opens': df['open'].tolist(),
            'highs': df['high'].tolist(),
            'lows': df['low'].tolist(),
            'closes': df['close'].tolist(),
            'volumes': df['volume'].tolist(),
            'ema9': df['EMA9'].tolist(),
            'ema21': df['EMA21'].tolist(),
            'rsi': df['RSI'].tolist(),
            'macd': df['MACD'].tolist(),
            'macd_signal': df['MACD_signal'].tolist(),
            'macd_hist': df['MACD_hist'].tolist(),
            'macd_hist_colors': ['green' if x >= 0 else 'red' for x in df['MACD_hist']],
            'stoch_k': df['STOCH_K'].tolist(),
            'stoch_d': df['STOCH_D'].tolist(),
            'bb_upper': df['BB_upper'].tolist(),
            'bb_middle': df['BB_middle'].tolist(),
            'bb_lower': df['BB_lower'].tolist(),
            'bollinger_percent_b': df['BB_%B'].tolist(),
            'atr': df['ATR'].tolist(),
            'volume_sma': df['Volume_SMA20'].tolist(),
            'volume_colors': ['green' if df.iloc[i]['close'] >= df.iloc[i]['open'] else 'red' for i in range(len(df))],
            'trend_strength': float(df.iloc[-1]['trend_strength']),
            'adx': float(df.iloc[-1]['ADX'])
        }
        
        return render_template('analysis.html',
            company={'name': company_name, 'symbol': symbol},
            indicators=indicators,
            trading_plan=trading_plan,
            latest=latest,
            price_change=float(price_change),
            trend_analysis=trend_analysis,
            fib_levels=fib_levels,
            support_levels=identify_support_resistance(df),
            resistance_levels=identify_support_resistance(df, mode='resistance')
        )
        
    except Exception as e:
        return render_template('error.html', message=str(e))

import numpy as np
import numpy as np
from scipy.signal import argrelextrema
from sklearn.cluster import MeanShift

def identify_support_resistance(df, mode='support', merge_distance=0.015, min_touches=3, 
                              lookback_window=90, recent_weight=0.7):
    
    # Limit analysis to recent data
    recent_df = df.iloc[-lookback_window:] if len(df) > lookback_window else df
    
    # Find swing points using scipy's extrema detection
    if mode == 'support':
        # Find local minima in low prices
        lows = recent_df['low'].values
        min_idx = argrelextrema(lows, np.less, order=3)[0]
        points = lows[min_idx]
    else:
        # Find local maxima in high prices
        highs = recent_df['high'].values
        max_idx = argrelextrema(highs, np.greater, order=3)[0]
        points = highs[max_idx]
    
    if len(points) < min_touches:
        return []
    
    # Cluster points using MeanShift (automatically determines cluster count)
    clustering = MeanShift(bandwidth=merge_distance * np.median(points)).fit(points.reshape(-1, 1))
    clusters = {}
    
    # Process clusters
    for i, label in enumerate(clustering.labels_):
        price = points[i]
        if label not in clusters:
            clusters[label] = {'prices': [], 'timestamps': []}
        clusters[label]['prices'].append(price)
        clusters[label]['timestamps'].append(min_idx[i] if mode == 'support' else max_idx[i])
    
    levels = []
    for cluster in clusters.values():
        touch_count = len(cluster['prices'])
        if touch_count >= min_touches:
            # Weighted average favoring recent touches
            weights = np.linspace(recent_weight, 1-recent_weight, touch_count)
            weighted_prices = np.average(cluster['prices'], weights=weights)
            
            # Determine strength based on touches and recency
            recency_score = np.mean([1 - (len(recent_df) - ts) / len(recent_df) for ts in cluster['timestamps']])
            strength_score = (touch_count * 0.4) + (recency_score * 0.6)
            
            levels.append({
                'price': round(float(weighted_prices), 2),
                'touches': touch_count,
                'strength': 'Strong' if strength_score > 0.7 else 'Moderate',
                'last_touch': min(cluster['timestamps'])
            })
    
    # Sort levels by strength (touches + recency)
    levels.sort(key=lambda x: (-x['touches'], -x['last_touch']))
    
    return levels[:10]  # Return top 10 strongest levels

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/add-company', methods=['GET', 'POST'])
def add_company():
    if request.method == 'POST':
        name = request.form['name']
        symbol = request.form['symbol'].upper()
        
        if Company.query.filter_by(symbol=symbol).first():
            flash('Company with this symbol already exists!', 'danger')
            return redirect(url_for('add_company'))
        
        new_company = Company(name=name, symbol=symbol)
        db.session.add(new_company)
        db.session.commit()
        flash('Company added successfully!', 'success')
        return redirect(url_for('index'))
    return render_template('add_company.html')

@app.route('/add-stock', methods=['GET', 'POST'])
def add_stock():
    companies = Company.query.all()
    if request.method == 'POST':
        if 'excel_file' in request.files:
            file = request.files['excel_file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                try:
                    df = pd.read_excel(filepath)
                    required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                    if not all(col in df.columns for col in required_columns):
                        flash('Excel file missing required columns', 'danger')
                        return redirect(url_for('add_stock'))
                    
                    company_id = request.form['company']
                    new_entries = 0
                    errors = []
                    
                    for index, row in df.iterrows():
                        try:
                            date_obj = pd.to_datetime(row['Date']).date()
                            if date_obj.weekday() >= 5:
                                errors.append(f"Row {index+1}: Weekend date")
                                continue
                            
                            stock = StockData(
                                company_id=company_id,
                                date=date_obj,
                                open=float(row['Open']),
                                high=float(row['High']),
                                low=float(row['Low']),
                                close=float(row['Close']),
                                volume=convert_volume(row['Volume'])
                            )
                            db.session.add(stock)
                            new_entries += 1
                            
                        except Exception as e:
                            errors.append(f"Row {index+1}: {str(e)}")
                            
                    db.session.commit()
                    flash(f'{new_entries} records added successfully', 'success')
                    if errors:
                        flash(f'Errors in {len(errors)} rows: {", ".join(errors)}', 'warning')
                    
                except Exception as e:
                    flash(f'Error processing file: {str(e)}', 'danger')
                
                return redirect(url_for('add_stock'))
        
        # Handle manual entry
        company_id = request.form['company']
        date_str = request.form['date']
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        
        if date_obj.weekday() >= 5:
            flash('Cannot add data for weekends!', 'danger')
            return redirect(url_for('add_stock'))
            
        stock_data = StockData(
            company_id=company_id,
            date=date_obj,
            open=float(request.form['open']),
            high=float(request.form['high']),
            low=float(request.form['low']),
            close=float(request.form['close']),
            volume=int(request.form['volume']))
        db.session.add(stock_data)
        db.session.commit()
        flash('Stock data added successfully!', 'success')
        return redirect(url_for('index'))
    return render_template('add_stock.html', companies=companies)

@app.route('/database')
def view_database():
    companies = Company.query.options(joinedload(Company.stock_data)).all()
    for company in companies:
        company.stock_data = sorted(company.stock_data, key=lambda x: x.date, reverse=True)
    return render_template('database.html', companies=companies)

@app.route('/api/stock-data/<symbol>')
def stock_data_api(symbol):
    company = Company.query.filter_by(symbol=symbol).first()
    if not company:
        return jsonify({'error': 'Company not found'}), 404
    
    stock_data = StockData.query.filter_by(company_id=company.id).order_by(StockData.date).all()
    data = [{
        'date': s.date.strftime('%Y-%m-%d'),
        'open': s.open,
        'high': s.high,
        'low': s.low,
        'close': s.close,
        'volume': s.volume
    } for s in stock_data]
    
    return jsonify(data)

@app.route('/delete-stock/<int:stock_id>', methods=['POST'])
def delete_stock(stock_id):
    stock = StockData.query.get_or_404(stock_id)
    db.session.delete(stock)
    db.session.commit()
    flash('Stock entry deleted successfully', 'success')
    return redirect(url_for('view_database'))

@app.route('/edit-stock/<int:stock_id>', methods=['GET', 'POST'])
def edit_stock(stock_id):
    stock = StockData.query.get_or_404(stock_id)
    if request.method == 'POST':
        try:
            date_str = request.form['date']
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            if date_obj.weekday() >= 5:
                flash('Cannot use weekend dates', 'danger')
                return redirect(url_for('edit_stock', stock_id=stock_id))
            
            stock.date = date_obj
            stock.open = float(request.form['open'])
            stock.high = float(request.form['high'])
            stock.low = float(request.form['low'])
            stock.close = float(request.form['close'])
            stock.volume = int(request.form['volume'])
            
            db.session.commit()
            flash('Stock entry updated successfully', 'success')
            return redirect(url_for('view_database'))
        
        except Exception as e:
            flash(f'Error updating entry: {str(e)}', 'danger')
            return redirect(url_for('edit_stock', stock_id=stock_id))
    
    return render_template('edit_stock.html', stock=stock)

@app.route('/delete-company/<int:company_id>', methods=['POST'])
def delete_company(company_id):
    company = Company.query.get_or_404(company_id)
    StockData.query.filter_by(company_id=company_id).delete()
    db.session.delete(company)
    db.session.commit()
    flash('Company and all associated stock data deleted', 'success')
    return redirect(url_for('view_database'))

def generate_decision_analysis(df, symbol):
    latest = df.iloc[-1]
    prev_close = df.iloc[-2]['close'] if len(df) > 1 else latest['close']
    price_change = ((latest['close'] - prev_close) / prev_close) * 100

    # Indicator Analysis
    indicators = [
        {
            'name': 'RSI (14)',
            'value': f"{latest['RSI']:.2f}",
            'signal': 'Bullish' if latest['RSI'] < 30 else 'Bearish' if latest['RSI'] > 70 else 'Neutral',
            'interpretation': 'Oversold (<30) or Overbought (>70)'
        },
        {
            'name': 'MACD',
            'value': f"{latest['MACD']:.2f} ({latest['MACD_hist']:.2f})",
            'signal': 'Bullish' if latest['MACD_hist'] > 0 else 'Bearish',
            'interpretation': 'Positive histogram suggests bullish momentum'
        },
        {
            'name': '50/200 MA',
            'value': f"{latest['SMA50']:.2f}/{latest['SMA200']:.2f}",
            'signal': 'Golden Cross' if latest['SMA50'] > latest['SMA200'] else 'Death Cross',
            'interpretation': 'Long-term trend indicator'
        },
        {
            'name': 'Bollinger %B',
            'value': f"{latest['BB_%B']:.2f}",
            'signal': 'Overbought' if latest['BB_%B'] > 0.8 else 'Oversold' if latest['BB_%B'] < 0.2 else 'Neutral',
            'interpretation': 'Price relative to Bollinger Bands'
        },
        {
            'name': 'Volume Trend',
            'value': f"{latest['Volume_SMA20']/1e6:.2f}M",
            'signal': 'Bullish' if latest['volume'] > latest['Volume_SMA20'] else 'Bearish',
            'interpretation': 'Volume vs 20-day average'
        },
        {
            'name': 'ADX Trend',
            'value': f"{latest['ADX']:.1f}",
            'signal': 'Strong Trend' if latest['ADX'] > 25 else 'Weak Trend',
            'interpretation': 'Trend strength indicator'
        }
    ]

    # Generate Recommendation
    buy_signals = sum(1 for ind in indicators if ind['signal'] in ['Bullish', 'Golden Cross'])
    sell_signals = sum(1 for ind in indicators if ind['signal'] in ['Bearish', 'Death Cross'])
    
    if buy_signals > sell_signals and latest['ADX'] > 25:
        recommendation = "BUY"
        summary = "Strong bullish signals with confirmed trend strength"
    elif sell_signals > buy_signals and latest['ADX'] > 25:
        recommendation = "SELL"
        summary = "Bearish momentum with strong trend confirmation"
    else:
        recommendation = "HOLD"
        summary = "Mixed signals or weak trend - wait for confirmation"

    return {
        'recommendation': recommendation,
        'summary': summary,
        'current_price': latest['close'],
        'price_change': price_change,
        'trend_strength': latest['trend_strength'],
        'atr': latest['ATR'],
        'indicators': indicators,
        'plan': generate_trading_plan(df)
    }

@app.route('/decision', methods=['GET', 'POST'])
def decision():
    companies = Company.query.all()
    analysis = None
    company = None
    
    if request.method == 'POST':
        symbol = request.form.get('symbol').upper()
        company = Company.query.filter_by(symbol=symbol).first()
        
        if company:
            df = get_stock_data(symbol)
            if df is not None and len(df) > 20:
                analysis = generate_decision_analysis(df, symbol)
                
    return render_template('decision.html', 
                         companies=companies, 
                         analysis=analysis,
                         selected_company=company)






def generate_market_trend_chart(market_data):
    return {
        'data': [{
            'x': [d['symbol'] for d in market_data],
            'y': [d['price'] for d in market_data],
            'type': 'scatter',
            'mode': 'markers+lines',
            'name': 'Price Trend',
            'line': {'color': '#1f77b4'},
            'marker': {
                'size': [np.log(d['volume']/1e3) for d in market_data],
                'color': [d['change'] for d in market_data],
                'colorscale': 'RdYlGn',
                'showscale': True,
                'colorbar': {'title': 'Daily Change (%)'}
            }
        }],
        'layout': {
            'title': 'Real-Time Price Trends with Volume',
            'xaxis': {'title': 'Company Symbol', 'tickangle': 45},
            'yaxis': {'title': 'Price'},
            'hovermode': 'closest'
        }
    }

def calculate_volatility(stock_data):
    if len(stock_data) < 5:
        return 0
    closes = [s.close for s in stock_data[:5]]
    return (max(closes) - min(closes)) / min(closes) * 100

def generate_price_distribution(market_data):
    prices = [d['price'] for d in market_data]
    return {
        'data': [{
            'x': prices,
            'type': 'histogram',
            'name': 'Price Distribution',
            'marker': {'color': '#1f77b4'}
        }],
        'layout': {
            'title': 'Price Distribution Across Market',
            'xaxis': {'title': 'Price'},
            'yaxis': {'title': 'Number of Companies'}
        }
    }

def generate_volume_analysis(market_data):
    symbols = [d['symbol'] for d in market_data]
    volumes = [d['volume'] for d in market_data]
    return {
        'data': [{
            'x': symbols,
            'y': volumes,
            'type': 'bar',
            'name': 'Trading Volume',
            'marker': {'color': '#2ca02c'}
        }],
        'layout': {
            'title': 'Trading Volume by Company',
            'xaxis': {'title': 'Company Symbol'},
            'yaxis': {'title': 'Volume'}
        }
    }

def generate_volatility_analysis(market_data):
    return {
        'data': [{
            'x': [d['symbol'] for d in market_data],
            'y': [d['volatility'] for d in market_data],
            'type': 'scatter',
            'mode': 'markers',
            'marker': {
                'size': [d['volume']/1e6 for d in market_data],
                'color': [d['change'] for d in market_data],
                'colorscale': 'RdYlGn',
                'showscale': True
            }
        }],
        'layout': {
            'title': 'Volatility Analysis',
            'xaxis': {'title': 'Company Symbol'},
            'yaxis': {'title': '5-Day Volatility (%)'},
            'hovermode': 'closest'
        }
    }

@app.route('/market-overview')
def market_overview():
    # Get all companies with recent data
    companies = Company.query.options(joinedload(Company.stock_data)).all()
    
    # Calculate daily performance metrics
    market_data = []
    for company in companies:
        if company.stock_data:
            latest = company.stock_data[0]
            prev_close = company.stock_data[1].close if len(company.stock_data) > 1 else latest.close
            change = ((latest.close - prev_close) / prev_close) * 100
            
            market_data.append({
                'symbol': company.symbol,
                'name': company.name,
                'price': latest.close,
                'change': change,
                'volume': latest.volume,
                'volatility': calculate_volatility(company.stock_data)
            })

    # Generate technical charts data
    plot_data = {
        'price_distribution': generate_price_distribution(market_data),
        'volume_analysis': generate_volume_analysis(market_data),
        'volatility_analysis': generate_volatility_analysis(market_data),
        'market_trend': generate_market_trend_chart(market_data),
        'volume_leaders': sorted(market_data, key=lambda x: x['volume'], reverse=True)[:10],
        'top_gainers': sorted(market_data, key=lambda x: x['change'], reverse=True)[:5],
        'top_losers': sorted(market_data, key=lambda x: x['change'])[:5]
    }
    
    return render_template('market_dashboard.html', 
                         plot_data=plot_data,
                         market_data=market_data)


from datetime import datetime  # Add this at the top of your app.py


@app.route('/swing-trading')
def swing_trading():
    try:
        # Get all companies with sufficient data
        companies = Company.query.options(joinedload(Company.stock_data)).all()
        
        swing_opportunities = []
        
        for company in companies:
            if len(company.stock_data) < 20:  # Need at least 20 days of data
                continue
                
            # Convert stock data to DataFrame
            data = [{
                'date': sd.date,
                'open': sd.open,
                'high': sd.high,
                'low': sd.low,
                'close': sd.close,
                'volume': sd.volume
            } for sd in company.stock_data]
            
            df = pd.DataFrame(data).sort_values('date')
            
            # Calculate technical indicators
            closes = df['close'].values
            highs = df['high'].values
            lows = df['low'].values
            volumes = df['volume'].values
            
            # Calculate indicators
            df['SMA20'] = talib.SMA(closes, timeperiod=20)
            df['SMA50'] = talib.SMA(closes, timeperiod=50)
            df['ADX'] = talib.ADX(highs, lows, closes, timeperiod=14)
            df['RSI'] = talib.RSI(closes, timeperiod=14)
            df['ATR'] = talib.ATR(highs, lows, closes, timeperiod=14)
            
            latest = df.iloc[-1]
            
            # Calculate metrics
            price_change_5d = ((latest['close'] - df.iloc[-5]['close']) / df.iloc[-5]['close']) * 100 if len(df) >= 5 else 0
            volume_pct_change = ((latest['volume'] - df['volume'].rolling(20).mean().iloc[-1]) / 
                               df['volume'].rolling(20).mean().iloc[-1]) * 100 if len(df) >= 20 else 0
            
            # Determine setup
            setup, setup_description = identify_swing_setup(df)
            
            # Generate trading plan
            entry = latest['close']
            stop_loss = max(latest['low'] - (latest['ATR'] * 1.5), 0)
            target = entry + ((entry - stop_loss) * 3)
            risk_reward = round((target - entry) / (entry - stop_loss), 1) if (entry - stop_loss) > 0 else 0
            
            # Trend strength
            trend_strength = min(max(latest['ADX'], 0), 100)
            trend_direction = "Bullish" if latest['close'] > df['SMA50'].iloc[-1] else "Bearish"
            
            opportunity = {
                'symbol': company.symbol,
                'company_name': company.name,
                'current_price': latest['close'],
                'price_change_5d': round(price_change_5d, 2),
                'setup': setup,
                'setup_description': setup_description,
                'entry_low': max(entry * 0.99, 0),
                'entry_high': entry * 1.01,
                'stop_loss': stop_loss,
                'target': target,
                'risk_reward': risk_reward,
                'trend_strength': round(trend_strength),
                'trend_direction': trend_direction,
                'volume': latest['volume'],
                'volume_pct_change': round(volume_pct_change),
                'adx': round(latest['ADX'], 1),
                'rsi': round(latest['RSI'], 1)
            }
            
            swing_opportunities.append(opportunity)
        
        # Categorize opportunities
        top_bullish = [op for op in swing_opportunities if op['trend_direction'] == "Bullish" and op['adx'] > 25]
        top_bearish = [op for op in swing_opportunities if op['trend_direction'] == "Bearish" and op['adx'] > 25]
        
        # Sort by ADX strength
        top_bullish.sort(key=lambda x: -x['adx'])
        top_bearish.sort(key=lambda x: -x['adx'])
        
        return render_template('swing_trading.html',
                            swing_opportunities=swing_opportunities,
                            top_bullish=top_bullish[:5],
                            top_bearish=top_bearish[:5],
                            current_time=datetime.now())
    
    except Exception as e:
        app.logger.error(f"Error in swing_trading: {str(e)}")
        return render_template('error.html', message=f"Swing trading analysis failed: {str(e)}")
    
@app.route('/analyze-swing/<symbol>')
def analyze_swing(symbol):
    # Get the company data
    company = Company.query.filter_by(symbol=symbol).first_or_404()
    
    # Convert stock data to DataFrame
    data = [{
        'date': sd.date,
        'open': sd.open,
        'high': sd.high,
        'low': sd.low,
        'close': sd.close,
        'volume': sd.volume
    } for sd in company.stock_data]
    
    df = pd.DataFrame(data).sort_values('date')
    
    # Calculate technical indicators
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    
    df['SMA20'] = talib.SMA(closes, timeperiod=20)
    df['SMA50'] = talib.SMA(closes, timeperiod=50)
    df['ADX'] = talib.ADX(highs, lows, closes, timeperiod=14)
    df['ATR'] = talib.ATR(highs, lows, closes, timeperiod=14)
    
    latest = df.iloc[-1]
    
    # Generate trading plan
    entry = latest['close']
    stop_loss = latest['low'] - (latest['ATR'] * 1.5)
    target = entry + ((entry - stop_loss) * 3)
    
    return jsonify({
        'symbol': company.symbol,
        'name': company.name,
        'price': latest['close'],
        'entry': entry,
        'stop_loss': stop_loss,
        'target': target,
        'atr': latest['ATR'],
        'adx': latest['ADX'],
        'sma20': df['SMA20'].iloc[-1],
        'sma50': df['SMA50'].iloc[-1]
    })

def identify_swing_setup(df):
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Check for breakout
    if latest['close'] > df['high'].rolling(20).max().iloc[-2]:  # 20-day high breakout
        return 'breakout', '20-day High Breakout'
    elif latest['close'] > latest['SMA50'] and prev['close'] <= prev['SMA50']:  # Crossing 50MA
        return 'breakout', '50MA Cross'
    
    # Check for pullback
    elif latest['close'] > latest['SMA50'] and latest['low'] <= latest['SMA20']:  # Pullback to 20MA
        return 'pullback', 'To 20MA'
    elif (latest['close'] > latest['SMA50'] and 
          latest['RSI'] < 40 and 
          latest['close'] < prev['close']):  # Pullback with RSI < 40
        return 'pullback', 'RSI < 40'
    
    # Default case
    return 'neutral', 'Watching'



@app.route('/plain-analysis', methods=['GET', 'POST'])
def plain_analysis():
    # Get all companies for dropdown
    companies = Company.query.order_by(Company.symbol).all()
    
    if request.method == 'POST':
        symbol = request.form.get('symbol', 'AAPL').upper()
        return redirect(url_for('generate_analysis', symbol=symbol))
    
    return render_template('plain_analysis.html', 
                         companies=companies,
                         current_time=datetime.now())

@app.route('/generate-analysis/<symbol>')
def generate_analysis(symbol):
    try:
        company = Company.query.filter_by(symbol=symbol.upper()).first_or_404()
        df = get_stock_data(symbol)
        
        if df is None or len(df) < 20:
            return render_template('error.html', message=f"Not enough data for {symbol}")
        
        latest = df.iloc[-1]
        prev_close = df.iloc[-2]['close'] if len(df) > 1 else latest['close']
        price_change_5d = ((latest['close'] - df.iloc[-5]['close']) / df.iloc[-5]['close']) * 100 if len(df) >= 5 else 0
        
        # Generate analysis components
        trading_plan = generate_trading_plan(df)
        recommendation, recommendation_explanation = generate_recommendation(df)
        trend_analysis_text, short_term_trend, medium_term_trend, long_term_trend = generate_trend_analysis(df)
        market_context_text = generate_market_context(df)
        final_thoughts = generate_final_thoughts(df)
        
        return render_template('analysis_result.html',
            company=company,
            latest=latest,
            price_change_5d=price_change_5d,
            trading_plan=trading_plan,
            recommendation=recommendation,
            recommendation_explanation=recommendation_explanation,
            trend_analysis_text=trend_analysis_text,
            short_term_trend=short_term_trend,
            medium_term_trend=medium_term_trend,
            long_term_trend=long_term_trend,
            support_levels=identify_support_resistance(df),
            resistance_levels=identify_support_resistance(df, mode='resistance'),
            market_context_text=market_context_text,
            final_thoughts=final_thoughts,
            current_time=datetime.now()
        )
        
    except Exception as e:
        return render_template('error.html', message=str(e))

# Helper functions (unchanged from your original)
def generate_recommendation(df):
    latest = df.iloc[-1]
    
    if latest['ADX'] > 25:  # Strong trend
        if latest['close'] > latest['SMA50'] and latest['close'] > latest['SMA200']:
            return "BUY", "Strong uptrend detected with good momentum. The stock is above both 50-day and 200-day averages, suggesting bullish conditions."
        else:
            return "SELL", "Strong downtrend detected. The stock is below key moving averages, suggesting bearish conditions."
    else:
        return "HOLD", "The market is currently range-bound with no clear trend. Wait for stronger momentum before taking a position."

def generate_trend_analysis(df):
    latest = df.iloc[-1]
    
    text = ""
    
    # Short-term
    if latest['EMA9'] > latest['EMA21']:
        short_term = "Bullish (9-day EMA above 21-day EMA)"
    else:
        short_term = "Bearish (9-day EMA below 21-day EMA)"
    
    # Medium-term
    if latest['close'] > latest['SMA50']:
        medium_term = "Bullish (Price above 50-day average)"
    else:
        medium_term = "Bearish (Price below 50-day average)"
    
    # Long-term
    if latest['close'] > latest['SMA200']:
        long_term = "Bullish (Price above 200-day average)"
    else:
        long_term = "Bearish (Price below 200-day average)"
    
    text = f"The overall trend strength is {'strong' if latest['ADX'] > 25 else 'weak'} (ADX at {latest['ADX']:.1f}). "
    
    return text, short_term, medium_term, long_term

def generate_market_context(df):
    latest = df.iloc[-1]
    
    if latest['volume'] > latest['Volume_SMA20'] * 1.5:
        return "High trading volume recently indicates strong investor interest in this stock."
    elif latest['volume'] < latest['Volume_SMA20'] * 0.8:
        return "Below-average trading volume suggests cautious market participation."
    else:
        return "Normal trading volume patterns observed, with no exceptional buying or selling pressure."

def generate_final_thoughts(df):
    latest = df.iloc[-1]
    
    thoughts = []
    
    if latest['RSI'] > 70:
        thoughts.append("The stock appears overbought on RSI, suggesting potential for a pullback.")
    elif latest['RSI'] < 30:
        thoughts.append("The stock appears oversold on RSI, suggesting potential for a bounce.")
        
    if latest['MACD_hist'] > 0:
        thoughts.append("Positive MACD histogram shows upward momentum is increasing.")
    else:
        thoughts.append("Negative MACD histogram shows downward momentum is increasing.")
        
    if len(thoughts) == 0:
        thoughts.append("No strong contrarian signals detected - the current trend may continue.")
        
    return " ".join(thoughts)

def create_tables():
    with app.app_context():
        db.create_all()

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)