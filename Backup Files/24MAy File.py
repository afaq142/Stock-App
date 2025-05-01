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

def get_stock_data(symbol, days=365):
    """Get stock data from database and calculate technical indicators using TA-Lib"""
    cache_key = f"{symbol}_{days}"
    if cache_key in data_cache:
        return data_cache[cache_key]
    
    company = Company.query.filter_by(symbol=symbol).first()
    if not company:
        return None
    
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    
    stock_data = StockData.query.filter(
        StockData.company_id == company.id,
        StockData.date >= start_date,
        StockData.date <= end_date
    ).order_by(StockData.date).all()
    
    if not stock_data or len(stock_data) < 20:  # Minimum data points
        return None
    
    # Convert to DataFrame
    data = [{
        'date': s.date.strftime('%Y-%m-%d'),
        'open': float(s.open),
        'high': float(s.high),
        'low': float(s.low),
        'close': float(s.close),
        'volume': int(s.volume)
    } for s in stock_data]
    
    df = pd.DataFrame(data)
    
    # Calculate indicators using TA-Lib
    df = calculate_technical_indicators(df)
    
    data_cache[cache_key] = df
    return df

def calculate_technical_indicators(df):
    """Calculate all technical indicators using TA-Lib with proper type conversion"""
    df = df.copy()
    
    # Convert to numpy arrays with float64 dtype
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
    
    # ADX (for trend strength)
    df['ADX'] = talib.ADX(highs, lows, closes, timeperiod=14)
    
    # OBV
    df['OBV'] = talib.OBV(closes, volumes.astype(np.float64))  # Ensure volume is float64
    
    # Volume SMA
    df['Volume_SMA20'] = talib.SMA(volumes, timeperiod=20)
    
    # Calculate trend strength (0-100)
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
        
        # Convert all numeric columns to float to ensure JSON serialization
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].astype(float)
        
        company = Company.query.filter_by(symbol=symbol).first()
        company_name = company.name if company else symbol
        
        # Generate analysis
        trading_plan = generate_trading_plan(df)
        trend_analysis = get_trend_analysis(df)
        fib_levels = calculate_fib_levels(df)
        
        # Prepare data for template
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

def identify_support_resistance(df, mode='support', tolerance=0.02, min_touches=3):
    """Identify support/resistance levels using swing points and clustering"""
    levels = []
    if mode == 'support':
        # Find swing lows
        points = df[(df['low'] < df['low'].shift(1)) & (df['low'] < df['low'].shift(-1))]['low'].tolist()
    else:
        # Find swing highs
        points = df[(df['high'] > df['high'].shift(1)) & (df['high'] > df['high'].shift(-1))]['high'].tolist()

    # Cluster nearby levels
    clusters = []
    for point in points:
        found = False
        for cluster in clusters:
            if abs(point - cluster['mean']) / cluster['mean'] <= tolerance:
                cluster['points'].append(point)
                cluster['mean'] = np.mean(cluster['points'])
                found = True
                break
        if not found:
            clusters.append({'mean': point, 'points': [point]})

    # Filter and format levels
    for cluster in clusters:
        touches = len(cluster['points'])
        if touches >= min_touches:
            strength = "Strong" if touches > 5 else "Moderate"
            levels.append({
                'price': round(float(cluster['mean']), 2),
                'touches': touches,
                'strength': strength
            })
    
    return sorted(levels, key=lambda x: x['price'], reverse=(mode == 'resistance'))

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

def create_tables():
    with app.app_context():
        db.create_all()

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)