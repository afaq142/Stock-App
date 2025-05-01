from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os
import json
from werkzeug.utils import secure_filename
from sqlalchemy.orm import joinedload


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///stock.db'
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'xlsx', 'xls'}
db = SQLAlchemy(app)

# Initialize data cache
data_cache = {}

# Database Models
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

# Helper Functions
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
    
    if not stock_data:
        return None
    
    data = [{
        'date': s.date.strftime('%Y-%m-%d'),
        'open': s.open,
        'high': s.high,
        'low': s.low,
        'close': s.close,
        'volume': s.volume
    } for s in stock_data]
    
    df = pd.DataFrame(data)
    df = calculate_technical_indicators(df)
    
    data_cache[cache_key] = df
    return df

def calculate_technical_indicators(df):
    df = df.copy()
    
    # Convert to numeric
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
    
    # Moving Averages
    df['SMA20'] = df['close'].rolling(20).mean()
    df['SMA50'] = df['close'].rolling(50).mean()
    df['SMA200'] = df['close'].rolling(200).mean()
    
    # Exponential Moving Averages
    df['EMA9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['EMA21'] = df['close'].ewm(span=21, adjust=False).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    
    # Bollinger Bands
    df['BB_middle'] = df['close'].rolling(20).mean()
    df['BB_upper'] = df['BB_middle'] + 2 * df['close'].rolling(20).std()
    df['BB_lower'] = df['BB_middle'] - 2 * df['close'].rolling(20).std()
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    # OBV
    df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    
    # Volume SMA
    df['Volume_SMA20'] = df['volume'].rolling(20).mean()
    
    return df.dropna()

# Analysis Functions
def generate_trading_plan(df):
    latest = df.iloc[-1]
    atr = latest['ATR']
    entry = latest['close']
    stop_loss = latest['low'] - (atr * 1.5)
    risk_per_share = entry - stop_loss
    target = entry + (risk_per_share * 3)
    
    return {
        'entry': entry,
        'stop_loss': stop_loss,
        'target': target,
        'atr_value': atr,
        'risk_reward': 3,
        'target_pct': round(((target / entry) - 1) * 100, 2)
    }

def get_trend_analysis(df):
    latest = df.iloc[-1]
    short_term = "Bullish" if latest['EMA9'] > latest['EMA21'] else "Bearish"
    medium_term = "Bullish" if latest['close'] > latest['SMA50'] else "Bearish"
    long_term = "Bullish" if latest['close'] > latest['SMA200'] else "Bearish"
    
    if all([short_term == "Bullish", medium_term == "Bullish", long_term == "Bullish"]):
        overall = "Strong Bullish"
    elif all([short_term == "Bearish", medium_term == "Bearish", long_term == "Bearish"]):
        overall = "Strong Bearish"
    else:
        overall = "Neutral/Mixed"
    
    return {
        'overall_trend': overall,
        'short_term_trend': short_term,
        'medium_term_trend': medium_term,
        'long_term_trend': long_term
    }

# Routes
@app.route('/')
def index():
    return render_template('index.html')

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
        if df is None or len(df) < 10:
            return render_template('error.html', message=f"Insufficient data for {symbol}")
        
        company = Company.query.filter_by(symbol=symbol).first()
        company_name = company.name if company else symbol
        
        # Prepare data for charts
        dates = df['date'].tolist()
        closes = df['close'].tolist()
        opens = df['open'].tolist()
        highs = df['high'].tolist()
        lows = df['low'].tolist()
        volumes = df['volume'].tolist()
        
        # Prepare indicators
        indicators = {
            'ema9': df['EMA9'].tolist(),
            'ema21': df['EMA21'].tolist(),
            'rsi': df['RSI'].tolist(),
            'macd': df['MACD'].tolist(),
            'macd_signal': df['MACD_signal'].tolist(),
            'macd_hist': df['MACD_hist'].tolist(),
            'macd_hist_colors': ['green' if x >= 0 else 'red' for x in df['MACD_hist']],
            'volume_colors': ['green' if df.iloc[i]['close'] >= df.iloc[i]['open'] else 'red' for i in range(len(df))]
        }
        
        # Generate analysis
        trading_plan = generate_trading_plan(df)
        trend_analysis = get_trend_analysis(df)
        
        # Calculate price change
        latest = df.iloc[-1]
        prev_close = df.iloc[-2]['close'] if len(df) > 1 else latest['close']
        price_change = ((latest['close'] - prev_close) / prev_close) * 100
        
        return render_template('analysis.html',
            company={'name': company_name, 'symbol': symbol},
            dates=json.dumps(dates),
            opens=json.dumps(opens),
            highs=json.dumps(highs),
            lows=json.dumps(lows),
            closes=json.dumps(closes),
            volumes=json.dumps(volumes),
            indicators=indicators,
            trading_plan=trading_plan,
            latest=latest,
            price_change=price_change,
            trend_analysis=trend_analysis
        )
        
    except Exception as e:
        return render_template('error.html', message=str(e))

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