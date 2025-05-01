from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os
from werkzeug.utils import secure_filename
from sqlalchemy.orm import joinedload

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///stock.db'
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'xlsx', 'xls'}
db = SQLAlchemy(app)

class Company(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    symbol = db.Column(db.String(10), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    stock_data = db.relationship('StockData', backref='company', lazy='select', order_by='StockData.date.desc()')

class StockData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    company_id = db.Column(db.Integer, db.ForeignKey('company.id'), nullable=False)
    date = db.Column(db.DateTime, nullable=False, index=True)
    open = db.Column(db.Float, nullable=False)
    high = db.Column(db.Float, nullable=False)
    low = db.Column(db.Float, nullable=False)
    close = db.Column(db.Float, nullable=False)
    volume = db.Column(db.Integer, nullable=False)
    
    def __repr__(self):
        return f'<StockData {self.date} {self.close}>'

# Technical Indicators Calculations
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series, slow=26, fast=12, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calculate_bollinger_bands(series, window=20, num_std=2):
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    return sma + (std * num_std), sma - (std * num_std)

def calculate_support_resistance(df, window=14):
    recent = df[-window:]
    support = recent['low'].min()
    resistance = recent['high'].max()
    return support, resistance

def calculate_fib_levels(df, window=30):
    recent = df[-window:]
    high = recent['high'].max()
    low = recent['low'].min()
    diff = high - low
    return {
        '23.6%': high - diff * 0.236,
        '38.2%': high - diff * 0.382,
        '61.8%': high - diff * 0.618
    }

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    symbol = request.form['symbol'].upper()
    company = Company.query.filter_by(symbol=symbol).first()
    if not company:
        flash('Company not found!', 'danger')
        return redirect(url_for('index'))
    
    stock_data = StockData.query.filter_by(company_id=company.id).order_by(StockData.date.desc()).all()
    if not stock_data:
        flash('No stock data available for this company!', 'warning')
        return redirect(url_for('index'))
    
    df = pd.DataFrame([{
        'date': s.date,
        'open': s.open,
        'high': s.high,
        'low': s.low,
        'close': s.close,
        'volume': s.volume
    } for s in stock_data])
    
    # Calculate technical indicators
    df['SMA20'] = df['close'].rolling(20).mean()
    df['SMA50'] = df['close'].rolling(50).mean()
    df['RSI'] = calculate_rsi(df['close'])
    df['MACD'], df['Signal'] = calculate_macd(df['close'])
    df['ATR'] = calculate_atr(df, 14)
    df['Upper_Bollinger'], df['Lower_Bollinger'] = calculate_bollinger_bands(df['close'])
    
    latest = df.iloc[-1]
    support, resistance = calculate_support_resistance(df)
    fib_levels = calculate_fib_levels(df)
    
    return render_template('analysis.html', 
                         company=company,
                         stock_dates=[d.date.strftime('%Y-%m-%d') for d in stock_data],
                         open_prices=[d.open for d in stock_data],
                         high_prices=[d.high for d in stock_data],
                         low_prices=[d.low for d in stock_data],
                         close_prices=[d.close for d in stock_data],
                         latest=latest.to_dict(),
                         support=support,
                         resistance=resistance,
                         fib_levels=fib_levels,
                         indicators=df.iloc[-1].to_dict())

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
                                volume=int(float(str(row['Volume']).replace(',', ''))))
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
    companies = Company.query.options(db.joinedload(Company.stock_data)).all()
    for company in companies:
        company.stock_data = sorted(company.stock_data, key=lambda x: x.date, reverse=True)
    return render_template('database.html', companies=companies)

@app.route('/api/stock-data/<symbol>')
def stock_data_api(symbol):
    company = Company.query.filter_by(symbol=symbol).first()
    if not company:
        return jsonify({'error': 'Company not found'}), 404
    
    stock_data = company.stock_data.order_by(StockData.date).all()
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