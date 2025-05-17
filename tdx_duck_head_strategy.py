import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
from pytdx.hq import TdxHq_API
from pytdx.exhq import TdxExHq_API

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TdxDuckHeadStrategy:
    def __init__(self):
        """初始化通达信接口和策略参数"""
        self.api = TdxHq_API()
        self.ex_api = TdxExHq_API()
        self.connected = False
        
        # 老鸭头策略参数
        self.short_ma = 5  # 短期均线
        self.medium_ma = 10  # 中期均线
        self.long_ma = 60  # 长期均线
        self.buy_threshold = 0.03  # 买入阈值
        self.sell_threshold = -0.02  # 卖出阈值
        
    def connect(self):
        """连接通达信行情服务器"""
        try:
            # 尝试连接深圳服务器
            if self.api.connect('119.147.212.81', 7709):
                logger.info("成功连接通达信深圳行情服务器")
                self.connected = True
                return True
            else:
                # 尝试连接上海服务器
                if self.api.connect('101.227.73.20', 7709):
                    logger.info("成功连接通达信上海行情服务器")
                    self.connected = True
                    return True
                else:
                    logger.error("无法连接通达信行情服务器")
                    return False
        except Exception as e:
            logger.error(f"连接服务器时出错: {e}")
            return False
    
    def disconnect(self):
        """断开与通达信服务器的连接"""
        if self.connected and self.api:
            self.api.disconnect()
            self.connected = False
            logger.info("已断开与通达信服务器的连接")
    
    def get_stock_list(self, market=0):
        """
        获取指定市场的股票列表
        
        参数:
            market (int): 市场代码，0-深圳，1-上海
            
        返回:
            list: 股票列表，每个元素为字典，包含股票代码和名称
        """
        if not self.connected:
            if not self.connect():
                return []
                
        stock_list = []
        for i in range(1, 100):  # 最多尝试100页
            stocks = self.api.get_security_list(market, (i-1)*100)
            if not stocks or len(stocks) == 0:
                break
                
            for stock in stocks:
                # 只保留A股
                if (market == 0 and stock['code'].startswith(('00', '30'))) or \
                   (market == 1 and stock['code'].startswith(('60', '688'))):
                    stock_list.append({
                        'code': stock['code'],
                        'name': stock['name']
                    })
        
        return stock_list
    
    def get_all_stocks(self):
        """
        获取沪深A股所有股票
        
        返回:
            list: 股票列表，每个元素为字典，包含股票代码和名称
        """
        sz_stocks = self.get_stock_list(0)  # 深圳市场
        sh_stocks = self.get_stock_list(1)  # 上海市场
        
        return sz_stocks + sh_stocks
    
    def get_stock_data(self, stock_code, start_date=None, end_date=None, period='day'):
        """
        获取股票历史数据
        
        参数:
            stock_code (str): 股票代码，如'000001'
            start_date (str): 开始日期，格式'YYYY-MM-DD'
            end_date (str): 结束日期，格式'YYYY-MM-DD'
            period (str): 周期，'day'（日线）或'min'（分钟线）
            
        返回:
            pd.DataFrame: 包含股票历史数据的DataFrame
        """
        if not self.connected:
            if not self.connect():
                return None
        
        # 确定市场代码
        market = 0 if stock_code.startswith(('0', '3')) else 1  # 0: 深圳，1: 上海
        
        # 如果未指定日期，获取最近300天的数据
        if not start_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=300)).strftime('%Y-%m-%d')
        
        # 转换日期格式
        start_datetime = datetime.strptime(start_date, '%Y-%m-%d')
        end_datetime = datetime.strptime(end_date, '%Y-%m-%d')
        days_diff = (end_datetime - start_datetime).days
        
        # 计算需要获取的K线数量
        if period == 'day':
            # 日线数据，大约250个交易日/年
            total_days = min(days_diff + 1, 800)  # 限制最大获取数量
            max_try = (total_days // 800) + 1
            all_data = []
            
            for i in range(max_try):
                start = i * 800
                data = self.api.get_security_bars(9, market, stock_code, start, 800)
                if data:
                    all_data.extend(data)
                else:
                    break
            
            if not all_data:
                logger.warning(f"未能获取到股票 {stock_code} 的数据")
                return None
                
            # 转换为DataFrame
            df = pd.DataFrame(all_data)
            df['date'] = pd.to_datetime(df['datetime']).dt.strftime('%Y-%m-%d')
            df = df.rename(columns={
                'open': 'open', 'high': 'high', 'low': 'low', 
                'close': 'close', 'vol': 'volume', 'amount': 'amount'
            })
            df = df[['date', 'open', 'high', 'low', 'close', 'volume', 'amount']]
            df = df.sort_values('date')
            
            # 过滤指定日期范围内的数据
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            df = df.reset_index(drop=True)
            
            return df
        
        elif period == 'min':
            # 分钟线数据
            # 注意：通达信分钟线接口每次最多获取240条数据
            logger.warning("分钟线数据获取功能尚未完全实现")
            return None
        
        else:
            logger.error(f"不支持的周期类型: {period}")
            return None
    
    def identify_duck_head_pattern(self, df):
        """
        识别老鸭头形态
        
        参数:
            df (pd.DataFrame): 包含股票历史数据的DataFrame
            
        返回:
            bool: 是否符合老鸭头形态
            dict: 形态特征描述
        """
        if df is None or len(df) < self.long_ma * 2:
            return False, {}
            
        # 计算均线
        df['MA5'] = df['close'].rolling(window=self.short_ma).mean()
        df['MA10'] = df['close'].rolling(window=self.medium_ma).mean()
        df['MA60'] = df['close'].rolling(window=self.long_ma).mean()
        
        # 计算MACD指标
        df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['DIF'] = df['EMA12'] - df['EMA26']
        df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
        df['MACD'] = 2 * (df['DIF'] - df['DEA'])
        
        # 老鸭头形态特征识别
        # 1. 均线多头排列
        df['bullish_ma'] = (df['MA5'] > df['MA10']) & (df['MA10'] > df['MA60']) & (df['MA5'] > df['MA5'].shift(1))
        
        # 2. 股价经过一波上涨后形成头部（高点）
        window_size = 20
        df['local_max'] = df['high'] >= df['high'].rolling(window=window_size, center=True).max()
        
        # 3. 头部形成后股价回落，MA5下穿MA10形成死叉
        df['ma_cross_down'] = (df['MA5'] < df['MA10']) & (df['MA5'].shift(1) > df['MA10'].shift(1))
        
        # 4. 回落过程中成交量萎缩
        df['volume_shrink'] = df['volume'] < df['volume'].rolling(window=5).mean() * 0.7
        
        # 5. 股价在MA60附近获得支撑
        df['near_ma60'] = (df['low'] >= df['MA60'] * 0.95) & (df['low'] <= df['MA60'] * 1.05)
        
        # 6. 之后MA5上穿MA10形成金叉
        df['ma_cross_up'] = (df['MA5'] > df['MA10']) & (df['MA5'].shift(1) < df['MA10'].shift(1))
        
        # 7. MACD指标在0轴上方形成鸭子嘴形态
        df['macd_duck_mouth'] = (df['DIF'] > 0) & (df['DEA'] > 0) & (df['DIF'] > df['DEA']) & \
                                 (df['DIF'].shift(1) < df['DEA'].shift(1))
        
        # 寻找老鸭头形态
        for i in range(len(df) - 30, len(df)):  # 只在最近30个交易日内寻找
            if not df.iloc[i]['bullish_ma']:
                continue
                
            # 寻找头部
            head_index = None
            for j in range(i-30, i):
                if df.iloc[j]['local_max']:
                    head_index = j
                    break
                    
            if head_index is None:
                continue
                
            # 寻找死叉点
            cross_down_index = None
            for j in range(head_index+1, i):
                if df.iloc[j]['ma_cross_down'] and df.iloc[j]['volume_shrink']:
                    cross_down_index = j
                    break
                    
            if cross_down_index is None:
                continue
                
            # 寻找支撑点
            support_index = None
            for j in range(cross_down_index+1, i):
                if df.iloc[j]['near_ma60']:
                    support_index = j
                    break
                    
            if support_index is None:
                continue
                
            # 寻找金叉点
            cross_up_index = None
            for j in range(support_index+1, i):
                if df.iloc[j]['ma_cross_up'] and df.iloc[j]['macd_duck_mouth']:
                    cross_up_index = j
                    break
                    
            if cross_up_index is None:
                continue
                
            # 如果找到了完整的老鸭头形态
            return True, {
                'head_date': df.iloc[head_index]['date'],
                'cross_down_date': df.iloc[cross_down_index]['date'],
                'support_date': df.iloc[support_index]['date'],
                'cross_up_date': df.iloc[cross_up_index]['date'],
                'current_date': df.iloc[i]['date'],
                'current_price': df.iloc[i]['close'],
                'price_at_head': df.iloc[head_index]['high'],
                'price_at_support': df.iloc[support_index]['low'],
                'volume_ratio': df.iloc[cross_down_index]['volume'] / df.iloc[head_index]['volume'],
                'head_position': head_index,
                'cross_down_position': cross_down_index,
                'support_position': support_index,
                'cross_up_position': cross_up_index
            }
            
        return False, {}
    
    def select_stocks_by_duck_head(self, stock_list=None, industry=None):
        """
        根据老鸭头策略筛选股票
        
        参数:
            stock_list (list): 待筛选的股票列表，格式为['000001', '000002', ...]
            industry (str): 行业名称，如'金融'、'医药'等，若指定则从该行业筛选
            
        返回:
            list: 符合老鸭头形态的股票列表
        """
        if not self.connected:
            if not self.connect():
                return []
                
        # 如果未指定股票列表，获取沪深A股所有股票
        if not stock_list:
            if industry:
                logger.info(f"从行业 '{industry}' 获取股票列表")
                # 这里需要调用通达信的行业分类接口
                # 简化处理，实际应用中需要实现
                stock_list = []
            else:
                logger.info("获取沪深A股所有股票列表")
                all_stocks = self.get_all_stocks()
                stock_list = [stock['code'] for stock in all_stocks]
                logger.info(f"共获取到 {len(stock_list)} 只股票")
        
        # 筛选符合老鸭头形态的股票
        selected_stocks = []
        total = len(stock_list)
        
        for i, stock_code in enumerate(stock_list):
            if i % 50 == 0:
                logger.info(f"正在处理第 {i}/{total} 只股票")
                
            try:
                # 获取股票数据
                df = self.get_stock_data(stock_code)
                
                # 识别老鸭头形态
                is_duck_head, pattern_info = self.identify_duck_head_pattern(df)
                
                if is_duck_head:
                    stock_name = self.api.get_security_name(stock_code)
                    pattern_info['stock_code'] = stock_code
                    pattern_info['stock_name'] = stock_name
                    selected_stocks.append(pattern_info)
                    logger.info(f"股票 {stock_code} ({stock_name}) 符合老鸭头形态")
                    
            except Exception as e:
                logger.error(f"处理股票 {stock_code} 时出错: {e}")
                continue
                
        return selected_stocks
    
    def plot_duck_head_pattern(self, stock_code, df, pattern_info):
        """
        绘制老鸭头形态图表
        
        参数:
            stock_code (str): 股票代码
            df (pd.DataFrame): 股票数据
            pattern_info (dict): 形态特征描述
        """
        if df is None or not pattern_info:
            return
            
        plt.figure(figsize=(14, 8))
        
        # 绘制K线图
        plt.subplot(2, 1, 1)
        plt.plot(df['date'], df['close'], label='收盘价')
        plt.plot(df['date'], df['MA5'], label='MA5')
        plt.plot(df['date'], df['MA10'], label='MA10')
        plt.plot(df['date'], df['MA60'], label='MA60')
        
        # 标记关键点
        head_idx = pattern_info['head_position']
        cross_down_idx = pattern_info['cross_down_position']
        support_idx = pattern_info['support_position']
        cross_up_idx = pattern_info['cross_up_position']
        
        plt.scatter(df.iloc[head_idx]['date'], df.iloc[head_idx]['high'], color='red', s=100, label='头部')
        plt.scatter(df.iloc[cross_down_idx]['date'], df.iloc[cross_down_idx]['close'], color='purple', s=100, label='死叉')
        plt.scatter(df.iloc[support_idx]['date'], df.iloc[support_idx]['low'], color='green', s=100, label='支撑')
        plt.scatter(df.iloc[cross_up_idx]['date'], df.iloc[cross_up_idx]['close'], color='blue', s=100, label='金叉')
        
        plt.title(f'{stock_code} 老鸭头形态')
        plt.legend()
        plt.grid(True)
        
        # 绘制MACD指标
        plt.subplot(2, 1, 2)
        plt.plot(df['date'], df['DIF'], label='DIF')
        plt.plot(df['date'], df['DEA'], label='DEA')
        plt.bar(df['date'], df['MACD'], label='MACD', color=['red' if x > 0 else 'green' for x in df['MACD']])
        
        plt.axhline(y=0, color='black', linestyle='-')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

# 使用示例
if __name__ == "__main__":
    strategy = TdxDuckHeadStrategy()
    
    try:
        # 连接通达信服务器
        if strategy.connect():
            # 获取所有A股股票
            all_stocks = strategy.get_all_stocks()
            print(f"获取到 {len(all_stocks)} 只A股股票")
            
            # 显示前10只股票
            print("前10只股票:")
            for i, stock in enumerate(all_stocks[:10]):
                print(f"{i+1}. {stock['code']} {stock['name']}")
            
            # 选择部分股票进行分析（例如前50只）
            sample_stocks = [stock['code'] for stock in all_stocks[:50]]
            
            # 使用老鸭头策略筛选股票
            selected_stocks = strategy.select_stocks_by_duck_head(sample_stocks)
            print(f"共筛选出 {len(selected_stocks)} 只符合老鸭头形态的股票")
            
            # 显示筛选结果
            if selected_stocks:
                print("\n符合老鸭头形态的股票:")
                for i, stock in enumerate(selected_stocks):
                    print(f"{i+1}. {stock['stock_code']} {stock['stock_name']}")
                    print(f"   头部日期: {stock['head_date']}, 价格: {stock['price_at_head']}")
                    print(f"   死叉日期: {stock['cross_down_date']}")
                    print(f"   支撑日期: {stock['support_date']}, 价格: {stock['price_at_support']}")
                    print(f"   金叉日期: {stock['cross_up_date']}")
                    print(f"   当前日期: {stock['current_date']}, 价格: {stock['current_price']}")
                    print(f"   成交量萎缩比例: {stock['volume_ratio']:.2f}")
                    print("-" * 50)
                
                # 绘制第一只符合条件的股票图表
                if len(selected_stocks) > 0:
                    first_stock = selected_stocks[0]
                    stock_code = first_stock['stock_code']
                    df = strategy.get_stock_data(stock_code)
                    strategy.plot_duck_head_pattern(stock_code, df, first_stock)
            else:
                print("没有找到符合老鸭头形态的股票")
                
    finally:
        # 断开连接
        strategy.disconnect()    