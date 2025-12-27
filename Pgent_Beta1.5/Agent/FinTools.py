import pandas as pd
import numpy as np
import akshare as ak
import yfinance as yf
import ta
from typing import Optional, List, Dict
from arch import arch_model
from scipy.stats import norm
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from langchain_core.tools import tool

@tool
def recommend_default_stocks(
    risk_preference: str = "稳健",
    market: str = "A股",
    style: Optional[str] = None
) -> list:
    """
    根据用户风险偏好/市场偏好推荐默认股票（用户未指定时调用）
    """
    STOCK_POOL = {
    "A股": {
        "蓝筹股": ["601318.SH", "600036.SH", "000858.SZ"],
        "成长股": ["300750.SZ", "600570.SH", "002594.SZ"],
        "价值股": ["601689.SH", "000001.SZ", "601899.SH"],
        "低风险": ["600000.SH", "601988.SH", "000061.SZ"],
        "科技龙头": ["600570.SH", "300750.SZ", "002594.SZ"]  # 补充A股科技龙头分类
    },
    "美股": {
        "科技龙头": ["AAPL", "MSFT", "GOOGL"],
        "消费龙头": ["AMZN", "MCD", "COST"],
        "高成长": ["NVDA", "TSLA", "META"]
    },
    "港股": {
        "互联网": ["0700.HK", "9988.HK", "9618.HK"],
        "金融": ["0005.HK", "0939.HK", "3988.HK"]
    },
    "风险偏好": {
        "保守": ["601988.SH", "600000.SH", "000061.SZ"],
        "稳健": ["601318.SH", "600036.SH", "000858.SZ"],
        "激进": ["300750.SZ", "600570.SH", "NVDA"]
    }
}
    # 优先按风险偏好推荐
    if risk_preference in STOCK_POOL["风险偏好"]:
        default_stocks = STOCK_POOL["风险偏好"][risk_preference]
    # 其次按市场+风格推荐
    elif style and market in STOCK_POOL and style in STOCK_POOL[market]:
        default_stocks = STOCK_POOL[market][style]
    # 兜底：默认A股稳健型
    else:
        default_stocks = STOCK_POOL["风险偏好"]["稳健"]
    
    return default_stocks[:3]


def convert_date_to_8digit(date_str: str) -> str:
    """将 YYYY-MM-DD 转换为 YYYYMMDD（适配AKShare参数要求）"""
    if not date_str:
        return pd.Timestamp.now().strftime("%Y%m%d")
    return date_str.replace("-", "")

def get_stock_data(
    ts_code: str, 
    start_date: str = "2022-01-01", 
    end_date: str = None,
    source: str = "akshare" 
) -> pd.DataFrame:
    """
    获取股票历史数据（仅支持AKShare/Yahoo Finance，禁用Tushare）
    :param ts_code: 股票代码（A股：600519.SH/000001.SZ；美股：AAPL；港股：0700.HK）
    :param start_date: 开始日期（格式：2022-01-01）
    :param end_date: 结束日期（默认当前）
    :param source: 数据源（akshare/yfinance）
    :return: 包含日期/开盘/最高/最低/收盘/成交量的DataFrame
    """
    try:
        # 统一处理股票代码（剥离.SH/.SZ/.HK后缀）
        symbol = ts_code.split(".")[0] if "." in ts_code else ts_code
        
        if source == "akshare":
            if ts_code.endswith((".SH", ".SZ")):
                # A股（适配AKShare 1.17.95）
                df = ak.stock_zh_a_hist(
                    symbol=symbol, 
                    period="daily", 
                    start_date=convert_date_to_8digit(start_date),
                    end_date=convert_date_to_8digit(end_date),
                    adjust="qfq"  # 前复权
                )
                # AKShare A股列名是中文：日期/开盘/最高/最低/收盘/成交量
                df["date"] = pd.to_datetime(df["日期"])
                df = df.rename(columns={
                    "开盘": "开盘", "最高": "最高", "最低": "最低", 
                    "收盘": "收盘", "成交量": "成交量"
                }).drop("日期", axis=1)
            
            elif ts_code.endswith(".HK"):
                # 港股
                df = ak.stock_hk_hist(
                    symbol=symbol, 
                    start_date=start_date, 
                    end_date=end_date if end_date else pd.Timestamp.now().strftime("%Y-%m-%d")
                )
                df["date"] = pd.to_datetime(df["日期"])
                df = df.rename(columns={
                    "开盘价": "开盘", "最高价": "最高", "最低价": "最低", 
                    "收盘价": "收盘", "成交量": "成交量"
                }).drop("日期", axis=1)
            
            else:
                # 美股
                df = ak.stock_us_hist(
                    symbol=symbol, 
                    start_date=start_date, 
                    end_date=end_date if end_date else pd.Timestamp.now().strftime("%Y-%m-%d")
                )
                df["date"] = pd.to_datetime(df["日期"])
                df = df.rename(columns={
                    "开盘价": "开盘", "最高价": "最高", "最低价": "最低", 
                    "收盘价": "收盘", "成交量": "成交量"
                }).drop("日期", axis=1)
            
            df = df.sort_values("date").reset_index(drop=True)
        
        elif source == "yfinance":
            # 全球市场通用（推荐美股/港股）
            ticker = yf.Ticker(ts_code)
            df = ticker.history(start=start_date, end=end_date)
            df = df.reset_index()
            df = df.rename(columns={
                "Date": "date", "Open": "开盘", "High": "最高", 
                "Low": "最低", "Close": "收盘", "Volume": "成交量"
            })
        
        # 统一字段格式 + 去空
        df = df[["date", "开盘", "最高", "最低", "收盘", "成交量"]].dropna()
        if len(df) == 0:
            raise ValueError("返回空数据，请检查股票代码/日期范围")
        return df
    
    except Exception as e:
        raise ValueError(f"数据获取失败：{str(e)}")

@tool
def predict_stock_trend(
    ts_code: str, 
    forecast_days: int = 5,
    source: str = "akshare"
) -> str:
    """
    股票短期趋势预测（LightGBM + 技术指标）
    :param ts_code: 股票代码（如600519.SH）
    :param forecast_days: 预测未来天数（默认5天）
    :param source: 数据源（仅支持akshare/yfinance）
    :return: 包含预测结果和投资建议的自然语言报告
    """
    try:
        # 1. 获取数据并生成技术指标
        df = get_stock_data(ts_code, start_date="2022-01-01", source=source)
        if len(df) < 100:
            return f"错误：{ts_code} 数据量不足，无法进行趋势预测（至少需要100个交易日）。"
        
        # 生成技术指标（趋势/动量/波动率）
        df["收益率"] = df["收盘"].pct_change()
        df["MA5"] = df["收盘"].rolling(window=5).mean()
        df["MA20"] = df["收盘"].rolling(window=20).mean()
        df["RSI"] = ta.momentum.RSIIndicator(df["收盘"], window=14).rsi()
        df["MACD"] = ta.trend.MACD(df["收盘"]).macd()
        df["布林带宽度"] = (df["最高"].rolling(20).max() - df["最低"].rolling(20).min()) / df["收盘"]
        df["成交量变化"] = df["成交量"].pct_change()
        
        # 特征工程：滞后特征 + 技术指标
        feature_cols = []
        # 滞后收益率（1-5天）
        for i in range(1, 6):
            df[f"收益率_滞后{i}"] = df["收益率"].shift(i)
            feature_cols.append(f"收益率_滞后{i}")
        # 技术指标特征
        feature_cols += ["MA5", "MA20", "RSI", "MACD", "布林带宽度", "成交量变化"]
        df = df.dropna()
        
        # 2. 构建标签：未来N天收益率（回归目标）
        df["标签"] = df["收益率"].shift(-forecast_days)
        df = df.dropna()
        
        # 3. 时间序列划分训练集/测试集
        X = df[feature_cols]
        y = df["标签"]
        tscv = TimeSeriesSplit(n_splits=3)
        train_idx, test_idx = list(tscv.split(X))[-1]  # 取最后一个折
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # 标准化特征
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 4. 训练LightGBM模型
        model = LGBMRegressor(
            n_estimators=100, learning_rate=0.05, 
            max_depth=5, random_state=42, verbose=-1
        )
        model.fit(X_train_scaled, y_train)
        
        # 5. 模型评估
        y_pred = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        # 方向准确率（预测涨跌是否正确）
        direction_accuracy = (np.sign(y_pred) == np.sign(y_test)).mean()
        
        # 6. 预测未来N天收益率
        last_features = scaler.transform(X.iloc[-1:])
        future_return = model.predict(last_features)[0]
        future_return_annual = (1 + future_return) ** 250 - 1  # 年化
        
        # 7. 生成投资建议
        trend = "上涨" if future_return > 0 else "下跌"
        confidence = "高" if direction_accuracy > 0.6 else "中" if direction_accuracy > 0.5 else "低"
        
        report = (
            f"**【{ts_code} 短期趋势预测报告】**\n"
            f"预测周期：未来{forecast_days}天\n"
            f"1. 模型预测收益率：{future_return:.2%}（年化：{future_return_annual:.2%}）\n"
            f"2. 趋势判断：{trend}（置信度：{confidence}）\n"
            f"3. 模型评估：测试集MAE={mae:.4f}，涨跌方向准确率={direction_accuracy:.2%}\n"
            f"4. 投资建议：\n"
            f"   - 若预测上涨（置信度高）：可小仓位布局，止损位设为当前价格的5%\n"
            f"   - 若预测下跌（置信度高）：建议减仓或观望，避免追高\n"
            f"   - 置信度中/低：建议结合基本面分析，不依赖单一模型结论"
        )
        return report
    
    except Exception as e:
        return f"趋势预测失败：{str(e)}"

@tool
def evaluate_stock_risk(
    ts_code: str,
    confidence_level: float = 0.99,
    source: str = "akshare"
) -> str:
    """
    股票风险评估（GARCH波动率 + VaR + 最大回撤）
    :param ts_code: 股票代码
    :param confidence_level: VaR置信水平（默认99%）
    :param source: 数据源（仅支持akshare/yfinance）
    :return: 风险评估报告
    """
    try:
        # 1. 获取数据并计算对数收益率
        df = get_stock_data(ts_code, start_date="2022-01-01", source=source)
        df["对数收益率"] = np.log(df["收盘"] / df["收盘"].shift(1))
        df = df.dropna()
        returns = df["对数收益率"].values * 100  # 转换为百分比

        # 2. GARCH(1,1)模型预测波动率
        am = arch_model(returns, vol="Garch", p=1, q=1)
        res = am.fit(update_freq=5, disp="off")
        # 预测未来1天条件波动率
        forecasts = res.forecast(horizon=1)
        daily_vol = np.sqrt(forecasts.variance.iloc[-1, 0]) / 100  # 转换为小数
        annual_vol = daily_vol * np.sqrt(250)  # 年化波动率

        # 3. 计算VaR（Value at Risk）
        z_score = norm.ppf(1 - confidence_level)  # 分位数
        daily_var = z_score * daily_vol  # 日VaR
        annual_var = daily_var * np.sqrt(250)  # 年化VaR

        # 4. 计算最大回撤
        df["累计收益"] = (1 + df["对数收益率"]).cumprod()
        df["峰值"] = df["累计收益"].cummax()
        df["回撤"] = (df["累计收益"] - df["峰值"]) / df["峰值"]
        max_drawdown = df["回撤"].min()  # 最大回撤（负数）

        # 5. 风险等级划分
        if annual_vol < 0.15:
            risk_level = "低风险"
        elif annual_vol < 0.3:
            risk_level = "中风险"
        else:
            risk_level = "高风险"

        # 6. 生成风险评估报告（修复：风险等级匹配文案）
        risk_suggestion = (
            f"   - {risk_level}标的：保守型投资者建议配置比例≤{10 if risk_level=='低风险' else 5 if risk_level=='中风险' else 2}%\n"
            f"   - {risk_level}标的：建议分批建仓，设置严格止损（如最大回撤的50%）\n"
            f"   - 极端风险防范：避免单一标的仓位过高，分散配置降低非系统性风险"
        )
        
        report = (
            f"**【{ts_code} 风险评估报告】**\n"
            f"1. 波动率分析（GARCH模型）：\n"
            f"   - 日波动率：{daily_vol:.2%}，年化波动率：{annual_vol:.2%}\n"
            f"   - 风险等级：{risk_level}\n"
            f"2. 下行风险（VaR）：\n"
            f"   - {confidence_level*100}%置信度下，单日最大损失：{daily_var:.2%}\n"
            f"   - 年化VaR：{annual_var:.2%}（极端市场下的潜在损失）\n"
            f"3. 历史最大回撤：{max_drawdown:.2%}\n"
            f"4. 风险建议：\n"
            f"{risk_suggestion}"
        )
        return report
    
    except Exception as e:
        return f"风险评估失败：{str(e)}"

@tool
def analyze_stock_valuation(
    ts_code: str,
    source: str = "akshare"
) -> str:
    """
    股票估值分析（仅基于AKShare实时行情的PE/PB/行业，简化功能避免报错）
    :param ts_code: 股票代码（仅支持A股：600519.SH/000001.SZ）
    :param source: 数据源（仅支持akshare）
    :return: 估值分析报告
    """
    try:
        # 仅支持AKShare + A股
        if source != "akshare" or not ts_code.endswith((".SH", ".SZ")):
            return f"估值分析失败：仅支持AKShare数据源的A股股票（如600519.SH）"
        
        # 提取纯代码（剥离.SH/.SZ）
        symbol = ts_code.split(".")[0] if "." in ts_code else ts_code
        
        # 1. 获取实时行情数据（稳定接口）
        spot_df = ak.stock_zh_a_spot_em()
        # 严格过滤：确保代码匹配 + 数据非空
        stock_spot = spot_df[spot_df["代码"] == symbol].reset_index(drop=True)
        if stock_spot.empty:
            return f"估值分析失败：未查询到{ts_code}的实时行情数据"
        
        # 2. 安全提取字段（避免索引越界）
        industry = stock_spot["所属行业"].iloc[0] if "所属行业" in stock_spot.columns and len(stock_spot) > 0 else "未知"
        pe_ttm = stock_spot["市盈率-动态"].iloc[0] if "市盈率-动态" in stock_spot.columns and len(stock_spot) > 0 else np.nan
        pb = stock_spot["市净率"].iloc[0] if "市净率" in stock_spot.columns and len(stock_spot) > 0 else np.nan
        
        # 过滤异常值（inf/-inf → nan）
        pe_ttm = np.nan if np.isinf(pe_ttm) else pe_ttm
        pb = np.nan if np.isinf(pb) else pb
        
        # 3. 简化估值判断（仅基于PB，PE因行业均值拿不到暂不判断）
        val_status = []
        if not np.isnan(pb):
            if pb < 1:
                val_status.append("PB<1（破净，安全边际较高）")
            elif pb > 5:
                val_status.append("PB>5（估值偏高）")
            else:
                val_status.append("PB处于合理区间（1-5）")
        if not np.isnan(pe_ttm):
            val_status.append(f"动态PE：{pe_ttm:.2f}（无行业均值对比）")
        
        # 4. 生成简化版估值报告
        report = (
            f"**【{ts_code} 估值分析报告】**\n"
            f"所属行业：{industry}\n"
            f"1. 核心估值指标：\n"
            f"   - 动态市盈率（PE）：{pe_ttm:.2f}\n"
            f"   - 市净率（PB）：{pb:.2f}\n"
            f"2. 估值判断：{'; '.join(val_status) if val_status else '暂无有效估值数据'}\n"
            f"3. 投资建议：\n"
            f"   - PB>5：需警惕估值回调风险，建议等待合理价格\n"
            f"   - PB<1：若基本面无恶化，安全边际较高，可逢低关注\n"
            f"   - 估值判断仅基于实时行情，建议结合基本面进一步分析"
        )
        return report
    
    except Exception as e:
        return f"估值分析失败：{str(e)}"

@tool
def comprehensive_stock_analysis(
    ts_code: str,
    risk_preference: str = "稳健",  # 保守/稳健/激进
    source: str = "akshare"
) -> str:
    """
    股票综合分析（趋势+风险+估值），适配不同风险偏好的投资建议
    :param ts_code: 股票代码
    :param risk_preference: 风险偏好（保守/稳健/激进）
    :param source: 数据源（仅支持akshare）
    :return: 综合分析报告
    """
    try:
        # 1. 调用子分析函数
        trend_report = predict_stock_trend(ts_code, source=source)
        risk_report = evaluate_stock_risk(ts_code, source=source)
        val_report = analyze_stock_valuation(ts_code, source=source)
        
        # 2. 提取关键指标（修复年化波动率提取逻辑）
        try:
            # 正确提取：拆分后取数值，转换为小数（15.62 → 0.1562）
            annual_vol_str = risk_report.split("年化波动率：")[1].split("%")[0].strip()
            annual_vol = float(annual_vol_str) / 100  # 转换为小数
        except:
            annual_vol = 0.2  # 默认值
        
        # 提取估值状态和趋势（简化：不依赖估值判断字段）
        trend = "上涨" if "上涨" in trend_report else "下跌" if "下跌" in trend_report else "震荡"
        
        # 3. 按风险偏好生成建议（修复波动率显示）
        if risk_preference == "保守":
            core_suggestion = (
                f"**保守型投资者建议：**\n"
                f"   - 波动率要求：仅考虑年化波动率<15%的标的（当前{annual_vol*100:.2f}%）\n"
                f"   - 估值要求：优先选择PB<1的破净标的\n"
                f"   - 仓位控制：单一标的仓位≤5%，总股票仓位≤30%\n"
                f"   - 操作策略：若当前标的波动率>15%，建议回避；严格止损"
            )
        elif risk_preference == "稳健":
            core_suggestion = (
                f"**稳健型投资者建议：**\n"
                f"   - 波动率要求：年化波动率15%-30%（当前{annual_vol*100:.2f}%）\n"
                f"   - 估值要求：PB处于1-5的合理区间\n"
                f"   - 仓位控制：单一标的仓位≤10%，总股票仓位≤50%\n"
                f"   - 操作策略：若趋势向上且估值合理，可分批建仓；若趋势向下，减仓至5%以下"
            )
        elif risk_preference == "激进":
            core_suggestion = (
                f"**激进型投资者建议：**\n"
                f"   - 波动率要求：可接受年化波动率>30%（当前{annual_vol*100:.2f}%）\n"
                f"   - 估值要求：可容忍短期高PB，但需结合趋势判断\n"
                f"   - 仓位控制：单一标的仓位≤20%，总股票仓位≤80%\n"
                f"   - 操作策略：若趋势向上，可重仓布局；设置10%止损，避免极端损失"
            )
        else:
            core_suggestion = "**风险偏好不明确：** 建议先明确风险承受能力，保守型优先保值，激进型可承担高波动追求高收益。"
        
        # 4. 整合报告
        comprehensive_report = (
            f"**【{ts_code} 综合投资分析报告（{risk_preference}型）】**\n"
            f"========== 趋势分析 ==========\n{trend_report}\n"
            f"========== 风险评估 ==========\n{risk_report}\n"
            f"========== 估值分析 ==========\n{val_report}\n"
            f"========== 个性化建议 ==========\n{core_suggestion}"
        )
        return comprehensive_report
    
    except Exception as e:
        return f"综合分析失败：{str(e)}"
