import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ScatterChart, Scatter } from 'recharts';
import { Activity, TrendingUp, TrendingDown, Target, Database, BarChart3 } from 'lucide-react';
import CandlestickChart from './CandlestickChart';

// Simulated trading data
const generateCandleData = () => {
  const data = [];
  let price = 100;
  for (let i = 0; i < 200; i++) {
    const open = price;
    const high = open + Math.random() * 2;
    const low = open - Math.random() * 2;
    const close = low + Math.random() * (high - low);
    data.push({
      timestamp: new Date(Date.now() - (200 - i) * 60000).toLocaleTimeString(),
      open: parseFloat(open.toFixed(2)),
      high: parseFloat(high.toFixed(2)),
      low: parseFloat(low.toFixed(2)),
      close: parseFloat(close.toFixed(2)),
      volume: Math.floor(Math.random() * 1000) + 100
    });
    price = close;
  }
  return data;
};

const generateTradingSignals = () => {
  return [
    { id: 1, type: 'MAJOR_TREND_INTERSECTION', price: 98.45, confidence: 0.85, status: 'ACTIVE' },
    { id: 2, type: 'SUPPORT_RESISTANCE', price: 101.23, confidence: 0.72, status: 'PENDING' },
    { id: 3, type: 'MINOR_TREND_INTERSECTION', price: 99.87, confidence: 0.68, status: 'EXECUTED' }
  ];
};

const generateTradeHistory = () => {
  return [
    { id: 1, entry: 98.45, exit: 99.12, pnl: 0.67, duration: '3m', type: 'LONG', timestamp: '10:23:45' },
    { id: 2, entry: 101.23, exit: 100.89, pnl: -0.34, duration: '5m', type: 'SHORT', timestamp: '10:18:12' },
    { id: 3, entry: 99.87, exit: 100.45, pnl: 0.58, duration: '3m', type: 'LONG', timestamp: '10:15:23' }
  ];
};

const TradingBot = () => {
  console.log('TradingBot component rendering');
  
  const [candleData, setCandleData] = useState(generateCandleData());
  const [signals, setSignals] = useState(generateTradingSignals());
  const [trades, setTrades] = useState(generateTradeHistory());
  const [isRunning, setIsRunning] = useState(false);
  const [currentPrice, setCurrentPrice] = useState(100.45);
  const [selectedSymbol, setSelectedSymbol] = useState('BTC-USD');
  const [stats, setStats] = useState({
    totalTrades: 15,
    winRate: 73.3,
    totalPnL: 12.45,
    activeSignals: 2
  });

  // Simulate real-time price updates
  useEffect(() => {
    if (isRunning) {
      const interval = setInterval(() => {
        setCurrentPrice(prev => {
          const change = (Math.random() - 0.5) * 0.1;
          return parseFloat((prev + change).toFixed(2));
        });
      }, 1000);
      return () => clearInterval(interval);
    }
  }, [isRunning]);

  const toggleBot = () => {
    setIsRunning(!isRunning);
  };

  const SignalCard = ({ signal }) => (
    <Card className="mb-4">
      <CardContent className="p-4">
        <div className="flex justify-between items-center">
          <div>
            <h4 className="font-semibold">{signal.type.replace(/_/g, ' ')}</h4>
            <p className="text-sm text-muted-foreground">Target: ${signal.price}</p>
            <p className="text-sm">Confidence: {(signal.confidence * 100).toFixed(1)}%</p>
          </div>
          <Badge variant={
            signal.status === 'ACTIVE' ? 'default' :
            signal.status === 'PENDING' ? 'secondary' : 'outline'
          }>
            {signal.status}
          </Badge>
        </div>
      </CardContent>
    </Card>
  );

  console.log('TradingBot render with selectedSymbol:', selectedSymbol);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-4xl font-bold text-white mb-2">
              Price Action Trading Bot
            </h1>
            <p className="text-blue-200">
              Advanced reversal strategy with trendline & level analysis
            </p>
          </div>
          <div className="flex items-center gap-4">
            <div className="text-right text-white">
              <p className="text-sm text-blue-200">Current Price</p>
              <p className="text-2xl font-bold">${currentPrice}</p>
            </div>
            <div>
              <select 
                value={selectedSymbol} 
                onChange={(e) => setSelectedSymbol(e.target.value)}
                className="px-3 py-2 bg-white/10 text-white rounded border border-white/20 mr-4"
              >
                <option value="BTC-USD">BTC-USD</option>
                <option value="ETH-USD">ETH-USD</option>
                <option value="AAPL">AAPL</option>
              </select>
            </div>
            <Button 
              onClick={toggleBot}
              size="lg"
              className={`${isRunning ? 'bg-red-600 hover:bg-red-700' : 'bg-green-600 hover:bg-green-700'} text-white`}
            >
              <Activity className="mr-2 h-4 w-4" />
              {isRunning ? 'Stop Bot' : 'Start Bot'}
            </Button>
          </div>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <Card className="bg-white/10 backdrop-blur-sm border-white/20">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-blue-200 text-sm font-medium">Total Trades</p>
                  <p className="text-3xl font-bold text-white">{stats.totalTrades}</p>
                </div>
                <BarChart3 className="h-8 w-8 text-blue-400" />
              </div>
            </CardContent>
          </Card>

          <Card className="bg-white/10 backdrop-blur-sm border-white/20">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-blue-200 text-sm font-medium">Win Rate</p>
                  <p className="text-3xl font-bold text-white">{stats.winRate}%</p>
                </div>
                <TrendingUp className="h-8 w-8 text-green-400" />
              </div>
            </CardContent>
          </Card>

          <Card className="bg-white/10 backdrop-blur-sm border-white/20">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-blue-200 text-sm font-medium">Total P&L</p>
                  <p className="text-3xl font-bold text-white">${stats.totalPnL}</p>
                </div>
                <TrendingDown className="h-8 w-8 text-blue-400" />
              </div>
            </CardContent>
          </Card>

          <Card className="bg-white/10 backdrop-blur-sm border-white/20">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-blue-200 text-sm font-medium">Active Signals</p>
                  <p className="text-3xl font-bold text-white">{stats.activeSignals}</p>
                </div>
                <Target className="h-8 w-8 text-yellow-400" />
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Main Content */}
        <Tabs defaultValue="chart" className="w-full">
          <TabsList className="grid w-full grid-cols-4 bg-white/10 backdrop-blur-sm">
            <TabsTrigger value="chart" className="text-white data-[state=active]:bg-white data-[state=active]:text-black">
              Price Chart
            </TabsTrigger>
            <TabsTrigger value="signals" className="text-white data-[state=active]:bg-white data-[state=active]:text-black">
              Live Signals
            </TabsTrigger>
            <TabsTrigger value="trades" className="text-white data-[state=active]:bg-white data-[state=active]:text-black">
              Trade History
            </TabsTrigger>
            <TabsTrigger value="analysis" className="text-white data-[state=active]:bg-white data-[state=active]:text-black">
              Technical Analysis
            </TabsTrigger>
          </TabsList>

          <TabsContent value="chart" className="mt-6">
            <Card className="bg-white/10 backdrop-blur-sm border-white/20">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <BarChart3 className="h-5 w-5" />
                  Real-Time Candlestick Chart with Trendlines
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-96">
                  <CandlestickChart symbol={selectedSymbol} className="h-full" />
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="signals" className="mt-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card className="bg-white/10 backdrop-blur-sm border-white/20">
                <CardHeader>
                  <CardTitle className="text-white flex items-center gap-2">
                    <Target className="h-5 w-5" />
                    Active Signals
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  {signals.map(signal => (
                    <SignalCard key={signal.id} signal={signal} />
                  ))}
                </CardContent>
              </Card>

              <Card className="bg-white/10 backdrop-blur-sm border-white/20">
                <CardHeader>
                  <CardTitle className="text-white flex items-center gap-2">
                    <Activity className="h-5 w-5" />
                    Signal Detection Status
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex justify-between items-center p-3 bg-white/5 rounded-lg">
                    <span className="text-white">Major Trendlines</span>
                    <Badge variant="outline" className="text-green-400 border-green-400">
                      {isRunning ? 'Analyzing' : 'Standby'}
                    </Badge>
                  </div>
                  <div className="flex justify-between items-center p-3 bg-white/5 rounded-lg">
                    <span className="text-white">Minor Trendlines</span>
                    <Badge variant="outline" className="text-green-400 border-green-400">
                      {isRunning ? 'Analyzing' : 'Standby'}
                    </Badge>
                  </div>
                  <div className="flex justify-between items-center p-3 bg-white/5 rounded-lg">
                    <span className="text-white">Support/Resistance</span>
                    <Badge variant="outline" className="text-green-400 border-green-400">
                      {isRunning ? 'Analyzing' : 'Standby'}
                    </Badge>
                  </div>
                  <div className="flex justify-between items-center p-3 bg-white/5 rounded-lg">
                    <span className="text-white">Intersection Points</span>
                    <Badge variant="outline" className="text-yellow-400 border-yellow-400">
                      {isRunning ? '3 Found' : 'Standby'}
                    </Badge>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="trades" className="mt-6">
            <Card className="bg-white/10 backdrop-blur-sm border-white/20">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <Database className="h-5 w-5" />
                  Recent Trade History
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b border-white/20">
                        <th className="text-left text-white p-3">Time</th>
                        <th className="text-left text-white p-3">Type</th>
                        <th className="text-left text-white p-3">Entry</th>
                        <th className="text-left text-white p-3">Exit</th>
                        <th className="text-left text-white p-3">P&L</th>
                        <th className="text-left text-white p-3">Duration</th>
                      </tr>
                    </thead>
                    <tbody>
                      {trades.map(trade => (
                        <tr key={trade.id} className="border-b border-white/10">
                          <td className="text-blue-200 p-3">{trade.timestamp}</td>
                          <td className="p-3">
                            <Badge variant={trade.type === 'LONG' ? 'default' : 'secondary'}>
                              {trade.type}
                            </Badge>
                          </td>
                          <td className="text-white p-3">${trade.entry}</td>
                          <td className="text-white p-3">${trade.exit}</td>
                          <td className={`p-3 ${trade.pnl > 0 ? 'text-green-400' : 'text-red-400'}`}>
                            {trade.pnl > 0 ? '+' : ''}${trade.pnl}
                          </td>
                          <td className="text-blue-200 p-3">{trade.duration}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="analysis" className="mt-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card className="bg-white/10 backdrop-blur-sm border-white/20">
                <CardHeader>
                  <CardTitle className="text-white">Technical Levels</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="p-3 bg-white/5 rounded-lg">
                      <h4 className="text-white font-semibold">Major Support</h4>
                      <p className="text-blue-200">$98.45 (3 touches)</p>
                    </div>
                    <div className="p-3 bg-white/5 rounded-lg">
                      <h4 className="text-white font-semibold">Major Resistance</h4>
                      <p className="text-blue-200">$102.15 (4 touches)</p>
                    </div>
                    <div className="p-3 bg-white/5 rounded-lg">
                      <h4 className="text-white font-semibold">Major Trendline</h4>
                      <p className="text-blue-200">Ascending (200 candles)</p>
                    </div>
                    <div className="p-3 bg-white/5 rounded-lg">
                      <h4 className="text-white font-semibold">Minor Trendline</h4>
                      <p className="text-blue-200">Descending (30 candles)</p>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card className="bg-white/10 backdrop-blur-sm border-white/20">
                <CardHeader>
                  <CardTitle className="text-white">Strategy Performance</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={[
                        { day: 'Mon', pnl: 2.3 },
                        { day: 'Tue', pnl: 1.8 },
                        { day: 'Wed', pnl: 3.2 },
                        { day: 'Thu', pnl: 2.7 },
                        { day: 'Fri', pnl: 2.5 }
                      ]}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                        <XAxis dataKey="day" stroke="rgba(255,255,255,0.7)" />
                        <YAxis stroke="rgba(255,255,255,0.7)" />
                        <Tooltip 
                          contentStyle={{ 
                            backgroundColor: 'rgba(0,0,0,0.8)', 
                            border: '1px solid rgba(255,255,255,0.2)',
                            borderRadius: '8px',
                            color: 'white'
                          }} 
                        />
                        <Line type="monotone" dataKey="pnl" stroke="#34d399" strokeWidth={3} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};

export default TradingBot;
