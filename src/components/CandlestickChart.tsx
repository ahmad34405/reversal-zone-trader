
import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import { useQuery } from '@tanstack/react-query';

interface CandleData {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface ChartData {
  symbol: string;
  timeframe: string;
  data: CandleData[];
  count: number;
}

interface CandlestickChartProps {
  symbol: string;
  className?: string;
}

const fetchCandleData = async (symbol: string): Promise<ChartData> => {
  const response = await fetch(`http://localhost:8000/api/market-data/${symbol}?timeframe=1m&days=1`);
  if (!response.ok) {
    throw new Error('Failed to fetch candle data');
  }
  return response.json();
};

const CandlestickChart: React.FC<CandlestickChartProps> = ({ symbol, className = "" }) => {
  const [dimensions, setDimensions] = useState({ width: 800, height: 400 });

  const { data, error, refetch } = useQuery({
    queryKey: ['candleData', symbol],
    queryFn: () => fetchCandleData(symbol),
    refetchInterval: 60000, // Auto-update every 60 seconds
    retry: 3,
  });

  // Handle window resize for responsive design
  useEffect(() => {
    const handleResize = () => {
      const container = document.getElementById('chart-container');
      if (container) {
        const rect = container.getBoundingClientRect();
        setDimensions({
          width: Math.max(300, rect.width - 40),
          height: Math.max(300, window.innerHeight * 0.4)
        });
      }
    };

    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  if (error) {
    console.error('Error fetching candle data:', error);
    return (
      <div className="flex items-center justify-center h-96 text-white">
        <div className="text-center">
          <p className="text-red-400 mb-2">Failed to load chart data</p>
          <button 
            onClick={() => refetch()} 
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="flex items-center justify-center h-96 text-white">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-white"></div>
      </div>
    );
  }

  // Prepare data for Plotly candlestick chart
  const candleData = data.data.map(candle => ({
    x: new Date(candle.timestamp),
    open: candle.open,
    high: candle.high,
    low: candle.low,
    close: candle.close,
    volume: candle.volume
  }));

  const plotData = [
    {
      x: candleData.map(d => d.x),
      open: candleData.map(d => d.open),
      high: candleData.map(d => d.high),
      low: candleData.map(d => d.low),
      close: candleData.map(d => d.close),
      type: 'candlestick' as const,
      name: symbol,
      increasing: {
        line: { color: '#00ff88' },
        fillcolor: '#00ff88'
      },
      decreasing: {
        line: { color: '#ff4444' },
        fillcolor: '#ff4444'
      },
      line: { width: 1 },
    }
  ];

  const layout = {
    title: {
      text: `${symbol} Candlestick Chart`,
      font: { color: 'white', size: 16 },
    },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: 'white' },
    xaxis: {
      title: 'Time',
      gridcolor: 'rgba(255,255,255,0.1)',
      color: 'rgba(255,255,255,0.7)',
      rangeslider: { visible: false },
      type: 'date' as const,
    },
    yaxis: {
      title: 'Price ($)',
      gridcolor: 'rgba(255,255,255,0.1)',
      color: 'rgba(255,255,255,0.7)',
    },
    margin: { l: 50, r: 50, t: 50, b: 50 },
    showlegend: false,
  };

  const config = {
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
    responsive: true,
  };

  return (
    <div id="chart-container" className={`w-full ${className}`}>
      <Plot
        data={plotData}
        layout={{
          ...layout,
          width: dimensions.width,
          height: dimensions.height,
        }}
        config={config}
        style={{ width: '100%', height: '100%' }}
        useResizeHandler={true}
      />
      <div className="text-xs text-blue-200 mt-2 text-center">
        Last updated: {new Date().toLocaleTimeString()} | Data points: {data.count}
      </div>
    </div>
  );
};

export default CandlestickChart;
