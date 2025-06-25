
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
  console.log('Fetching candle data for symbol:', symbol);
  try {
    const response = await fetch(`http://localhost:8000/api/market-data/${symbol}?timeframe=1m&days=1`);
    console.log('Response status:', response.status);
    if (!response.ok) {
      throw new Error(`Failed to fetch candle data: ${response.status} ${response.statusText}`);
    }
    const data = await response.json();
    console.log('Received data:', data);
    return data;
  } catch (error) {
    console.error('Fetch error:', error);
    throw error;
  }
};

const CandlestickChart: React.FC<CandlestickChartProps> = ({ symbol, className = "" }) => {
  const [dimensions, setDimensions] = useState({ width: 800, height: 400 });

  console.log('CandlestickChart rendering with symbol:', symbol);

  const { data, error, isLoading, refetch } = useQuery({
    queryKey: ['candleData', symbol],
    queryFn: () => fetchCandleData(symbol),
    refetchInterval: 60000,
    retry: false, // Disable retry to see errors immediately
    enabled: true, // Ensure query is enabled
  });

  console.log('Query state:', { data, error, isLoading });

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
          <p className="text-red-300 text-sm mb-4">{error.message}</p>
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

  if (isLoading) {
    console.log('Loading chart data...');
    return (
      <div className="flex items-center justify-center h-96 text-white">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-white mx-auto mb-2"></div>
          <p className="text-blue-200">Loading chart data...</p>
        </div>
      </div>
    );
  }

  if (!data || !data.data || data.data.length === 0) {
    console.log('No data available');
    return (
      <div className="flex items-center justify-center h-96 text-white">
        <div className="text-center">
          <p className="text-yellow-400 mb-2">No chart data available</p>
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

  console.log('Processing chart data, count:', data.data.length);

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

  console.log('Rendering Plotly chart');

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
