
declare module 'react-plotly.js' {
  import { Component } from 'react';
  
  interface PlotParams {
    data: Array<any>;
    layout?: any;
    frames?: Array<any>;
    config?: any;
    revision?: number;
    onInitialized?: (figure: any, graphDiv: HTMLElement) => void;
    onUpdate?: (figure: any, graphDiv: HTMLElement) => void;
    onPurge?: (figure: any, graphDiv: HTMLElement) => void;
    onError?: (error: any) => void;
    debug?: boolean;
    style?: React.CSSProperties;
    className?: string;
    useResizeHandler?: boolean;
    divId?: string;
  }
  
  export default class Plot extends Component<PlotParams> {}
}
