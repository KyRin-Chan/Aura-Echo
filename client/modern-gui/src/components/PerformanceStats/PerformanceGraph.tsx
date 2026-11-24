
import React from 'react';
import { ResponsiveContainer, ComposedChart, Area, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import CustomTooltip from './PerformanceGraphTooltip';
import { ChartDataPoint } from './PerformanceStatsCard';
import { useThemeContext } from '../../context/ThemeContext';
import { t } from '../../locales';

interface PerformanceGraphProps {
  chartData: ChartDataPoint[];
}

// Accent colors dynamically matching the current selected theme
const STROKE_COLOR = 'var(--md-sys-color-primary)';
const REF_LINE_COLOR = 'var(--md-sys-color-secondary)';

const CHART_MARGIN = { top: 15, right: 10, left: -20, bottom: 0 };
const AXIS_TICK = { fontSize: 9, fill: 'var(--md-sys-color-on-surface-variant)' };
const AXIS_LINE = { stroke: 'var(--md-sys-color-outline-variant)', strokeWidth: 1 };
const TICK_LINE = { stroke: 'var(--md-sys-color-outline-variant)' };
const Y_AXIS_LABEL = { value: 'ms', angle: -90, position: 'insideLeft' as const, fill: 'var(--md-sys-color-on-surface-variant)', fontSize: 10, offset: 10 };
const Y_AXIS_DOMAIN: [number, (dataMax: number) => number] = [0, (dataMax: number) => Math.max(Math.ceil((dataMax * 1.2) / 50) * 50, 150)];
const LEGEND_WRAPPER_STYLE = { fontSize: '11px', paddingTop: '5px', color: 'var(--md-sys-color-on-surface)' };
const AREA_ACTIVE_DOT = { r: 5, strokeWidth: 0, fill: STROKE_COLOR };

const formatXAxisTick = (unixTime: number) =>
  new Date(unixTime).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

const formatYAxisTick = (value: number) => String(Math.round(value));

const TOOLTIP_CONTENT = <CustomTooltip />;

const PerformanceGraph = React.memo(function PerformanceGraph({ chartData }: PerformanceGraphProps) {
  // ---------------- State ----------------
  const { theme } = useThemeContext();

  const gridColor = theme === 'dark' ? 'rgba(255, 255, 255, 0.08)' : 'rgba(0, 0, 0, 0.06)';

  // ---------------- Render ----------------

  return (
    <div className="absolute inset-0 p-1">
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={chartData} margin={CHART_MARGIN}>
          <defs>
            <linearGradient id="colorPerf" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={STROKE_COLOR} stopOpacity={0.3}/>
              <stop offset="95%" stopColor={STROKE_COLOR} stopOpacity={0}/>
            </linearGradient>
          </defs>
          
          <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />
          
          <XAxis
            dataKey="timestamp"
            tickFormatter={formatXAxisTick}
            stroke="var(--md-sys-color-outline)"
            tick={AXIS_TICK}
            axisLine={AXIS_LINE}
            tickLine={TICK_LINE}
          />
          
          <YAxis
            label={Y_AXIS_LABEL}
            stroke="var(--md-sys-color-outline)"
            tick={AXIS_TICK}
            tickFormatter={formatYAxisTick}
            axisLine={AXIS_LINE}
            tickLine={TICK_LINE}
            domain={Y_AXIS_DOMAIN}
          />
          
          <Tooltip content={TOOLTIP_CONTENT} />
          
          <Legend 
            wrapperStyle={LEGEND_WRAPPER_STYLE} 
            verticalAlign="top"
            height={32}
          />

          {/* Area Chart with dynamic gradient fill */}
          <Area
            type="monotone"
            dataKey="perfTimeValue"
            stroke={STROKE_COLOR}
            strokeWidth={2}
            fill="url(#colorPerf)"
            activeDot={AREA_ACTIVE_DOT}
            name={t('mainProcessTimeLabel')}
            isAnimationActive={false}
          />

          {/* Chunk Time Reference Line */}
          <Line
            type="monotone"
            dataKey="chunkTime"
            stroke={REF_LINE_COLOR}
            strokeWidth={1.5}
            strokeDasharray="4 4"
            dot={false}
            name={t('chunkSizeLimitLabel')}
            activeDot={false}
            isAnimationActive={false}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
});

export default PerformanceGraph;
