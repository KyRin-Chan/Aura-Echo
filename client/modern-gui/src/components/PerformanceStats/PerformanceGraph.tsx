
import { ResponsiveContainer, ComposedChart, Area, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import CustomTooltip from './PerformanceGraphTooltip';
import { ChartDataPoint } from './PerformanceStatsCard';
import { useThemeContext } from '../../context/ThemeContext';
import { t } from '../../locales';

interface PerformanceGraphProps {
  chartData: ChartDataPoint[];
}

function PerformanceGraph({ chartData }: PerformanceGraphProps) {
  // ---------------- State ----------------
  const { theme } = useThemeContext();

  // Accent colors dynamically matching the current selected theme
  const strokeColor = 'var(--md-sys-color-primary)';
  const refLineColor = 'var(--md-sys-color-secondary)';
  const gridColor = theme === 'dark' ? 'rgba(255, 255, 255, 0.08)' : 'rgba(0, 0, 0, 0.06)';

  // ---------------- Render ----------------

  return (
    <div className="absolute inset-0 p-1">
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={chartData} margin={{ top: 15, right: 10, left: -20, bottom: 0 }}>
          <defs>
            <linearGradient id="colorPerf" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={strokeColor} stopOpacity={0.3}/>
              <stop offset="95%" stopColor={strokeColor} stopOpacity={0}/>
            </linearGradient>
          </defs>
          
          <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />
          
          <XAxis
            dataKey="timestamp"
            tickFormatter={(unixTime: number) => new Date(unixTime).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
            stroke="var(--text-tertiary)"
            tick={{ fontSize: 9, fill: "var(--text-secondary)" }}
            axisLine={{ stroke: "var(--border-primary)", strokeWidth: 1 }}
            tickLine={{ stroke: "var(--border-primary)" }}
          />
          
          <YAxis
            label={{ value: 'ms', angle: -90, position: 'insideLeft', fill: 'var(--text-secondary)', fontSize: 10, offset: 10 }}
            stroke="var(--text-tertiary)"
            tick={{ fontSize: 9, fill: "var(--text-secondary)" }}
            tickFormatter={(value: number) => String(Math.round(value))}
            axisLine={{ stroke: "var(--border-primary)", strokeWidth: 1 }}
            tickLine={{ stroke: "var(--border-primary)" }}
            domain={[0, (dataMax: number) => Math.max(Math.ceil(dataMax * 1.2 / 50) * 50, 150)]}
          />
          
          <Tooltip content={<CustomTooltip />} />
          
          <Legend 
            wrapperStyle={{ fontSize: "11px", paddingTop: "5px", color: "var(--text-primary)" }} 
            verticalAlign="top"
            height={32}
          />

          {/* Area Chart with dynamic gradient fill */}
          <Area
            type="monotone"
            dataKey="perfTimeValue"
            stroke={strokeColor}
            strokeWidth={2}
            fill="url(#colorPerf)"
            activeDot={{ r: 5, strokeWidth: 0, fill: strokeColor }}
            name={t('mainProcessTimeLabel')}
            isAnimationActive={false}
          />

          {/* Chunk Time Reference Line */}
          <Line
            type="monotone"
            dataKey="chunkTime"
            stroke={refLineColor}
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
}

export default PerformanceGraph;
