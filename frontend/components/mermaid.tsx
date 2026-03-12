'use client';

import { useEffect, useId, useState } from 'react';
import { useTheme } from 'next-themes';

export function Mermaid({ chart }: { chart: string }) {
  const id = useId().replace(/:/g, '-');
  const [svg, setSvg] = useState('');
  const { resolvedTheme } = useTheme();
  const isDark = resolvedTheme === 'dark';

  useEffect(() => {
    let cancelled = false;

    async function render() {
      const mermaid = (await import('mermaid')).default;

      mermaid.initialize({
        startOnLoad: false,
        theme: 'base',
        fontFamily: 'Inter, system-ui, sans-serif',
        fontSize: 14,
        flowchart: {
          curve: 'basis',
          padding: 20,
          htmlLabels: true,
          nodeSpacing: 60,
          rankSpacing: 50,
        },
        block: {
          padding: 12,
        },
        themeVariables: isDark
          ? {
              // Dark mode — deep navy + amber accents
              background: '#0f1629',
              primaryColor: '#1e293b',
              primaryTextColor: '#e2e8f0',
              primaryBorderColor: '#f59e0b',
              secondaryColor: '#172033',
              secondaryTextColor: '#e2e8f0',
              secondaryBorderColor: '#334155',
              tertiaryColor: '#1a2332',
              tertiaryTextColor: '#e2e8f0',
              tertiaryBorderColor: '#475569',
              lineColor: '#475569',
              textColor: '#e2e8f0',
              mainBkg: '#1e293b',
              nodeBorder: '#f59e0b',
              clusterBkg: '#111827',
              clusterBorder: '#334155',
              titleColor: '#f59e0b',
              edgeLabelBackground: '#0f1629',
              nodeTextColor: '#e2e8f0',
              // Subgraph label styling
              fontSize: '14px',
            }
          : {
              // Light mode — warm whites + amber accents
              background: '#ffffff',
              primaryColor: '#fffbeb',
              primaryTextColor: '#1e293b',
              primaryBorderColor: '#d97706',
              secondaryColor: '#fef3c7',
              secondaryTextColor: '#1e293b',
              secondaryBorderColor: '#fbbf24',
              tertiaryColor: '#fff7ed',
              tertiaryTextColor: '#1e293b',
              tertiaryBorderColor: '#e5e7eb',
              lineColor: '#9ca3af',
              textColor: '#374151',
              mainBkg: '#fffbeb',
              nodeBorder: '#d97706',
              clusterBkg: '#fefce8',
              clusterBorder: '#fde68a',
              titleColor: '#92400e',
              edgeLabelBackground: '#ffffff',
              nodeTextColor: '#1e293b',
              fontSize: '14px',
            },
      });

      try {
        const { svg: rendered } = await mermaid.render(
          `mermaid-${id}`,
          chart,
        );
        if (!cancelled) setSvg(rendered);
      } catch {
        if (!cancelled) setSvg('');
      }
    }

    render();
    return () => {
      cancelled = true;
    };
  }, [chart, id, isDark]);

  if (!svg) {
    return (
      <div
        className={`my-8 flex items-center justify-center rounded-2xl border p-12 text-sm ${
          isDark
            ? 'border-white/10 bg-[#0f1629] text-slate-400'
            : 'border-amber-200/60 bg-amber-50/50 text-amber-700'
        }`}
      >
        <div className="flex items-center gap-3">
          <div className="h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
          Rendering diagram...
        </div>
      </div>
    );
  }

  return (
    <div
      className={`not-prose my-8 flex justify-center overflow-x-auto rounded-2xl border p-8 transition-colors ${
        isDark
          ? 'border-white/10 bg-[#0f1629] shadow-lg shadow-black/20'
          : 'border-amber-200/60 bg-gradient-to-br from-amber-50/50 to-orange-50/30 shadow-sm'
      }`}
      dangerouslySetInnerHTML={{ __html: svg }}
    />
  );
}
