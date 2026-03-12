import type { BaseLayoutProps } from 'fumadocs-ui/layouts/shared';
import { Activity } from 'lucide-react';

export const gitConfig = {
  user: 'abdifatah',
  repo: 'fl-thesis-project',
  branch: 'main',
};

export function baseOptions(): BaseLayoutProps {
  return {
    nav: {
      title: (
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-amber-500 to-orange-600 flex items-center justify-center shadow-lg shadow-amber-500/25 dark:shadow-amber-500/40">
            <Activity className="w-4 h-4 text-white" />
          </div>
          <span className="font-heading font-bold tracking-tight text-slate-900 dark:text-white text-base">
            FedLearn<span className="text-amber-600 dark:text-amber-500 font-medium ml-1">Docs</span>
          </span>
        </div>
      ),
      transparentMode: 'top',
    },
    links: [
      {
        text: 'Home',
        url: '/',
        active: 'nested-url',
      },
      {
        text: 'Dashboard',
        url: '/server',
        active: 'nested-url',
      },
    ],
  };
}
